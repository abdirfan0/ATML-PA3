"""Task 2 visual connector, Qwen input formatting, attention LoRA, and loss.

No model downloads or training on import. Core equations follow the PA3 manual.
"""
from __future__ import annotations
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F

VERSION='pa3-task2-vlm-v1'
TARGETS={'q_proj','k_proj','v_proj','o_proj'}

class Connector(nn.Module):
    def __init__(self,vision_dim=768,language_dim=1024):
        super().__init__()
        self.fc1=nn.Linear(vision_dim,language_dim,dtype=torch.float32)
        self.fc2=nn.Linear(language_dim,language_dim,dtype=torch.float32)
        self.register_buffer('output_scale',torch.tensor(1.,dtype=torch.float32))
    def forward(self,features):
        return self.output_scale*self.fc2(F.gelu(self.fc1(features.to(self.fc1.weight.dtype))))

class LoRALinear(nn.Module):
    def __init__(self,base,name,rank=16,alpha=32,dropout=.05,seed=6304):
        super().__init__()
        assert isinstance(base,nn.Linear)
        self.base=base.requires_grad_(False);self.rank=rank;self.alpha=alpha
        self.dropout=nn.Dropout(dropout)
        local_seed=int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8],'big')%(2**63)
        generator=torch.Generator(device='cpu').manual_seed(local_seed)
        initial=torch.empty(rank,base.in_features,dtype=torch.float32)
        nn.init.kaiming_uniform_(initial,a=math.sqrt(5),generator=generator)
        self.A=nn.Parameter(initial.to(base.weight.device))
        self.B=nn.Parameter(torch.zeros(base.out_features,rank,dtype=torch.float32,device=base.weight.device))
    def forward(self,x):
        original=self.base(x)
        update=F.linear(F.linear(self.dropout(x).to(self.A.dtype),self.A),self.B)
        return original+(self.alpha/self.rank)*update.to(original.dtype)
    def factors(self):
        return {'A':self.A.detach().cpu().clone(),'B':self.B.detach().cpu().clone(),'scale':self.alpha/self.rank}


def install_attention_lora(model,seed=6304):
    names=[name for name,module in model.named_modules() if isinstance(module,nn.Linear) and name.rsplit('.',1)[-1] in TARGETS]
    assert names,'No prescribed attention projections found.'
    for name in names:
        parent,leaf=name.rsplit('.',1)
        setattr(model.get_submodule(parent),leaf,LoRALinear(model.get_submodule(name),name,seed=seed))
    return names


def encode_example(tokenizer,question,target=None,kind='answer'):
    """Insert visual embeddings after the user role header and before the question.

    The temporary marker is split out before tokenization; it never enters the
    model vocabulary. The same textual instructions are used when images are absent.
    """
    marker='PA3_VISUAL_INSERTION_POINT_6304'
    template=tokenizer.apply_chat_template([{'role':'user','content':marker}],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)
    assert template.count(marker)==1
    before,after=template.split(marker)
    def ids(text):return list(tokenizer(text,add_special_tokens=False)['input_ids'])
    content=ids(question)
    assert 0<len(content)<=80
    response=[]
    if target is not None:
        response=ids(target)
        assert 0<len(response)<=(64 if kind=='caption' else 16)
    eot=tokenizer.convert_tokens_to_ids('<|im_end|>')
    assert tokenizer.convert_ids_to_tokens(eot)=='<|im_end|>'
    return {'before_ids':ids(before),'after_ids':ids('\n'+question+after),
            'response_ids':response+([eot] if target is not None else []),
            'content_ids':response,'eot_id':eot,'question':question,'target':target}

class VisualLanguageModel(nn.Module):
    def __init__(self,backbone,connector=None):
        super().__init__()
        self.backbone=backbone.requires_grad_(False)
        self.connector=connector if connector is not None else Connector(language_dim=backbone.config.hidden_size)
    def enable_lora(self,seed=6304):
        assert not any(isinstance(module,LoRALinear) for module in self.backbone.modules()),'Adapters already installed.'
        return install_attention_lora(self.backbone,seed)
    def make_batch(self,examples,features=None,padding_side='right',include_response=True,content_only_labels=False):
        assert padding_side in ('left','right') and examples
        embed=self.backbone.get_input_embeddings()
        device=embed.weight.device
        if features is not None:
            assert features.ndim==3 and features.shape[0]==len(examples) and features.shape[1:]==(196,768)
            visual=self.connector(features.to(device)).to(embed.weight.dtype)
        else:visual=None
        sequences=[];labels=[]
        for index,example in enumerate(examples):
            before=torch.tensor(example['before_ids'],device=device,dtype=torch.long)
            after=torch.tensor(example['after_ids'],device=device,dtype=torch.long)
            pieces=[embed(before)]
            if visual is not None:pieces.append(visual[index])
            pieces.append(embed(after))
            prefix_length=sum(piece.shape[0] for piece in pieces)
            response=example['response_ids'] if include_response else []
            if response:pieces.append(embed(torch.tensor(response,device=device,dtype=torch.long)))
            sequences.append(torch.cat(pieces,dim=0))
            response_labels=list(response)
            if content_only_labels and response_labels:
                assert response_labels[-1]==example['eot_id']
                response_labels[-1]=-100  # Answer NLL excludes termination.
            labels.append(torch.tensor([-100]*prefix_length+response_labels,device=device,dtype=torch.long))
        longest=max(sequence.shape[0] for sequence in sequences)
        embeddings=[];batch_labels=[];masks=[]
        for sequence,label in zip(sequences,labels):
            length=sequence.shape[0];pad=longest-length
            empty=sequence.new_zeros((pad,sequence.shape[-1]));ignored=label.new_full((pad,),-100)
            active=torch.ones(length,device=device,dtype=torch.long);inactive=active.new_zeros(pad)
            if padding_side=='right':
                embeddings.append(torch.cat([sequence,empty]));batch_labels.append(torch.cat([label,ignored]));masks.append(torch.cat([active,inactive]))
            else:
                embeddings.append(torch.cat([empty,sequence]));batch_labels.append(torch.cat([ignored,label]));masks.append(torch.cat([inactive,active]))
        mask=torch.stack(masks)
        positions=(mask.cumsum(-1)-1).clamp_min(0).masked_fill(mask==0,0)
        return {'inputs_embeds':torch.stack(embeddings),'attention_mask':mask,'position_ids':positions,
                'labels':torch.stack(batch_labels)}
    def forward(self,examples,features=None,content_only_labels=False):
        batch=self.make_batch(examples,features,content_only_labels=content_only_labels)
        labels=batch.pop('labels')
        output=self.backbone(**batch,use_cache=False,return_dict=True)
        return example_causal_losses(output.logits,labels),output


def example_causal_losses(logits,labels):
    """Average supervised token losses within each example, returning one per row."""
    shifted=labels[:,1:]
    active=shifted!=-100
    counts=active.sum(dim=1)
    assert bool((counts>0).all()),'Each example must have at least one supervised token.'
    sample_indices=torch.arange(labels.shape[0],device=labels.device)[:,None].expand_as(shifted)[active]
    token_losses=F.cross_entropy(logits[:,:-1][active].float(),shifted[active],reduction='none')
    summed=torch.zeros(labels.shape[0],dtype=torch.float32,device=logits.device).scatter_add(0,sample_indices,token_losses)
    return summed/counts


def calibrate_connector_scale(model,training_features,training_token_ids):
    """One fixed training-only RMS match; records scale without changing other weights.

    Call once before baseline development evaluation and Stage 1. Tokens must be
    selected from the training prompts/captions, and features from saved 128 train IDs.
    This function never selects data; the runner must enforce saved split identity.
    """
    assert training_features.shape==(128,196,768)
    assert training_token_ids.numel()>0
    with torch.no_grad():
        model.connector.output_scale.fill_(1.)
        visual=model.connector(training_features).float()
        text=model.backbone.get_input_embeddings()(training_token_ids).float()
        visual_rms=visual.square().mean().sqrt();text_rms=text.square().mean().sqrt()
        assert bool(torch.isfinite(visual_rms)) and float(visual_rms)>0
        scale=text_rms/visual_rms
        assert bool(torch.isfinite(scale)) and float(scale)>0
        model.connector.output_scale.copy_(scale)
    return {'rule':'global RMS of training text embeddings / global RMS of initial projected training visual tokens; fixed scalar after calibration',
            'training_images':128,'visual_rms_before':float(visual_rms),'text_rms':float(text_rms),'scale':float(scale)}
