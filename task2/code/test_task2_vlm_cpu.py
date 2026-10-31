import math
import unittest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config,Qwen3ForCausalLM
from task2_vlm import Connector,VisualLanguageModel,LoRALinear,example_causal_losses,calibrate_connector_scale
from task2_cpu_preflight import preprocess_image
from PIL import Image
import numpy as np

def example(before=2,after=3,answer=(7,8)):
    return {'before_ids':[1]*before,'after_ids':[2]*after,'response_ids':list(answer)+[3],
            'content_ids':list(answer),'eot_id':3}

def tiny():
    torch.manual_seed(6304)
    config=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
      num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0.,pad_token_id=0,
      bos_token_id=1,eos_token_id=3,tie_word_embeddings=True)
    backbone=Qwen3ForCausalLM(config)
    return VisualLanguageModel(backbone,Connector(language_dim=32))

class VLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(min(4,torch.get_num_threads()))
    def test_connector_shape_dtype_and_count(self):
        connector=Connector()
        self.assertEqual(sum(p.numel() for p in connector.parameters()),1837056)
        self.assertTrue(all(p.dtype==torch.float32 for p in connector.parameters()))
        self.assertEqual(connector(torch.randn(2,196,768)).shape,(2,196,1024))
    def test_example_weighted_loss_and_gradients(self):
        logits=torch.zeros(2,5,10,requires_grad=True)
        with torch.no_grad():logits[0,:,1]=4
        labels=torch.tensor([[-100,-100,1,1,-100],[-100,-100,-100,2,-100]])
        losses=example_causal_losses(logits,labels)
        expected=torch.stack([F.cross_entropy(logits[0,1:3],torch.tensor([1,1])),F.cross_entropy(logits[1,2:3],torch.tensor([2]))])
        torch.testing.assert_close(losses,expected)
        self.assertNotAlmostEqual(float(losses.mean().detach()),float(((losses[0]*2+losses[1])/3).detach()))
        losses.mean().backward();self.assertGreater(float(logits.grad.abs().sum()),0)
    def test_masks_visual_prompt_padding_and_eot(self):
        model=tiny();examples=[example(),example(after=1,answer=(9,))]
        batch=model.make_batch(examples,torch.randn(2,196,768))
        self.assertTrue((batch['labels'][0,:201]==-100).all())
        self.assertEqual(batch['labels'][0,201:].tolist(),[7,8,3])
        self.assertEqual(batch['labels'][1,199:].tolist(),[9,3,-100,-100,-100])
        self.assertEqual(batch['attention_mask'][1].sum().item(),201)
        content=model.make_batch(examples,torch.randn(2,196,768),content_only_labels=True)
        self.assertEqual(content['labels'][0,-1].item(),-100)
    def test_no_image_removes_exactly_visual_tokens_and_left_positions(self):
        model=tiny();examples=[example(),example(after=1,answer=(9,))]
        full=model.make_batch(examples,torch.randn(2,196,768),padding_side='left',include_response=False)
        empty=model.make_batch(examples,None,padding_side='left',include_response=False)
        self.assertEqual(full['inputs_embeds'].shape[1]-empty['inputs_embeds'].shape[1],196)
        for index in range(2):
            mask=empty['attention_mask'][index].bool()
            self.assertEqual(empty['position_ids'][index,mask].tolist(),list(range(int(mask.sum()))))
            self.assertTrue((empty['labels'][index]==-100).all())
    def test_connector_gradient_through_frozen_checkpointed_qwen(self):
        model=tiny();model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.train();losses,_=model([example(),example(answer=(5,))],torch.randn(2,196,768))
        losses.mean().backward()
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.backbone.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.connector.parameters()),0)
    def test_lora_targets_initial_identity_and_gradients(self):
        model=tiny().eval();features=torch.randn(1,196,768)
        with torch.no_grad():original=model([example()],features)[1].logits
        names=model.enable_lora()
        self.assertEqual(len(names),8)
        self.assertEqual({name.rsplit('.',1)[-1] for name in names},{'q_proj','k_proj','v_proj','o_proj'})
        with torch.no_grad():adapted=model([example()],features)[1].logits
        torch.testing.assert_close(original,adapted,rtol=0,atol=0)
        model.train();losses,_=model([example()],features);losses.mean().backward()
        self.assertGreater(sum(float(m.B.grad.abs().sum()) for m in model.modules() if isinstance(m,LoRALinear)),0)
        self.assertTrue(all(not p.requires_grad for n,p in model.backbone.named_parameters() if not n.endswith(('.A','.B'))))
    def test_deterministic_lora_initialization(self):
        first=tiny();second=tiny();first.enable_lora();torch.manual_seed(99);second.enable_lora()
        for (name,m),(other,n) in zip([(n,m) for n,m in first.named_modules() if isinstance(m,LoRALinear)],[(n,m) for n,m in second.named_modules() if isinstance(m,LoRALinear)]):
            self.assertEqual(name,other);torch.testing.assert_close(m.A,n.A,rtol=0,atol=0)
    def test_training_rms_calibration(self):
        model=tiny();features=torch.randn(128,196,768);tokens=torch.tensor([1,2,4,5])
        report=calibrate_connector_scale(model,features,tokens)
        with torch.no_grad():
            rms=model.connector(features).float().square().mean().sqrt()
            expected=model.backbone.get_input_embeddings()(tokens).float().square().mean().sqrt()
        self.assertGreater(report['scale'],0);torch.testing.assert_close(rms,expected,rtol=1e-5,atol=1e-7)
    def test_preprocessing_is_deterministic_and_rgb(self):
        image=Image.new('L',(32,64),128)
        a=preprocess_image(image);b=preprocess_image(image)
        self.assertEqual(a.shape,(3,224,224));self.assertEqual(a.dtype,np.float32)
        self.assertTrue(np.array_equal(a,b));self.assertTrue(np.array_equal(a[0],a[1]))

if __name__=='__main__':unittest.main()
