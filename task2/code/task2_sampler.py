"""Seeded stateful pair-preserving curriculum sampling; no imposed mixture quotas."""
import copy
from collections import Counter
import numpy as np

class CurriculumSampler:
    def __init__(self,captions,presence,counting,natural,seed=6304):
        self.pools={'caption':captions,'presence':presence,'counting':counting,'natural':natural}
        assert all(self.pools.values())
        self.rng=np.random.default_rng(seed)
        self.order=np.concatenate([self.rng.permutation(len(captions)),self.rng.permutation(len(captions))]).tolist()
        self.caption_position=0;self.draws={1:0,2:0,3:0};self.counts={1:Counter(),2:Counter(),3:Counter()}
    def state_dict(self):
        return {'rng':copy.deepcopy(self.rng.bit_generator.state),'order':list(self.order),
                'caption_position':self.caption_position,'draws':dict(self.draws),
                'counts':{k:dict(v) for k,v in self.counts.items()}}
    def load_state_dict(self,state):
        self.rng.bit_generator.state=copy.deepcopy(state['rng']);self.order=list(state['order'])
        self.caption_position=state['caption_position'];self.draws=dict(state['draws'])
        self.counts={k:Counter(v) for k,v in state['counts'].items()}
    def batch(self,stage,size=16):
        assert stage in (1,2,3) and size%2==0
        if stage==1:
            end=self.caption_position+size;assert end<=len(self.order)
            rows=[self.pools['caption'][i] for i in self.order[self.caption_position:end]]
            self.caption_position=end;draws={'caption':size//2}
        else:
            names=('presence','counting','caption') if stage==2 else ('natural','presence','counting','caption')
            probabilities=(.5,.25,.25) if stage==2 else (.7,.1,.1,.1)
            rows=[];draws=Counter()
            for _ in range(size//2):
                name=names[int(self.rng.choice(len(names),p=probabilities))];draws[name]+=1
                pool=self.pools[name]
                if name=='caption':rows.extend(pool[int(i)] for i in self.rng.integers(len(pool),size=2))
                else:rows.extend(pool[int(self.rng.integers(len(pool)))]['examples'])
        assert len(rows)==size
        self.draws[stage]+=size//2;self.counts[stage].update(draws)
        return rows,dict(draws)
