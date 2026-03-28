#!/usr/bin/env python3
"""Bayesian network with inference."""
from collections import defaultdict
import random
class BayesianNetwork:
    def __init__(self): self.nodes={};self.parents={};self.cpds={}
    def add_node(self,name,parents=None,cpd=None):
        self.nodes[name]=True;self.parents[name]=parents or []
        if cpd: self.cpds[name]=cpd
    def probability(self,node,value,evidence):
        cpd=self.cpds[node];parents=self.parents[node]
        if not parents: return cpd.get(value,0)
        parent_vals=tuple(evidence.get(p) for p in parents)
        return cpd.get(parent_vals,{}).get(value,0)
    def sample(self,evidence=None):
        if evidence is None: evidence={}
        sample=dict(evidence)
        order=self._topological_sort()
        for node in order:
            if node in sample: continue
            probs={}
            for val in self._values(node):
                probs[val]=self.probability(node,val,sample)
            r=random.random();cumsum=0
            for val,p in probs.items():
                cumsum+=p
                if cumsum>=r: sample[node]=val;break
        return sample
    def infer(self,query,evidence,n_samples=10000):
        counts=defaultdict(int);total=0
        for _ in range(n_samples):
            s=self.sample(evidence)
            total+=1;counts[s[query]]+=1
        return {k:v/total for k,v in counts.items()}
    def _values(self,node):
        cpd=self.cpds[node]
        if isinstance(list(cpd.values())[0],dict):
            vals=set()
            for d in cpd.values(): vals.update(d.keys())
            return vals
        return cpd.keys()
    def _topological_sort(self):
        visited=set();order=[]
        def dfs(n):
            if n in visited: return
            visited.add(n)
            for p in self.parents[n]: dfs(p)
            order.append(n)
        for n in self.nodes: dfs(n)
        return order
if __name__=="__main__":
    random.seed(42)
    bn=BayesianNetwork()
    bn.add_node("Rain",cpd={True:0.2,False:0.8})
    bn.add_node("Sprinkler",parents=["Rain"],cpd={(True,):{True:0.01,False:0.99},(False,):{True:0.4,False:0.6}})
    bn.add_node("WetGrass",parents=["Rain","Sprinkler"],cpd={(True,True):{True:0.99,False:0.01},(True,False):{True:0.8,False:0.2},(False,True):{True:0.9,False:0.1},(False,False):{True:0.0,False:1.0}})
    result=bn.infer("Rain",{"WetGrass":True},n_samples=50000)
    print(f"P(Rain|WetGrass=T) ≈ {result.get(True,0):.3f}")
    print("Bayesian network OK")
