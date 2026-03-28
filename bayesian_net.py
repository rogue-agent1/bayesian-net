#!/usr/bin/env python3
"""bayesian_net - Bayesian network with inference."""
import argparse, sys, json
from collections import defaultdict

class BayesNet:
    def __init__(self):
        self.nodes = {}  # name -> {"parents": [], "cpt": {}}

    def add_node(self, name, parents, cpt):
        self.nodes[name] = {"parents": parents, "cpt": cpt}

    def prob(self, node, value, evidence):
        info = self.nodes[node]
        if info["parents"]:
            key = tuple(evidence.get(p) for p in info["parents"])
            dist = info["cpt"].get(str(key), info["cpt"].get(key, {True: 0.5, False: 0.5}))
        else:
            dist = info["cpt"]
        return dist.get(value, 0)

    def enumerate_ask(self, query, evidence):
        """P(query | evidence) by enumeration."""
        vars_order = list(self.nodes.keys())
        results = {}
        for val in [True, False]:
            evidence[query] = val
            results[val] = self._enumerate_all(vars_order[:], dict(evidence))
            del evidence[query]
        total = sum(results.values())
        return {k: v/total for k, v in results.items()} if total > 0 else results

    def _enumerate_all(self, variables, evidence):
        if not variables: return 1.0
        var = variables[0]
        rest = variables[1:]
        if var in evidence:
            return self.prob(var, evidence[var], evidence) * self._enumerate_all(rest, evidence)
        total = 0
        for val in [True, False]:
            evidence[var] = val
            total += self.prob(var, val, evidence) * self._enumerate_all(rest, evidence)
            del evidence[var]
        return total

def demo():
    bn = BayesNet()
    bn.add_node("Burglary", [], {True: 0.001, False: 0.999})
    bn.add_node("Earthquake", [], {True: 0.002, False: 0.998})
    bn.add_node("Alarm", ["Burglary", "Earthquake"], {
        str((True,True)): {True: 0.95, False: 0.05},
        str((True,False)): {True: 0.94, False: 0.06},
        str((False,True)): {True: 0.29, False: 0.71},
        str((False,False)): {True: 0.001, False: 0.999},
    })
    bn.add_node("John", ["Alarm"], {
        str((True,)): {True: 0.90, False: 0.10},
        str((False,)): {True: 0.05, False: 0.95},
    })
    bn.add_node("Mary", ["Alarm"], {
        str((True,)): {True: 0.70, False: 0.30},
        str((False,)): {True: 0.01, False: 0.99},
    })
    print("Burglary Alarm Network (Russell & Norvig)")
    print("\nP(Burglary | John=T, Mary=T):")
    result = bn.enumerate_ask("Burglary", {"John": True, "Mary": True})
    for k, v in result.items(): print(f"  {k}: {v:.6f}")
    print("\nP(Burglary | John=T, Mary=F):")
    result = bn.enumerate_ask("Burglary", {"John": True, "Mary": False})
    for k, v in result.items(): print(f"  {k}: {v:.6f}")

def main():
    p = argparse.ArgumentParser(description="Bayesian network")
    p.add_argument("--demo", action="store_true", default=True)
    a = p.parse_args()
    demo()

if __name__ == "__main__": main()
