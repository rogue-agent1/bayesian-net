import argparse, json, itertools

class BayesNet:
    def __init__(self):
        self.nodes = {}
        self.parents = {}
        self.cpt = {}

    def add_node(self, name, parents, cpt):
        self.nodes[name] = True
        self.parents[name] = parents
        self.cpt[name] = cpt

    def prob(self, node, value, evidence):
        parents = self.parents[node]
        key = tuple(evidence.get(p, True) for p in parents)
        p_true = self.cpt[node].get(str(key), self.cpt[node].get(str(key).replace(" ", ""), 0.5))
        return p_true if value else (1 - p_true)

    def enumerate_ask(self, query, evidence):
        hidden = [n for n in self.nodes if n != query and n not in evidence]
        def enum_all(nodes, ev):
            if not nodes: return 1.0
            n = nodes[0]
            rest = nodes[1:]
            if n in ev:
                return self.prob(n, ev[n], ev) * enum_all(rest, ev)
            total = 0
            for val in [True, False]:
                ev2 = {**ev, n: val}
                total += self.prob(n, val, ev2) * enum_all(rest, ev2)
            return total
        order = list(self.nodes.keys())
        probs = {}
        for qval in [True, False]:
            ev = {**evidence, query: qval}
            probs[qval] = enum_all(order, ev)
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}

def main():
    p = argparse.ArgumentParser(description="Bayesian network")
    p.add_argument("file", nargs="?", help="JSON network definition")
    p.add_argument("--demo", action="store_true", help="Run rain/sprinkler demo")
    p.add_argument("--query")
    p.add_argument("--evidence", nargs="*")
    args = p.parse_args()
    if args.demo:
        bn = BayesNet()
        bn.add_node("rain", [], {"()": 0.2})
        bn.add_node("sprinkler", ["rain"], {"(True,)": 0.01, "(False,)": 0.4})
        bn.add_node("wet", ["sprinkler", "rain"], {
            "(True, True)": 0.99, "(True, False)": 0.9,
            "(False, True)": 0.8, "(False, False)": 0.0
        })
        print("P(rain | wet=True):", bn.enumerate_ask("rain", {"wet": True}))
        print("P(rain | wet=True, sprinkler=False):", bn.enumerate_ask("rain", {"wet": True, "sprinkler": False}))
    else:
        p.print_help()

if __name__ == "__main__":
    main()
