# %%
from cpp import *
import sys

# %%



if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: run.py corefile stochfile num_scenarios method")
        print("Methods are:\n\ta: Adaptive Benders\n\ts: Single-cut Benders\n\tm: Multi-cut Benders\n\tp: Adaptice Benders Single-cut\n\tn: Generalized Adaptive Partition Method\n\tf: Deterministic Equivalent")
        sys.exit(-1)
    prob = SCPP(sys.argv[1], sys.argv[2])
    prob.genScenarios(int(sys.argv[3]))
    prob.formulateMP()
    prob.formulateSP()
    if sys.argv[4] in ['a','s','m','p']:
        prob.Benders(sys.argv[4], 86400)
    elif sys.argv[4] == 'n':
        prob.GAPM()
    elif sys.argv[4] == 'f':
        prob.solveDE()




# %%
