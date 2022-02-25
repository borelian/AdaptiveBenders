# %%
from smcf import *
import sys

# %%



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: runSMCF.py instance-file scenario-file method")
        print("Methods are:\n\ta: Adaptive Benders\n\ts: Single-cut Benders\n\tm: Multi-cut Benders\n\tp: Adaptice Benders Single-cut\n\tn: Generalized Adaptive Partition Method\n\tf: Deterministic Equivalent")
        sys.exit(-1)
    prob = SMCF(sys.argv[1], sys.argv[2])
    if sys.argv[3] in ['a','s','m','p']:
        prob.Benders(sys.argv[3], 86400)
    elif sys.argv[3] == 'n':
        prob.GAPM()
    elif sys.argv[3] == 'f':
        prob.solveDE()




# %%
