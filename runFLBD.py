# %%
from flbd import *
import sys

# %%



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: run.py instance num_scenarios method")
        print("Methods are:\n\ta: Adaptive Benders\n\ts: Single-cut Benders\n\tm: Multi-cut Benders\n\tp: Adaptive Benders Single-cut\n\tn: Generalized Adaptive Partition Method\n\tf: Deterministic Equivalent")
        sys.exit(-1)
    prob = SFLBD(sys.argv[1])
    prob.genScenarios(int(sys.argv[2]))
    prob.formulateMP()

    if sys.argv[3] == 'a':
        prob.MPsolve(useMulticuts=True, useBenders=True)
    elif sys.argv[3] == 's':
        prob.MPsolve(useMulticuts=False, useBenders=False)
    elif sys.argv[3] == 'm':
        prob.MPsolve(useMulticuts=True, useBenders=False)
    elif sys.argv[3] == 'p':
        prob.MPsolve(useMulticuts=False, useBenders=True)



# %%
