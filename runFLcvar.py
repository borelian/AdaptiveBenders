# %%
from flcvar import *
import sys

# %%



if __name__ == "__main__":
    if (len(sys.argv) < 4) or (len(sys.argv) > 5):
        print("Usage: run.py instance num_scenarios method seed")
        print("Methods are:\n\ta: Adaptive Benders\n\ts: Single-cut Benders\n\tm: Multi-cut Benders\n\tp: Adaptive Benders Single-cut\n\tn: Generalized Adaptive Partition Method\n\tf: Deterministic Equivalent")
        sys.exit(-1)
    prob = SFLCVAR(sys.argv[1])
    if len(sys.argv) == 5:
        prob.genScenarios(int(sys.argv[2]), int(sys.argv[4]))
    else:
        prob.genScenarios(int(sys.argv[2]))
    prob.formulateMP()
    prob.formulateSP()

    if sys.argv[3] == 'a':
        obj, x, tau, y, t = prob.MPsolve(useMulticuts=True, useAdaptive=True)
    elif sys.argv[3] == 's':
        obj, x, tau, y, t = prob.MPsolve(useMulticuts=False, useAdaptive=False)
    elif sys.argv[3] == 'm':
        obj, x, tau, y, t = prob.MPsolve(useMulticuts=True, useAdaptive=False)
    elif sys.argv[3] == 'p':
        obj, x, tau, y, t = prob.MPsolve(useMulticuts=False, useAdaptive=True)
    elif sys.argv[3] == 'f':
        obj, x, tau, y, t = prob.solveDE()

    print("FinalSolution: Tau=", tau, " Solution:", x)


# %%
