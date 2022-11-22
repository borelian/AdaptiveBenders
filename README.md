# Adaptive Benders

Source code and instances for the computational experiments of the paper "Benders Adaptive-Cuts Method for Two-Stage Stochastic Programs" by Cristian Ramírez-Pico, Ivana Ljubić and Eduardo Moreno. [arXiv:2203.00752](https://doi.org/10.48550/arXiv.2203.00752)

It applies different Benders methods and other optimization methods to solve three stochastic network flow problems. Each problem as its own class file and a run-file to execute an instance

Problems are:
- Stochastic Capacity Planning Problem
  - Class: `cpp.py`
  - Run file: `runCPP.py`
  - Instances: `CPP_instances/`
- Stochastic Multicommodity Flow Problem
  - Class: `smcf.py`
  - Run file: `runSMCF.py`
  - Instances: `SMCF_instances/`
- Facility Location with CVaR 
  - Class: `flcvar.py`
  - Run file: `runFLcvar.py`
  - Instances: `instancesFlcvar/`
 
It requires NumPy library and Gurobi (https://www.gurobi.com) as optimization solver.
