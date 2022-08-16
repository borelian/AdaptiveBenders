# %%
import csv
from distutils.sysconfig import customize_compiler
import numpy as np
import time
import re
import gurobipy as gp
from gurobipy import GRB
from numpy.random import default_rng


# %%
# Facility location with bernoulli demand
class FLBD:
    def __init__(self, dataFile):
        with open(dataFile) as fileData:
            reader = csv.reader(fileData, delimiter=' ', skipinitialspace=True)
            for line in reader:
                if reader.line_num == 1:
                    self.numFacilities = int(line[0])
                    self.numCustomers = int(line[1])
                    self.openCost = np.zeros(self.numFacilities)
                    self.assignCost = np.zeros((self.numFacilities,self.numCustomers))
                    self.probs = np.zeros(self.numCustomers)
                    self.minCapac = np.zeros(self.numFacilities, dtype=int)
                    self.maxCapac = np.zeros(self.numFacilities, dtype=int)
                elif reader.line_num == 2:
                    for i in range(self.numFacilities):
                        self.openCost[i] = float(line[i])
                elif reader.line_num <= self.numFacilities+2:
                    for j in range(self.numCustomers):
                        self.assignCost[reader.line_num-3,j] = float(line[j])
                elif reader.line_num == self.numFacilities+3:
                    self.unSatCost = float(line[0])
                elif reader.line_num == self.numFacilities+4:
                    for j in range(self.numCustomers):
                        self.probs[j] = float(line[j])
                elif reader.line_num == self.numFacilities+5:
                    for i in range(self.numFacilities):
                        self.minCapac[i] = int(line[i])
                elif reader.line_num == self.numFacilities+6:
                    for i in range(self.numFacilities):
                        self.maxCapac[i] = int(line[i])


 # %%
class SFLBD(FLBD):
    def __init__(self, dataFile):
        FLBD.__init__(self, dataFile)
        self.nscen = 0
        self.demandScen = None

                    
    def genScenarios(self, numScenarios, seed=0):
        rng = default_rng(seed=seed)
        self.nscen = numScenarios
        self.demandScen = ( rng.random((self.numCustomers,self.nscen))
            < np.matmul( self.probs.reshape((self.numCustomers,1)) 
            ,np.ones((1,self.nscen))) )

    def evaluateSolution(self, solX, assign, outSample=False, nOSScen = 0):
        if outSample == False:
            samples = self.demandScen
        else:
            rng = default_rng()
            samples = ( rng.random((self.numCustomers,nOSScen))
            < np.matmul( self.probs.reshape((self.numCustomers,1)) 
            ,np.ones((1,nOSScen))) )



    def solveDE(self, timeLimit = 86400):
        start_time = time.time()
        m = gp.Model("DetEquiv")
        print("Primal part formulation")
        #Defining variables
        X = m.addVars(range(self.numFacilities), vtype=GRB.BINARY, name="X")
        Y = m.addVars(range(self.numFacilities), range(self.numCustomers), vtype=GRB.BINARY,name='Y')

        # assign each customer
        m.addConstrs(gp.quicksum(Y[i,j] for i in range(self.numFacilities)) == 1 for j in range(self.numCustomers))

        # only if open facility
        m.addConstrs(Y[i,j] <= X[i] for i in range(self.numFacilities) for j in range(self.numCustomers))

        # min assignment
        m.addConstrs(gp.quicksum(Y[i,j] for j in range(self.numCustomers)) >= self.minCapac[i]*X[i] for i in range(self.numFacilities) )

        #### Second stage
        print("Second part formulation")

        W = m.addVars(range(self.numFacilities), range(self.numCustomers), range(self.nscen), lb=0, name='w')

        m.addConstrs(W[i,j,s] <= Y[i,j] for i in range(self.numFacilities) for j in range(self.numCustomers) for s in range(self.nscen))

        m.addConstrs(gp.quicksum(W[i,j,s] for i in range(self.numFacilities)) <= self.demandScen[j,s] for j in range(self.numCustomers) for s in range(self.nscen))

        m.addConstrs(gp.quicksum(W[i,j,s] for j in range(self.numCustomers)) <= self.maxCapac[i] * X[i] for i in range(self.numFacilities) for s in range(self.nscen))

        m.setObjective(
            gp.quicksum(self.openCost[i]*X[i] for i in range(self.numFacilities)) + float(1/self.nscen) * gp.quicksum((self.assignCost[i,j] - self.unSatCost)*W[i,j,s] for i in range(self.numFacilities) for j in range(self.numCustomers) for s in range(self.nscen)), GRB.MINIMIZE)
        print("Updating and solving")
        m.update()
        m.Params.timeLimit = timeLimit
        m.Params.Threads = 4
        self.m = m
        m.optimize()
        if m.status == GRB.OPTIMAL:
            solX = np.array([int(X[i].x) for i in range(self.numFacilities)])
            solY = np.zeros(self.numCustomers, dtype=int)

            for j in range(self.numCustomers):
                solY[j] = np.argmax([Y[i,j].x for i in range(self.numFacilities)])
            print("FinalReport: %d %f %f %f %d %d %d %f"
            % (0,m.ObjVal,m.ObjVal,0,0,0,self.nscen,time.time()-start_time))
            return m.ObjVal, solX, solY
        else:
            raise Exception("Gurobi solStatus "+str(m.status))        

    # def formulateMP(self):
    #     self.MP =  gp.Model("MasterProblem")       
    #     #Defining variables
    #     X = self.MP.addVars(range(self.numTerminal), lb=0, ub=self.Boundx, name="X")
    #     resources = self.MP.addConstrs(gp.quicksum(self.Resources[k][i] * X[i] for i in range(self.numTerminal)) <= self.CapResources[k] for k in range(self.numResources))
    #     theta = self.MP.addVars(range(self.numScen), lb=-1e6, name="theta")
    #     self.MP.setObjective(
    #         gp.quicksum(self.Objx[i]*X[i] for i in range(self.numTerminal))
    #         + gp.quicksum(self.probs[s]*theta[s] for s in range(self.numScen))
    #     )
    #     self._varX = X
    #     self._varTheta = theta
    #     ## set parameters
    #     self.MP.Params.OutputFlag = 0
    #     self.MP.Params.Threads = 4

    # def formulateSP(self):
    #     self.SP =  gp.Model("SubProblemDual")       
    #     #Defining variables
    #     mu = self.SP.addVars(range(self.numCustomers), lb=0, name="mu")
    #     nu = self.SP.addVars(range(self.numTerminal), lb=0, name="nu")
    #     self.SP.addConstrs(mu[j] + nu[i] >= -self.Objy[(i,j)] for (i,j) in self.Arcs )
    #     self.SP.setObjective(0, GRB.MINIMIZE)
    #     ## Copy variable to acces them later
    #     self._varMu = mu
    #     self._varNu = nu
    #     ## set parameters
    #     self.SP.Params.InfUnbdInfo = 1
    #     self.SP.Params.OutputFlag = 0
    #     self.SP.Params.Threads = 4

    # # Set objective for mu variables given an x
    # def SPsetX(self, X):
    #     for i in range(self.numTerminal):
    #         self._varNu[i].obj = X[i]
    
    # # Set objective of lambda variables, solve the problem and returns solution
    # def SPsolve(self, Demand):
    #     for j in range(self.numCustomers):
    #         self._varMu[j].obj = Demand[j]
    #     self.SP.optimize()
    #     # Case optimum found (cannot be unbounded)
    #     if self.SP.status == GRB.OPTIMAL:
    #         solMu = np.array(self.SP.getAttr('x',self._varMu).values())
    #         solNu = np.array(self.SP.getAttr('x',self._varNu).values())
    #         return(1, -self.SP.ObjVal, solMu, solNu)
    #     else:
    #         raise Exception("Gurobi solStatus "+str(self.SP.status))

    # # Solve master problem
    # def MPsolve(self):
    #     self.MP.optimize()
    #     if self.MP.status == GRB.OPTIMAL:
    #         solX = np.array(self.MP.getAttr('x',self._varX).values())
    #         solT = np.array(self.MP.getAttr('x',self._varTheta).values())
    #         return(self.MP.ObjVal, solX, solT)
    #     else:
    #         raise Exception("Gurobi solStatus "+str(self.MP.status))

    # # Benders
    # def Benders(self, method = 'm', timeLimit = 86400, tol_optcut = 1e-5, tol_stopRgap = 1e-6, tol_stopAgap = 1e-6):
    #     ub = float('inf')
    #     lb = -float('inf')
    #     nOptCuts = 0
    #     nFeasCuts = 0
    #     partitionId = np.zeros(self.numScen)
    #     sizePartition = 1 
    #     if (method != 'a') and (method != 'p'):
    #         partitionId = np.arange(self.numScen)
    #         sizePartition = self.numScen
    #     start_time = time.time()
    #     dMuScen = np.zeros((self.numScen, self.numCustomers))
    #     it = 1
    #     while(time.time() - start_time < timeLimit):
    #         # Solve master
    #         (cLB,X,theta) = self.MPsolve()
    #         lb = max(lb,cLB)
    #         # fix X on the subproblem
    #         self.SPsetX(X)
    #         #current UB including X costs
    #         cUB = sum(self.Objx[i]*X[i] for i in range(self.numTerminal))
    #         #info for single cuts
    #         noInfCutAdded = True
    #         singleCutPartA = 0
    #         singleCutPartB = np.zeros(self.numCustomers)
    #         # info for adaptive cuts
    #         noCutAdded = True
    #         # Solve subproblem for each scenario
    #         # for s in range(self.numScen):
    #         #     (stat,objSP,dLambda, dMu) = self.SPsolve(self.scens[s])
    #         for p in range(sizePartition):
    #             # Warning: assuming equiprobable for numerical stability
    #             # if not it should be np.average()
    #             # demP = self.scens[s]
    #             demP = np.sum(self.scens[partitionId==p], axis=0)/np.sum(partitionId==p)
    #             probP = np.sum(partitionId==p)/self.numScen
    #             tmp = time.time()
    #             (stat,objSP,dMu, dNu) = self.SPsolve(demP)
    #             if stat == 1 :
    #                 if (method == 'm') or (method == 'a'):
    #                     #Optimality cut
    #                     partA = -sum(demP[j] * dMu[j] for j in range(self.numCustomers))
    #                     partB = -sum(dNu[i]*X[i] for i in range(self.numTerminal))
    #                     #print(partA, partB, (sum(theta[partitionId==p])/np.sum(partitionId==p)))
    #                     # Warning: assuming equiprobable for numerical stability
    #                     if partA+partB > (sum(theta[partitionId==p])/np.sum(partitionId==p)) + tol_optcut:
    #                         scen = np.extract(partitionId==p,range(self.numScen)).tolist()
    #                         self.MP.addConstr(
    #                             - gp.quicksum(demP[j] * dMu[j] for j in range(self.numCustomers))
    #                             - gp.quicksum(dNu[i]*self._varX[i] for i in range(self.numTerminal))
    #                             <= gp.quicksum(self._varTheta[s] for s in scen)/np.sum(partitionId==p))
    #                         nOptCuts += 1
    #                         noCutAdded = False
    #                 elif ((method == 's') or (method == 'p')):
    #                     singleCutPartA += -sum(demP[j] * dMu[j] for j in range(self.numCustomers))*probP
    #                     for i in range(self.numTerminal):
    #                         singleCutPartB[i] += -dNu[i]*probP
    #             else:
    #                 raise Exception("solStatus not Optimal")
    #             if (method != 'a') and (method != 'p') :
    #                 cUB += np.sum(self.probs[partitionId==p])*objSP 
    #             else:
    #                 cUB = float('inf')
    #         if ((method == 's') or (method == 'p')):
    #             if singleCutPartA + sum(singleCutPartB[i]*X[i] for i in range(self.numTerminal)) > sum(self.probs[s]*theta[s] for s in range(self.numScen)) + tol_optcut:
    #                 self.MP.addConstr(
    #                     singleCutPartA + gp.quicksum(singleCutPartB[i]*self._varX[i] for i in range(self.numTerminal))
    #                     <= sum(self.probs[s]*self._varTheta[s] for s in range(self.numScen)))                    
    #                 nOptCuts += 1
    #                 noCutAdded = False
    #         if ((method == 'a') or (method == 'p')) and noCutAdded:
    #             # No cut added. Check partition and compute UB
    #             cUB = sum(self.Objx[i]*X[i] for i in range(self.numTerminal))
    #             newSizePartition = sizePartition
    #             singleCutPartA = 0
    #             singleCutPartB = np.zeros(self.numTerminal)
    #             for p in range(sizePartition):
    #                 scen = np.extract(partitionId==p,range(self.numScen)).tolist()
    #                 for s in scen:
    #                     (stat,objSP,dMu, dNu) = self.SPsolve(self.scens[s])
    #                     dMuScen[s] = dMu
    #                     cUB += objSP*self.probs[s] 
    #                     singleCutPartA += -sum(self.scens[s,j] * dMu[j] for j in range(self.numCustomers))
    #                     for e in range(self.numTerminal):
    #                         singleCutPartB[e] += -dNu[e]
       
    #                 # Revise for repeated duals differences
    #                 (dualsUnique, inverse) = np.unique(dMuScen[scen,:],axis=0, return_inverse=True)
    #                 numSubsets = dualsUnique.shape[0]
    #                 if numSubsets > 1:
    #                     # we add new elements to the partition
    #                     partitionId[partitionId==p] = (inverse+newSizePartition)
    #                     # but rename the last one as the current one
    #                     partitionId[partitionId==(newSizePartition+numSubsets-1)] = p
    #                     newSizePartition += numSubsets -1
    #                     #print("Spliting %d into %d new subsets" % (p,numSubsets))
    #             print("Partition now has %d elements" % newSizePartition)    
    #             sizePartition = newSizePartition    
    #             self.dL = dMuScen
    #             self.part = partitionId

    #             #We add an extra optimality cut. I should be all scenarios feasible
    #             if (method == 'p'):
    #                 singleCutPartA = singleCutPartA/self.numScen
    #                 singleCutPartB = singleCutPartB/self.numScen

    #                 if singleCutPartA + sum(singleCutPartB[i]*X[i] for i in range(self.numTerminal)) > sum(self.probs[s]*theta[s] for s in range(self.numScen)) + tol_optcut:
    #                     self.MP.addConstr(
    #                         singleCutPartA + gp.quicksum(singleCutPartB[i]*self._varX[i] for i in range(self.numTerminal))
    #                         <= sum(self.probs[s]*self._varTheta[s] for s in range(self.numScen)))                    
    #                     nOptCuts += 1
    #                     noCutAdded = False                   


    #         #print("Iter %d: master = %f subp = %f gap = %f\n" % (it,cLB,cUB, -cUB/cLB+1))
    #         ub = min(ub, cUB)
    #         elap_time = time.time()
    #         #print("It=%d t=%f LB=%8.2f UB=%8.2f rgap=%8.2e nF=%d nO=%d"
    #         # % (it,elap_time-start_time,lb,ub,ub/(lb+1e-6)-1,nFeasCuts,nOptCuts))
    #         print("%d %8.2f %8.2f %8.2e %d %d %d %f"
    #          % (it,lb,ub,-ub/(lb-1e-6)+1,nFeasCuts,nOptCuts,sizePartition,elap_time-start_time))
    #         if (ub-lb < tol_stopRgap) or (-ub/(lb-1e-6)+1 < tol_stopRgap) :
    #             print("FinalReport: %d %f %f %f %d %d %d %f"
    #             % (it,lb,ub,-ub/(lb-1e-6)+1,nFeasCuts,nOptCuts,sizePartition,elap_time-start_time))
    #             break
    #         it += 1
    # def MPsolveFull(self,sizePartition,partitionId):
    #     m = gp.Model("GAPM")
    #     #Defining variables
    #     X = m.addVars(range(self.numTerminal), lb=0, ub=self.Boundx, name="X")
    #     Y = m.addVars(range(sizePartition), range(self.numArcs), lb=0, name='Y')
    #     resources = m.addConstrs(gp.quicksum(self.Resources[k][i] * X[i] for i in range(self.numTerminal)) <= self.CapResources[k] for k in range(self.numResources))
    #     cap = {}
    #     dem = {}
    #     demP = np.zeros((sizePartition,self.numCustomers))
    #     probP = np.zeros(sizePartition)
    #     for p in range(sizePartition):
    #         demP[p] = np.sum(self.scens[partitionId==p], axis=0)/np.sum(partitionId==p)
    #         probP[p] = np.sum(partitionId==p)/self.numScen

    #     for i in range(self.numTerminal):
    #         arcs = [k for k in range(self.numArcs) if self.Arcs[k][0] == i]
    #         cap[i] = m.addConstrs(gp.quicksum(Y[p,a] for a in arcs) <= X[i] for p in range(sizePartition))
    #     for j in range(self.numCustomers):
    #         arcs = [k for k in range(self.numArcs) if self.Arcs[k][1] == j]
    #         dem[j] = m.addConstrs(gp.quicksum(Y[p,a] for a in arcs) <= demP[p][j] for p in range(sizePartition)) 
    #     m.setObjective(
    #         gp.quicksum(self.Objx[i]*X[i] for i in range(self.numTerminal))
    #         + gp.quicksum(self.Objy[self.Arcs[a]]*probP[p]*Y[p,a] for p in range(sizePartition) for a in range(self.numArcs) ),
    #          GRB.MINIMIZE)
    #     m.update()
    #     m.Params.OutputFlag = 0
    #     m.Params.Threads = 4
    #     m.optimize()
    #     if m.status == GRB.OPTIMAL:
    #         solX = np.array(m.getAttr('x',X).values())
    #         return(m.ObjVal, solX)
    #     else:
    #         raise Exception("Gurobi solStatus "+str(m.status))

    # def GAPM(self, timeLimit = 86400, tol_optcut = 1e-5, tol_stopRgap = 1e-6, tol_stopAgap = 1e-6):
    #         ub = float('inf')
    #         lb = -float('inf')
    #         partitionId = np.zeros(self.numScen)
    #         sizePartition = 1 
    #         start_time = time.time()
    #         dMuScen = np.zeros((self.numScen, self.numCustomers))
    #         it = 1
    #         while(time.time() - start_time < timeLimit):
    #             # Solve master
    #             (cLB,X) = self.MPsolveFull(sizePartition,partitionId)
    #             #print("Iter %d: master = %f\n" % (it,cLB))
    #             lb = max(lb,cLB)
    #             # fix X on the subproblem
    #             self.SPsetX(X)
    #             #current UB including X costs
    #             cUB = sum(self.Objx[i]*X[i] for i in range(self.numTerminal))
    #             newSizePartition = sizePartition
    #             for p in range(sizePartition):
    #                 scen = np.extract(partitionId==p,range(self.numScen)).tolist()
    #                 for s in scen:
    #                     (stat,objSP,dMu, dNu) = self.SPsolve(self.scens[s])
    #                     dMuScen[s] = dMu
    #                     cUB += objSP*self.probs[s] 
    #                 # Revise for repeated duals differences
    #                 (dualsUnique, inverse) = np.unique(dMuScen[scen,:],axis=0, return_inverse=True)
    #                 numSubsets = dualsUnique.shape[0]
    #                 if numSubsets > 1:
    #                     # we add new elements to the partition
    #                     partitionId[partitionId==p] = (inverse+newSizePartition)
    #                     # but rename the last one as the current one
    #                     partitionId[partitionId==(newSizePartition+numSubsets-1)] = p
    #                     newSizePartition += numSubsets -1
    #                     #print("Spliting %d into %d new subsets" % (p,numSubsets))
    #             print("Partition now has %d elements" % newSizePartition)    
    #             sizePartition = newSizePartition    
    #             ub = min(ub, cUB)
    #             elap_time = time.time()
    #             #print("It=%d t=%f LB=%8.2f UB=%8.2f rgap=%8.2e nF=%d nO=%d"
    #             # % (it,elap_time-start_time,lb,ub,ub/(lb+1e-6)-1,nFeasCuts,nOptCuts))
    #             print("%d %8.2f %8.2f %8.2e %d %d %d %f"
    #             % (it,lb,ub,-ub/(lb-1e-6)+1,0,0,sizePartition,elap_time-start_time))
    #             if (ub-lb < tol_stopRgap) or (-ub/(lb-1e-6)+1 < tol_stopRgap) :
    #                 print("FinalReport: %d %f %f %f %d %d %d %f"
    #                 % (it,lb,ub,-ub/(lb-1e-6)+1,0,0,sizePartition,elap_time-start_time))
    #                 break
    #             it += 1


# # # %%
# corefile = '/Users/emoreno/Code/BendersGAPM/Electricity-Small/core_nospace.mps'
# stochfile = '/Users/emoreno/Code/BendersGAPM/Electricity-Small/stoch_nospace.mps'
# tmp = SCPP(corefile, stochfile)
# tmp.genScenarios(1000)
# tmp.formulateMP()
# tmp.formulateSP()
# tmp.Benders('p',timeLimit = 600)

# # %%
# corefile = '/Users/emoreno/Code/BendersGAPM/Electricity-Small/core_nospace.mps'
# stochfile = '/Users/emoreno/Code/BendersGAPM/Electricity-Small/stoch_nospace.mps'
# tmp = SCPP(corefile, stochfile)
# tmp.genScenarios(1000)
# tmp.formulateSP()
# tmp.GAPM()

# %%


# # %%
# # prob2 = SMCF('/Users/emoreno/Code/BendersGAPM-MCF/instances/r04.1.dow','/Users/emoreno/Code/BendersGAPM-MCF/instances/r04-0-100')
# # prob2.GAPM()
# # prob2.Benders('p', 500)
# # %%
# # dem = list(prob2.commDem.values())
# # prob2.SPsolve(dem)
# # # %%
# # prob2.SPsetX(np.ones(prob2.numArcs))
# # # %%
# # prob2.SPsolve(list(prob2.commDem.values()))
# # # %%
# # prob2.MPsolve()

# # # %%

# # # %%

# # %%

# %%
