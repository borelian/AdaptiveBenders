# %%
import csv
import numpy as np
import time
import re
import gurobipy as gp
from gurobipy import GRB
from numpy.random import default_rng


# %%
class CPP:
    def __init__(self, coreFile):
        self.numTerminal = 20
        self.numCustomers = 50
        self.numResources = 10
        self.numScen = 0
        self.Objx = np.zeros(self.numTerminal) # Objective Terminal
        self.Objy = np.zeros((self.numTerminal, self.numCustomers))
        self.Arcs = set()
        self.Resources = np.zeros((self.numResources,self.numTerminal)) # FRCx rows
        self.CapResources = np.zeros(self.numResources) # SCCTx rows
        self.Demand = np.zeros(self.numCustomers) # SDCDx rows 
        self.Boundx = np.zeros(self.numTerminal)
        with open(coreFile) as fileData:
            inColumns = False
            inRHS = False
            inBounds = False
            reader = csv.reader(fileData, delimiter=' ', skipinitialspace=True)
            for line in reader:
                #print("Line: %d" % reader.line_num)
                if line[0] == 'COLUMNS':
                    inColumns = True
                elif (line[0] == 'RHS') and (len(line) == 1) :
                    inColumns = False
                    inRHS = True
                elif line[0] == 'BOUNDS':
                    inRHS = False
                    inBounds = True
                elif line[0] == 'ENDATA':
                    inBounds = False
                else:
                    if inColumns:
                        # Read resources columns
                        mtc = re.search(r'FCVT([0-9]*)', line[0])
                        if mtc:
                            i = int(mtc.group(1)) -1
                            if line[1] == 'OBJ':
                                self.Objx[i] = float(line[2])
                                #print("Objective %d is %f" % (i,self.Objx[i]))
                            elif re.search(r'FRC([0-9]*)',line[1]):
                                mt2 = re.search(r'FRC([0-9]*)',line[1])
                                j = int(mt2.group(1))
                                self.Resources[j][i] = float(line[2])
                                #print("Resource %d of %d is %f" % (j,i,self.Resources[j][i]))
                        #read y variables
                        mtc = re.search(r'SVT([0-9]*)D([0-9]*)', line[0])
                        if mtc:
                            i = int(mtc.group(1)) -1
                            j = int(mtc.group(2)) -1
                            self.Arcs.add((i,j))
                            if line[1] == 'OBJ':
                                self.Objy[i][j] = float(line[2])
                                #print("Objective y[%d][%d] is %f" % (i,j,self.Objx[i]))
                                            # Read resources rhs
                    elif inRHS:
                        mtc = re.search(r'FRC([0-9]*)', line[1])
                        if mtc:
                            i = int(mtc.group(1))
                            self.CapResources[i] = float(line[2])
                        mtc = re.search(r'SDCD([0-9]*)', line[1])
                        if mtc:
                            i = int(mtc.group(1)) - 1
                            self.Demand[i] = float(line[2])
                    elif inBounds:
                        mtc = re.search(r'FCVT([0-9]*)', line[2])
                        if mtc:
                            i = int(mtc.group(1))-1
                            self.Boundx[i] = float(line[3])
        self.numArcs = len(self.Arcs)
        self.Arcs = list(self.Arcs)
 # %%
class SCPP(CPP):
    def __init__(self, coreFile, stochFile):
        CPP.__init__(self, coreFile)
        self.numScen = 0
        self.probs = None
        self.scens = None

        self.origProb = [[] for i in range(self.numCustomers)]
        with open(stochFile) as fileData:
            reader = csv.reader(fileData, delimiter=' ', skipinitialspace=True)
            for line in reader:
                #print("Line: %d Line0=%s" % (reader.line_num, line[0]))
                if line[0] == 'RHS':
                    mtc = re.search(r'SDCD([0-9]*)', line[1])
                    i = int(mtc.group(1)) -1
                    self.origProb[i].append((float(line[2]),float(line[3])))
                    
    def genScenarios(self, numScenarios, seed=0):
        rng = default_rng(seed=seed)
        self.numScen = numScenarios
        self.scens = np.zeros((numScenarios, self.numCustomers))
        self.probs = np.ones(numScenarios)/numScenarios
        for s in range(numScenarios):
            for i in range(self.numCustomers):
                t = rng.random()
                cum = 0
                for k in range(len(self.origProb[i])):
                    cum += self.origProb[i][k][1]
                    if (t <= cum):
                        self.scens[s][i] = float(self.origProb[i][k][0])
                        break

    def solveDE(self, timeLimit = 86400):
        start_time = time.time()
        m = gp.Model("DetEquiv")
        #Defining variables
        X = m.addVars(range(self.numTerminal), lb=0, ub=self.Boundx, name="X")
        Y = m.addVars(range(self.numScen), range(self.numArcs), lb=0, name='Y')
        resources = m.addConstrs(gp.quicksum(self.Resources[k][i] * X[i] for i in range(self.numTerminal)) <= self.CapResources[k] for k in range(self.numResources))
        cap = {}
        dem = {}
        for i in range(self.numTerminal):
            arcs = [k for k in range(self.numArcs) if self.Arcs[k][0] == i]
            cap[i] = m.addConstrs(gp.quicksum(Y[s,a] for a in arcs) <= X[i] for s in range(self.numScen))
        for j in range(self.numCustomers):
            arcs = [k for k in range(self.numArcs) if self.Arcs[k][1] == j]
            dem[j] = m.addConstrs(gp.quicksum(Y[s,a] for a in arcs) <= self.scens[s][j] for s in range(self.numScen)) 
        m.setObjective(
            gp.quicksum(self.Objx[i]*X[i] for i in range(self.numTerminal))
            + gp.quicksum(self.Objy[self.Arcs[a]]*self.probs[s]*Y[s,a] for s in range(self.numScen) for a in range(self.numArcs) ),
             GRB.MINIMIZE)
        m.update()
        m.Params.timeLimit = timeLimit
        m.Params.Threads = 4
        m.optimize()
        if m.status == GRB.OPTIMAL:
            print("FinalReport: %d %f %f %f %d %d %d %f"
            % (0,m.ObjVal,m.ObjVal,0,0,0,self.numScen,time.time()-start_time))
        else:
            raise Exception("Gurobi solStatus "+str(m.status))        

    def formulateMP(self):
        self.MP =  gp.Model("MasterProblem")       
        #Defining variables
        X = self.MP.addVars(range(self.numTerminal), lb=0, ub=self.Boundx, name="X")
        resources = self.MP.addConstrs(gp.quicksum(self.Resources[k][i] * X[i] for i in range(self.numTerminal)) <= self.CapResources[k] for k in range(self.numResources))
        theta = self.MP.addVars(range(self.numScen), lb=-1e6, name="theta")
        self.MP.setObjective(
            gp.quicksum(self.Objx[i]*X[i] for i in range(self.numTerminal))
            + gp.quicksum(self.probs[s]*theta[s] for s in range(self.numScen))
        )
        self._varX = X
        self._varTheta = theta
        ## set parameters
        self.MP.Params.OutputFlag = 0
        self.MP.Params.Threads = 4

    def formulateSP(self):
        self.SP =  gp.Model("SubProblemDual")       
        #Defining variables
        mu = self.SP.addVars(range(self.numCustomers), lb=0, name="mu")
        nu = self.SP.addVars(range(self.numTerminal), lb=0, name="nu")
        self.SP.addConstrs(mu[j] + nu[i] >= -self.Objy[(i,j)] for (i,j) in self.Arcs )
        self.SP.setObjective(0, GRB.MINIMIZE)
        ## Copy variable to acces them later
        self._varMu = mu
        self._varNu = nu
        ## set parameters
        self.SP.Params.InfUnbdInfo = 1
        self.SP.Params.OutputFlag = 0
        self.SP.Params.Threads = 4

    # Set objective for mu variables given an x
    def SPsetX(self, X):
        for i in range(self.numTerminal):
            self._varNu[i].obj = X[i]
    
    # Set objective of lambda variables, solve the problem and returns solution
    def SPsolve(self, Demand):
        for j in range(self.numCustomers):
            self._varMu[j].obj = Demand[j]
        self.SP.optimize()
        # Case optimum found (cannot be unbounded)
        if self.SP.status == GRB.OPTIMAL:
            solMu = np.array(self.SP.getAttr('x',self._varMu).values())
            solNu = np.array(self.SP.getAttr('x',self._varNu).values())
            return(1, -self.SP.ObjVal, solMu, solNu)
        else:
            raise Exception("Gurobi solStatus "+str(self.SP.status))

    # Solve master problem
    def MPsolve(self):
        self.MP.optimize()
        if self.MP.status == GRB.OPTIMAL:
            solX = np.array(self.MP.getAttr('x',self._varX).values())
            solT = np.array(self.MP.getAttr('x',self._varTheta).values())
            return(self.MP.ObjVal, solX, solT)
        else:
            raise Exception("Gurobi solStatus "+str(self.MP.status))

    # Benders
    def Benders(self, method = 'm', timeLimit = 86400, tol_optcut = 1e-5, tol_stopRgap = 1e-6, tol_stopAgap = 1e-6):
        ub = float('inf')
        lb = -float('inf')
        nOptCuts = 0
        nFeasCuts = 0
        partitionId = np.zeros(self.numScen)
        sizePartition = 1 
        if (method != 'a') and (method != 'p'):
            partitionId = np.arange(self.numScen)
            sizePartition = self.numScen
        start_time = time.time()
        dMuScen = np.zeros((self.numScen, self.numCustomers))
        it = 1
        while(time.time() - start_time < timeLimit):
            # Solve master
            (cLB,X,theta) = self.MPsolve()
            lb = max(lb,cLB)
            # fix X on the subproblem
            self.SPsetX(X)
            #current UB including X costs
            cUB = sum(self.Objx[i]*X[i] for i in range(self.numTerminal))
            #info for single cuts
            noInfCutAdded = True
            singleCutPartA = 0
            singleCutPartB = np.zeros(self.numCustomers)
            # info for adaptive cuts
            noCutAdded = True
            # Solve subproblem for each scenario
            # for s in range(self.numScen):
            #     (stat,objSP,dLambda, dMu) = self.SPsolve(self.scens[s])
            for p in range(sizePartition):
                # Warning: assuming equiprobable for numerical stability
                # if not it should be np.average()
                # demP = self.scens[s]
                demP = np.sum(self.scens[partitionId==p], axis=0)/np.sum(partitionId==p)
                probP = np.sum(partitionId==p)/self.numScen
                tmp = time.time()
                (stat,objSP,dMu, dNu) = self.SPsolve(demP)
                if stat == 1 :
                    if (method == 'm') or (method == 'a'):
                        #Optimality cut
                        partA = -sum(demP[j] * dMu[j] for j in range(self.numCustomers))
                        partB = -sum(dNu[i]*X[i] for i in range(self.numTerminal))
                        #print(partA, partB, (sum(theta[partitionId==p])/np.sum(partitionId==p)))
                        # Warning: assuming equiprobable for numerical stability
                        if partA+partB > (sum(theta[partitionId==p])/np.sum(partitionId==p)) + tol_optcut:
                            scen = np.extract(partitionId==p,range(self.numScen)).tolist()
                            self.MP.addConstr(
                                - gp.quicksum(demP[j] * dMu[j] for j in range(self.numCustomers))
                                - gp.quicksum(dNu[i]*self._varX[i] for i in range(self.numTerminal))
                                <= gp.quicksum(self._varTheta[s] for s in scen)/np.sum(partitionId==p))
                            nOptCuts += 1
                            noCutAdded = False
                    elif ((method == 's') or (method == 'p')):
                        singleCutPartA += -sum(demP[j] * dMu[j] for j in range(self.numCustomers))*probP
                        for i in range(self.numTerminal):
                            singleCutPartB[i] += -dNu[i]*probP
                else:
                    raise Exception("solStatus not Optimal")
                if (method != 'a') and (method != 'p') :
                    cUB += np.sum(self.probs[partitionId==p])*objSP 
                else:
                    cUB = float('inf')
            if ((method == 's') or (method == 'p')):
                if singleCutPartA + sum(singleCutPartB[i]*X[i] for i in range(self.numTerminal)) > sum(self.probs[s]*theta[s] for s in range(self.numScen)) + tol_optcut:
                    self.MP.addConstr(
                        singleCutPartA + gp.quicksum(singleCutPartB[i]*self._varX[i] for i in range(self.numTerminal))
                        <= sum(self.probs[s]*self._varTheta[s] for s in range(self.numScen)))                    
                    nOptCuts += 1
                    noCutAdded = False
            if ((method == 'a') or (method == 'p')) and noCutAdded:
                # No cut added. Check partition and compute UB
                cUB = sum(self.Objx[i]*X[i] for i in range(self.numTerminal))
                newSizePartition = sizePartition
                singleCutPartA = 0
                singleCutPartB = np.zeros(self.numTerminal)
                for p in range(sizePartition):
                    scen = np.extract(partitionId==p,range(self.numScen)).tolist()
                    for s in scen:
                        (stat,objSP,dMu, dNu) = self.SPsolve(self.scens[s])
                        dMuScen[s] = dMu
                        cUB += objSP*self.probs[s] 
                        singleCutPartA += -sum(self.scens[s,j] * dMu[j] for j in range(self.numCustomers))
                        for e in range(self.numTerminal):
                            singleCutPartB[e] += -dNu[e]
       
                    # Revise for repeated duals differences
                    (dualsUnique, inverse) = np.unique(dMuScen[scen,:],axis=0, return_inverse=True)
                    numSubsets = dualsUnique.shape[0]
                    if numSubsets > 1:
                        # we add new elements to the partition
                        partitionId[partitionId==p] = (inverse+newSizePartition)
                        # but rename the last one as the current one
                        partitionId[partitionId==(newSizePartition+numSubsets-1)] = p
                        newSizePartition += numSubsets -1
                        #print("Spliting %d into %d new subsets" % (p,numSubsets))
                print("Partition now has %d elements" % newSizePartition)    
                sizePartition = newSizePartition    
                self.dL = dMuScen
                self.part = partitionId

                #We add an extra optimality cut. I should be all scenarios feasible
                if (method == 'p'):
                    singleCutPartA = singleCutPartA/self.numScen
                    singleCutPartB = singleCutPartB/self.numScen

                    if singleCutPartA + sum(singleCutPartB[i]*X[i] for i in range(self.numTerminal)) > sum(self.probs[s]*theta[s] for s in range(self.numScen)) + tol_optcut:
                        self.MP.addConstr(
                            singleCutPartA + gp.quicksum(singleCutPartB[i]*self._varX[i] for i in range(self.numTerminal))
                            <= sum(self.probs[s]*self._varTheta[s] for s in range(self.numScen)))                    
                        nOptCuts += 1
                        noCutAdded = False                   


            #print("Iter %d: master = %f subp = %f gap = %f\n" % (it,cLB,cUB, -cUB/cLB+1))
            ub = min(ub, cUB)
            elap_time = time.time()
            #print("It=%d t=%f LB=%8.2f UB=%8.2f rgap=%8.2e nF=%d nO=%d"
            # % (it,elap_time-start_time,lb,ub,ub/(lb+1e-6)-1,nFeasCuts,nOptCuts))
            print("%d %8.2f %8.2f %8.2e %d %d %d %f"
             % (it,lb,ub,-ub/(lb-1e-6)+1,nFeasCuts,nOptCuts,sizePartition,elap_time-start_time))
            if (ub-lb < tol_stopRgap) or (-ub/(lb-1e-6)+1 < tol_stopRgap) :
                print("FinalReport: %d %f %f %f %d %d %d %f"
                % (it,lb,ub,-ub/(lb-1e-6)+1,nFeasCuts,nOptCuts,sizePartition,elap_time-start_time))
                break
            it += 1
    def MPsolveFull(self,sizePartition,partitionId):
        m = gp.Model("GAPM")
        #Defining variables
        X = m.addVars(range(self.numTerminal), lb=0, ub=self.Boundx, name="X")
        Y = m.addVars(range(sizePartition), range(self.numArcs), lb=0, name='Y')
        resources = m.addConstrs(gp.quicksum(self.Resources[k][i] * X[i] for i in range(self.numTerminal)) <= self.CapResources[k] for k in range(self.numResources))
        cap = {}
        dem = {}
        demP = np.zeros((sizePartition,self.numCustomers))
        probP = np.zeros(sizePartition)
        for p in range(sizePartition):
            demP[p] = np.sum(self.scens[partitionId==p], axis=0)/np.sum(partitionId==p)
            probP[p] = np.sum(partitionId==p)/self.numScen

        for i in range(self.numTerminal):
            arcs = [k for k in range(self.numArcs) if self.Arcs[k][0] == i]
            cap[i] = m.addConstrs(gp.quicksum(Y[p,a] for a in arcs) <= X[i] for p in range(sizePartition))
        for j in range(self.numCustomers):
            arcs = [k for k in range(self.numArcs) if self.Arcs[k][1] == j]
            dem[j] = m.addConstrs(gp.quicksum(Y[p,a] for a in arcs) <= demP[p][j] for p in range(sizePartition)) 
        m.setObjective(
            gp.quicksum(self.Objx[i]*X[i] for i in range(self.numTerminal))
            + gp.quicksum(self.Objy[self.Arcs[a]]*probP[p]*Y[p,a] for p in range(sizePartition) for a in range(self.numArcs) ),
             GRB.MINIMIZE)
        m.update()
        m.Params.OutputFlag = 0
        m.Params.Threads = 4
        m.optimize()
        if m.status == GRB.OPTIMAL:
            solX = np.array(m.getAttr('x',X).values())
            return(m.ObjVal, solX)
        else:
            raise Exception("Gurobi solStatus "+str(m.status))

    def GAPM(self, timeLimit = 86400, tol_optcut = 1e-5, tol_stopRgap = 1e-6, tol_stopAgap = 1e-6):
            ub = float('inf')
            lb = -float('inf')
            partitionId = np.zeros(self.numScen)
            sizePartition = 1 
            start_time = time.time()
            dMuScen = np.zeros((self.numScen, self.numCustomers))
            it = 1
            while(time.time() - start_time < timeLimit):
                # Solve master
                (cLB,X) = self.MPsolveFull(sizePartition,partitionId)
                #print("Iter %d: master = %f\n" % (it,cLB))
                lb = max(lb,cLB)
                # fix X on the subproblem
                self.SPsetX(X)
                #current UB including X costs
                cUB = sum(self.Objx[i]*X[i] for i in range(self.numTerminal))
                newSizePartition = sizePartition
                for p in range(sizePartition):
                    scen = np.extract(partitionId==p,range(self.numScen)).tolist()
                    for s in scen:
                        (stat,objSP,dMu, dNu) = self.SPsolve(self.scens[s])
                        dMuScen[s] = dMu
                        cUB += objSP*self.probs[s] 
                    # Revise for repeated duals differences
                    (dualsUnique, inverse) = np.unique(dMuScen[scen,:],axis=0, return_inverse=True)
                    numSubsets = dualsUnique.shape[0]
                    if numSubsets > 1:
                        # we add new elements to the partition
                        partitionId[partitionId==p] = (inverse+newSizePartition)
                        # but rename the last one as the current one
                        partitionId[partitionId==(newSizePartition+numSubsets-1)] = p
                        newSizePartition += numSubsets -1
                        #print("Spliting %d into %d new subsets" % (p,numSubsets))
                print("Partition now has %d elements" % newSizePartition)    
                sizePartition = newSizePartition    
                ub = min(ub, cUB)
                elap_time = time.time()
                #print("It=%d t=%f LB=%8.2f UB=%8.2f rgap=%8.2e nF=%d nO=%d"
                # % (it,elap_time-start_time,lb,ub,ub/(lb+1e-6)-1,nFeasCuts,nOptCuts))
                print("%d %8.2f %8.2f %8.2e %d %d %d %f"
                % (it,lb,ub,-ub/(lb-1e-6)+1,0,0,sizePartition,elap_time-start_time))
                if (ub-lb < tol_stopRgap) or (-ub/(lb-1e-6)+1 < tol_stopRgap) :
                    print("FinalReport: %d %f %f %f %d %d %d %f"
                    % (it,lb,ub,-ub/(lb-1e-6)+1,0,0,sizePartition,elap_time-start_time))
                    break
                it += 1


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
