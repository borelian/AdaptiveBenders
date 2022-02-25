# %%
import csv
import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB

# %%
class MCF:
    def __init__(self, dowFile):
        self.nunNodes = 0
        self.numArcs = 0
        self.numComm = 0
        self.numScen = 0
        self.arcOD = {}
        self.arcV = {}
        self.arcCap = {}
        self.arcCost = {}
        self.commOD = {}
        self.commDem = {}
        #dowFile = '/Users/emoreno/Code/BendersGAPM-MCF/instances/r04.1.dow'
        with open(dowFile) as fileData:
            reader = csv.reader(fileData, delimiter=' ', skipinitialspace=True)
            for line in reader:
                if reader.line_num == 2:
                    self.numNodes = int(line[0])
                    self.numArcs = int(line[1])
                    self.numComm = int(line[2])
                elif (reader.line_num >= 3) and (reader.line_num <= self.numArcs+2):
                    e = reader.line_num - 3
                    self.arcOD[e] = (int(line[0]),int(line[1]))
                    self.arcV[e] = float(line[2])
                    self.arcCap[e] = float(line[3])
                    self.arcCost[e] = float(line[4])
                elif (reader.line_num > self.numArcs+2):
                    k = reader.line_num - self.numArcs-3
                    self.commOD[k] = (int(line[0]),int(line[1]))
                    self.commDem[k] = float(line[2])

 # %%
class SMCF(MCF):
    def __init__(self, dowFile, scenFile):
        MCF.__init__(self,dowFile)
        self.numScen = 0
        self.probs = None
        self.scens = None
        with open(scenFile) as fileData:
            reader = csv.reader(fileData, delimiter=' ', skipinitialspace=True)
            for line in reader:
                if reader.line_num == 1:
                    self.numScen = int(line[0])
                    self.probs = np.zeros(self.numScen)
                    self.scens = np.zeros((self.numScen,self.numComm))
                else:
                    s = reader.line_num-2
                    self.probs[s] = float(line[0])
                    for k in range(self.numComm):
                        self.scens[s,k] = line[k+1]
        self.formulateMP()
        self.formulateSP()
    
    def solveDE(self, timeLimit = 86400):
        start_time = time.time()
        m = gp.Model("DetEquiv")
        #Defining variables
        X = m.addVars(range(self.numArcs), lb=0, ub=1, name="X")
        Y = m.addVars(range(self.numScen), range(self.numArcs), range(self.numComm), lb=0, name='Y')
        cap = m.addConstrs(gp.quicksum(Y[s,e,k] for k in range(self.numComm)) <= self.arcCap[e]*X[e] for s in range(self.numScen) for e in range(self.numArcs))
        flow = {}
        for k in range(self.numComm):
            for v in range(1,self.numNodes+1):
                if self.commOD[k][0] == v:            
                    flow[k,v] = m.addConstrs(
                        gp.quicksum(Y[s,e,k] for e in range(self.numArcs) if self.arcOD[e][0] == v)
                        - gp.quicksum(Y[s,e,k] for e in range(self.numArcs) if self.arcOD[e][1] == v)
                        == self.scens[s,k] for s in range(self.numScen))        
                elif self.commOD[k][1] == v:            
                    flow[k,v] = m.addConstrs(
                        gp.quicksum(Y[s,e,k] for e in range(self.numArcs) if self.arcOD[e][0] == v)
                        - gp.quicksum(Y[s,e,k] for e in range(self.numArcs) if self.arcOD[e][1] == v)
                        == -self.scens[s,k] for s in range(self.numScen))        
                else:
                    flow[k,v] = m.addConstrs(
                        gp.quicksum(Y[s,e,k] for e in range(self.numArcs) if self.arcOD[e][0] == v)
                        - gp.quicksum(Y[s,e,k] for e in range(self.numArcs) if self.arcOD[e][1] == v)
                        == 0 for s in range(self.numScen))     
        m.setObjective(
            gp.quicksum(self.arcCost[e]*X[e] for e in range(self.numArcs))
            + gp.quicksum(self.arcV[e]*self.probs[s]*Y[s,e,k] for s in range(self.numScen) for e in range(self.numArcs) for k in  range(self.numComm))
        )
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
        X = self.MP.addVars(range(self.numArcs), lb=0, ub=1, name="X")
        theta = self.MP.addVars(range(self.numScen), lb=0, name="theta")
        self.MP.setObjective(
            gp.quicksum(self.arcCost[e]*X[e] for e in range(self.numArcs))
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
        lambd = self.SP.addVars(range(1,self.numNodes+1), range(self.numComm), lb=-float('inf'), name="lambda")
        mu = self.SP.addVars(range(self.numArcs), lb=0, name="mu")
        self.SP.addConstrs(lambd[self.arcOD[e][0],k] - lambd[self.arcOD[e][1],k] - mu[e] <= self.arcV[e] for e in range(self.numArcs) for k in range(self.numComm))
        self.SP.setObjective(0, GRB.MAXIMIZE)
        ## Copy variable to acces them later
        self._varLambda = lambd
        self._varMu = mu
        ## set parameters
        self.SP.Params.InfUnbdInfo = 1
        self.SP.Params.OutputFlag = 0
        self.SP.Params.Threads = 4

    # Set objective for mu variables given an x
    def SPsetX(self, X):
        for e in range(self.numArcs):
            self._varMu[e].obj = -self.arcCap[e]*X[e]
    
    # Set objective of lambda variables, solve the problem and returns solution
    def SPsolve(self, Demand):
        for k in range(self.numComm):
            self._varLambda[self.commOD[k][0],k].obj = Demand[k]
            self._varLambda[self.commOD[k][1],k].obj = -Demand[k]
        self.SP.optimize()
        # Case optimum found
        if self.SP.status == GRB.OPTIMAL:
            solMu = np.array(self.SP.getAttr('x',self._varMu).values())
            solDiffLambda = np.array([self._varLambda[self.commOD[k][0],k].x - self._varLambda[self.commOD[k][1],k].x for k in range(self.numComm)])
            return(1, self.SP.ObjVal, solDiffLambda, solMu)
        # if unbounded get ray
        elif self.SP.status == GRB.UNBOUNDED:
            solMu = np.array(self.SP.getAttr('UnbdRay',self._varMu).values())
            solDiffLambda = np.array([self._varLambda[self.commOD[k][0],k].UnbdRay - self._varLambda[self.commOD[k][1],k].UnbdRay for k in range(self.numComm)])
            return(0, float('inf'), solDiffLambda, solMu)
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
        dLambdasDiff = np.zeros((self.numScen, self.numComm))
        it = 1
        while(time.time() - start_time < timeLimit):
            # Solve master
            (cLB,X,theta) = self.MPsolve()
            #print("Iter %d: master = %f\n" % (it,cLB))
            lb = max(lb,cLB)
            # fix X on the subproblem
            self.SPsetX(X)
            #current UB including X costs
            cUB = sum(self.arcCost[e]*X[e] for e in range(self.numArcs))
            #info for single cuts
            noInfCutAdded = True
            singleCutPartA = 0
            singleCutPartB = np.zeros(self.numArcs)
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
                (stat,objSP,dLambda, dMu) = self.SPsolve(demP)
                if stat == 0: # Unbounded
                    # Feasibility cut
                    self.MP.addConstr(
                        gp.quicksum(demP[k] * dLambda[k] for k in range(self.numComm))
                        - gp.quicksum(dMu[e]*self.arcCap[e]*self._varX[e] for e in range(self.numArcs))
                        <= 0
                    )
                    nFeasCuts += 1
                    noInfCutAdded = False
                    noCutAdded = False
                else: # Optimum
                    # dLambdasDiff[s] = dLambda
                    if (method == 'm') or (method == 'a'):
                        #Optimality cut
                        partA = sum(demP[k] * dLambda[k] for k in range(self.numComm))
                        partB = -sum(dMu[e]*self.arcCap[e]*X[e] for e in range(self.numArcs))
                        # Warning: assuming equiprobable for numerical stability
                        if partA+partB > (sum(theta[partitionId==p])/np.sum(partitionId==p)) + tol_optcut:
                            scen = np.extract(partitionId==p,range(self.numScen)).tolist()
                            self.MP.addConstr(
                                gp.quicksum(demP[k] * dLambda[k] for k in range(self.numComm))
                                - gp.quicksum(dMu[e]*self.arcCap[e]*self._varX[e] for e in range(self.numArcs))
                                <= gp.quicksum(self._varTheta[s] for s in scen)/np.sum(partitionId==p))
                            nOptCuts += 1
                            noCutAdded = False
                    elif ((method == 's') or (method == 'p')) and noInfCutAdded:
                        singleCutPartA += sum(demP[k] * dLambda[k] for k in range(self.numComm))*probP
                        for e in range(self.numArcs):
                            singleCutPartB[e] += -dMu[e]*self.arcCap[e]*probP
                if (method != 'a') and (method != 'p') :
                    cUB += np.sum(self.probs[partitionId==p])*objSP 
                else:
                    cUB = float('inf')
            if ((method == 's') or (method == 'p')) and noInfCutAdded:
                if singleCutPartA + sum(singleCutPartB[e]*X[e] for e in range(self.numArcs)) > sum(self.probs[s]*theta[s] for s in range(self.numScen)) + tol_optcut:
                    self.MP.addConstr(
                        singleCutPartA + gp.quicksum(singleCutPartB[e]*self._varX[e] for e in range(self.numArcs))
                        <= sum(self.probs[s]*self._varTheta[s] for s in range(self.numScen)))                    
                    nOptCuts += 1
                    noCutAdded = False
            if ((method == 'a') or (method == 'p')) and noCutAdded:
                # No cut added. Check partition and compute UB
                cUB = sum(self.arcCost[e]*X[e] for e in range(self.numArcs))
                newSizePartition = sizePartition
                singleCutPartA = 0
                singleCutPartB = np.zeros(self.numArcs)
                for p in range(sizePartition):
                    scen = np.extract(partitionId==p,range(self.numScen)).tolist()

                    for s in scen:
                        (stat,objSP,dLambda, dMu) = self.SPsolve(self.scens[s])
                        dLambdasDiff[s] = dLambda
                        cUB += objSP*self.probs[s] 
                        singleCutPartA += sum(self.scens[s,k] * dLambda[k] for k in range(self.numComm))
                        for e in range(self.numArcs):
                            singleCutPartB[e] += -dMu[e]*self.arcCap[e]

       
                    # Revise for repeated duals differences
                    (dualsUnique, inverse) = np.unique(dLambdasDiff[scen,:],axis=0, return_inverse=True)
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
                self.dL = dLambdasDiff
                self.part = partitionId
                
                if (method == 'p'):
                    singleCutPartA = singleCutPartA/self.numScen
                    singleCutPartB = singleCutPartB/self.numScen
                    #We add an extra optimality cut. I should be all scenarios feasible
                    if singleCutPartA + sum(singleCutPartB[e]*X[e] for e in range(self.numArcs)) > sum(self.probs[s]*theta[s] for s in range(self.numScen)) + tol_optcut:
                        self.MP.addConstr(
                            singleCutPartA + gp.quicksum(singleCutPartB[e]*self._varX[e] for e in range(self.numArcs))
                            <= sum(self.probs[s]*self._varTheta[s] for s in range(self.numScen)))                    
                        nOptCuts += 1
                        noCutAdded = False                   


            #print("Iter %d: master = %f subp = %f gap = %f\n" % (it,cLB,cUB, cUB/cLB-1))
            ub = min(ub, cUB)
            elap_time = time.time()
            #print("It=%d t=%f LB=%8.2f UB=%8.2f rgap=%8.2e nF=%d nO=%d"
            # % (it,elap_time-start_time,lb,ub,ub/(lb+1e-6)-1,nFeasCuts,nOptCuts))
            print("%d %8.2f %8.2f %8.2e %d %d %d %f"
             % (it,lb,ub,ub/(lb+1e-6)-1,nFeasCuts,nOptCuts,sizePartition,elap_time-start_time))
            if (ub-lb < tol_stopRgap) or (ub/(lb+1e-6)-1 < tol_stopRgap) :
                print("FinalReport: %d %f %f %f %d %d %d %f"
                % (it,lb,ub,ub/(lb+1e-6)-1,nFeasCuts,nOptCuts,sizePartition,elap_time-start_time))
                break
            it += 1


    def MPsolveFull(self,sizePartition,partitionId):
        m = gp.Model("GAPM")
        #Defining variables
        X = m.addVars(range(self.numArcs), lb=0, ub=1, name="X")
        Y = m.addVars(range(sizePartition), range(self.numArcs), range(self.numComm), lb=0, name='Y')
        cap = m.addConstrs(gp.quicksum(Y[s,e,k] for k in range(self.numComm)) <= self.arcCap[e]*X[e] for s in range(sizePartition) for e in range(self.numArcs))
        flow = {}
        demP = np.zeros((sizePartition,self.numComm))
        probP = np.zeros(sizePartition)
        for p in range(sizePartition):
            demP[p] = np.sum(self.scens[partitionId==p], axis=0)/np.sum(partitionId==p)
            probP[p] = np.sum(partitionId==p)/self.numScen

        for k in range(self.numComm):
            for v in range(1,self.numNodes+1):

                if self.commOD[k][0] == v:            
                    flow[k,v] = m.addConstrs(
                        gp.quicksum(Y[s,e,k] for e in range(self.numArcs) if self.arcOD[e][0] == v)
                        - gp.quicksum(Y[s,e,k] for e in range(self.numArcs) if self.arcOD[e][1] == v)
                        == demP[s,k] for s in range(sizePartition))        
                elif self.commOD[k][1] == v:            
                    flow[k,v] = m.addConstrs(
                        gp.quicksum(Y[s,e,k] for e in range(self.numArcs) if self.arcOD[e][0] == v)
                        - gp.quicksum(Y[s,e,k] for e in range(self.numArcs) if self.arcOD[e][1] == v)
                        == -demP[s,k] for s in range(sizePartition))        
                else:
                    flow[k,v] = m.addConstrs(
                        gp.quicksum(Y[s,e,k] for e in range(self.numArcs) if self.arcOD[e][0] == v)
                        - gp.quicksum(Y[s,e,k] for e in range(self.numArcs) if self.arcOD[e][1] == v)
                        == 0 for s in range(sizePartition))     
        m.setObjective(
            gp.quicksum(self.arcCost[e]*X[e] for e in range(self.numArcs))
            + gp.quicksum(self.arcV[e]*probP[s]*Y[s,e,k] for s in range(sizePartition) for e in range(self.numArcs) for k in  range(self.numComm))
        )
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
            dLambdasDiff = np.zeros((self.numScen, self.numComm))
            it = 1
            while(time.time() - start_time < timeLimit):
                # Solve master
                (cLB,X) = self.MPsolveFull(sizePartition,partitionId)
                #print("Iter %d: master = %f\n" % (it,cLB))
                lb = max(lb,cLB)
                # fix X on the subproblem
                self.SPsetX(X)
                #current UB including X costs
                cUB = sum(self.arcCost[e]*X[e] for e in range(self.numArcs))
                newSizePartition = sizePartition
                for p in range(sizePartition):
                    scen = np.extract(partitionId==p,range(self.numScen)).tolist()
                    for s in scen:
                        (stat,objSP,dLambda, dMu) = self.SPsolve(self.scens[s])
                        dLambdasDiff[s] = dLambda
                        cUB += objSP*self.probs[s] 
                    # Revise for repeated duals differences
                    (dualsUnique, inverse) = np.unique(dLambdasDiff[scen,:],axis=0, return_inverse=True)
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
                % (it,lb,ub,ub/(lb+1e-6)-1,0,0,sizePartition,elap_time-start_time))
                if (ub-lb < tol_stopRgap) or (ub/(lb+1e-6)-1 < tol_stopRgap) :
                    print("FinalReport: %d %f %f %f %d %d %d %f"
                    % (it,lb,ub,ub/(lb+1e-6)-1,0,0,sizePartition,elap_time-start_time))
                    break
                it += 1





# %%
# prob2 = SMCF('/Users/emoreno/Code/BendersGAPM-MCF/instances/r04.1.dow','/Users/emoreno/Code/BendersGAPM-MCF/instances/r04-0-100')
# prob2.GAPM()
# prob2.Benders('p', 500)
# %%
# dem = list(prob2.commDem.values())
# prob2.SPsolve(dem)
# # %%
# prob2.SPsetX(np.ones(prob2.numArcs))
# # %%
# prob2.SPsolve(list(prob2.commDem.values()))
# # %%
# prob2.MPsolve()

# # %%

# # %%

# %%
