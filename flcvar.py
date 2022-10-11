import csv
from distutils.sysconfig import customize_compiler
import numpy as np
import time
import re
import gurobipy as gp
from gurobipy import GRB
from numpy.random import default_rng


class FLCVAR:
    def __init__(self, dataFile):
        with open(dataFile) as fileData:
            reader = csv.reader(fileData, delimiter=' ', skipinitialspace=True)

            line = reader.__next__()
            self.numFacilities = int(line[0])
            self.numCustomers = int(line[1])
            print(self.numFacilities, self.numCustomers)
            self.openCost = np.zeros(self.numFacilities)
            self.assignCost = np.zeros((self.numFacilities,self.numCustomers))
            self.demand = np.zeros(self.numCustomers)
            self.capacity = np.zeros(self.numFacilities)

            for id in range(self.numFacilities):
                line = reader.__next__()
                self.openCost[id] = float(line[0])
                self.capacity[id] = float(line[1])

            # print("Opening cost:", self.openCost)
            # print("Capacity:", self.capacity)
            for id in range(self.numCustomers):
                line = reader.__next__()
                self.demand[id] = float(line[0])
                # print("Reading costs for client ", id, " with demand ", self.demand[id])
                currfacility = 0
                while currfacility < self.numFacilities:
                    line = reader.__next__()
                    for val in line:
                        if val != '':
                            self.assignCost[currfacility, id] = float(val)/self.demand[id]
                            currfacility += 1
                # print("Costs: ", self.assignCost[:,id])


    def solveDeterministic(self, timeLimit = 86400):
        start_time = time.time()
        m = gp.Model("Deterministic")
        # Defining variables
        X = m.addVars(range(self.numFacilities), vtype=GRB.BINARY, name="X")
        Y = m.addVars(range(self.numFacilities), range(self.numCustomers), lb=0, name='Y')

        # sufficient for demand
        m.addConstr(gp.quicksum(self.capacity[i] * X[i] for i in range(self.numFacilities) ) >= np.sum(self.demand))

        # assign all demand
        m.addConstrs(gp.quicksum(Y[i,j] for i in range(self.numFacilities)) >= self.demand[j] for j in range(self.numCustomers))

        # only if open facility
        m.addConstrs(gp.quicksum(Y[i,j] for j in range(self.numCustomers)) <= self.capacity[i]*X[i] for i in range(self.numFacilities))

        # objective
        m.setObjective(
            gp.quicksum(self.openCost[i]*X[i] for i in range(self.numFacilities))
            + gp.quicksum(self.assignCost[i,j] * Y[i,j] for i in range(self.numFacilities) for j in range(self.numCustomers)), GRB.MINIMIZE)
        print("Updating and solving")
        m.update()
        m.Params.timeLimit = timeLimit
        m.Params.Threads = 1
        self.m = m
        self.m._varX = X
        self.m._varY = Y
        m.optimize()
        if m.status == GRB.OPTIMAL:
            solX = np.array([int(X[i].x) for i in range(self.numFacilities)])
            solY = np.zeros(self.numCustomers, dtype=int)
            print(np.argwhere(solX))

            for j in range(self.numCustomers):
                solY[j] = np.argmax([Y[i,j].x for i in range(self.numFacilities)])
            print("FinalReport: %d %f %f %f %d %d %d %f"
                  % (0,m.ObjVal,m.ObjVal,0,0,0,0,time.time()-start_time))
            return m.ObjVal, solX, solY
        else:
            raise Exception("Gurobi solStatus "+str(m.status))


class SFLCVAR(FLCVAR):
    def __init__(self, dataFile, alpha=0.9):
        FLCVAR.__init__(self, dataFile)
        self.nscen = 0
        self.demandScen = None
        self.scenProb = 0
        self.alpha = alpha

    def genScenarios(self, numScenarios, seed=0):
        rng = default_rng(seed=seed)
        self.nscen = numScenarios
        self.demandScen = np.einsum('cs,c->cs', rng.random((self.numCustomers,self.nscen)), 2*self.demand)
        self.scenProb = np.ones(self.nscen)*float(1/self.nscen)
        self.currentScen = self.demandScen
        self.currentNScen = self.nscen

    def solveDE(self, timeLimit = 86400):
        start_time = time.time()
        m = gp.Model("Deterministic")
        #Defining variables
        X = m.addVars(range(self.numFacilities), vtype=GRB.BINARY, name="X")
        Y = m.addVars(range(self.numFacilities), range(self.numCustomers), range(self.nscen), lb=0 ,name='Y')
        ObjectiveScen = m.addVars(range(self.nscen), lb=0, name='Z')
        Tau = m.addVar(0, name='T')

        # sufficient for demand
        demandScen = np.sum(self.demandScen, axis=0)
        maxDemand = np.maximum(np.sum(self.demand),np.max(demandScen))

        m.addConstr(gp.quicksum(self.capacity[i] * X[i] for i in range(self.numFacilities)) >= maxDemand)

        # assign all demand
        m.addConstrs(gp.quicksum(Y[i,j,s] for i in range(self.numFacilities))
                     >= self.demandScen[j,s] for j in range(self.numCustomers) for s in range(self.nscen))

        # only if open facility
        m.addConstrs(gp.quicksum(Y[i,j,s] for j in range(self.numCustomers))
                     <= self.capacity[i]*X[i] for i in range(self.numFacilities) for s in range(self.nscen))

        # CVaR scen
        m.addConstrs(gp.quicksum(self.assignCost[i,j] * Y[i,j,s] for i in range(self.numFacilities) for j in range(self.numCustomers))
                     <= Tau + ObjectiveScen[s] for s in range(self.nscen))
        # objective
        m.setObjective(
            gp.quicksum(self.openCost[i]*X[i] for i in range(self.numFacilities)) + Tau
            + 1/(1-self.alpha) * gp.quicksum(self.scenProb[s]*ObjectiveScen[s] for s in range(self.nscen)), GRB.MINIMIZE)
        print("Updating and solving")
        m.update()
        m.Params.timeLimit = timeLimit
        m.Params.Threads = 1
        self.m = m
        self.m._varX = X
        self.m._varY = Y
        m.optimize()
        if m.status == GRB.OPTIMAL:
            solX = np.array([int(X[i].x) for i in range(self.numFacilities)])
            solY = None
            solTau = Tau.x
            solT = np.array([ObjectiveScen[s].x for s in range(self.nscen)])
            print(np.argwhere(solX))

            print("FinalReport: %d %f %f %f %d %d %d %f"
                  % (0,m.ObjVal,m.ObjVal,0,0,0,0,time.time()-start_time))
            return m.ObjVal, solX, solTau, solY, solT
        else:
            raise Exception("Gurobi solStatus "+str(m.status))

    def formulateMP(self):
        self.MP =  gp.Model("MasterProblem")
        #Defining variables
        X = self.MP.addVars(range(self.numFacilities), vtype=GRB.BINARY, name="X")
        Tau = self.MP.addVar(lb=0, name='T')
        theta = self.MP.addVars(range(self.nscen), lb=0, name="theta")

        # sufficient for demand
        demandScen = np.sum(self.demandScen, axis=0)
        maxDemand = np.maximum(np.sum(self.demand),np.max(demandScen))

        self.MP.addConstr(gp.quicksum(self.capacity[i] * X[i] for i in range(self.numFacilities)) >= maxDemand)

        self.MP.setObjective( gp.quicksum(self.openCost[i]*X[i] for i in range(self.numFacilities)) + Tau
            + 1/(1-self.alpha) * gp.quicksum(self.scenProb[s]*theta[s] for s in range(self.nscen)), GRB.MINIMIZE)
        self.MP._varX = X
        self.MP._varTau = Tau
        self.MP._varTheta = theta
        ## set parameters
        self.MP.Params.OutputFlag = 1
        self.MP.Params.Threads = 4
        # Scenarios current (for APM)
        self.currentScen = self.demandScen
        self.currentNScen = self.nscen
        self.currentProbs = np.ones(self.nscen)/self.nscen
        self.currentPartition = np.arange(self.nscen)
        self.numLazycutsAdded = 0
        self.nRefinement = 0
        #
        self.MP._prob = self

    def formulateSP(self):
        self.SP =  gp.Model("SubProblemDual")
        #Defining variables
        alpha = self.SP.addVar(lb=0, ub=1, name='alpha')
        beta = self.SP.addVars(range(self.numCustomers), lb=0, name="beta")
        gamma = self.SP.addVars(range(self.numFacilities), lb=0, name="gamma")
        self.SP.addConstrs(-alpha*self.assignCost[i,j] + beta[j] - gamma[i] <= 0
                           for i in range(self.numFacilities) for j in range(self.numCustomers))
        self.SP.setObjective(0, GRB.MAXIMIZE)
        ## Copy variable to acces them later
        self._varAlpha = alpha
        self._varBeta = beta
        self._varGamma = gamma
        ## set parameters
        self.SP.Params.InfUnbdInfo = 1
        self.SP.Params.OutputFlag = 0
        self.SP.Params.Threads = 4

    # Set objective for mu variables given an x
    def SPsetX(self, X, Tau):
        for i in range(self.numFacilities):
            self._varGamma[i].obj = -X[i]*self.capacity[i]
        self._varAlpha.obj = -Tau

    # Set objective of beta variables, solve the problem and returns solution
    def SPsolve(self, Demand):
        for j in range(self.numCustomers):
            self._varBeta[j].obj = Demand[j]
        self.SP.optimize()
        # Case optimum found (cannot be unbounded)
        if self.SP.status == GRB.OPTIMAL:
            solAlpha = self._varAlpha.x
            solBeta = np.array(self.SP.getAttr('x',self._varBeta).values())
            solGamma = np.array(self.SP.getAttr('x',self._varGamma).values())
            return(1, self.SP.ObjVal, solAlpha, solBeta, solGamma)
        else:
            raise Exception("Gurobi solStatus "+str(self.SP.status))


    # Solve master problem
    def MPsolve(self, useMulticuts = True, useBenders = False, resetPartition = False):
        self.useMulticuts = useMulticuts
        self.MP.Params.LazyConstraints = 1
        self.MP.Params.TimeLimit = 86400
        self.MP.Params.Threads = 4
        #self.MP.Params.PreCrush = 0

        if useBenders:
            self.currentScen = np.einsum('js,s->j',self.demandScen,self.scenProb).reshape(-1,1)
            self.currentNScen = 1
            self.currentProbs = np.ones(1)
            self.currentPartition = np.zeros(self.nscen)
            self.resetPartition = resetPartition

        # Solve routine
        start_time = time.time()
        #self.MP.optimize(singleBenders)
        self.MP.optimize()
        elap_time = time.time()

        print("FinalReport: %d %f %f %f %d %d %d %f"
              % (self.nRefinement+1,
                 self.MP.getAttr(GRB.Attr.ObjVal),
                 self.MP.getAttr(GRB.Attr.ObjBound),
                 self.MP.getAttr(GRB.Attr.MIPGap),
                 0,
                 self.numLazycutsAdded,
                 self.currentNScen,
                 elap_time-start_time))


        # recover solution
        if self.MP.status == GRB.OPTIMAL:
            solX = np.array(self.MP.getAttr('x',self.MP._varX).values())
            print(np.argwhere(solX))
            #     solY = np.array(self.MP.getAttr('x',self.MP._varY).values())
            solTau = self.MP._varTau.x
            solT = np.array(self.MP.getAttr('x',self.MP._varTheta).values())
            solY = None
            return(self.MP.ObjVal, solX, solTau, solY, solT)
        else:
            raise Exception("Gurobi solStatus "+str(self.MP.status))



inst = FLCVAR("instancesFlcvar/cap101.txt")
inst.solveDeterministic()

inst = SFLCVAR("instancesFlcvar/cap101.txt")
inst.genScenarios(16)
objDE, xDE, tauDE, yDE, tDE = inst.solveDE()

inst.formulateMP()
inst.formulateSP()
obj, x, tau, y, t = inst.MPsolve()
inst.SPsetX(x,tau)
pricingObjval = np.zeros(inst.nscen)
for s in range(inst.nscen):
    demand = inst.demandScen[:,s]
    status, objSP, a, b, g = inst.SPsolve(demand)
    pricingObjval[s] = objSP
    #print("Scen:", s, " Objval:", objSP, " Beta:", b)
print(objDE/1e6, obj/1e6, (obj+(1/(1-inst.alpha))*np.sum(pricingObjval)/inst.nscen)/1e6)
