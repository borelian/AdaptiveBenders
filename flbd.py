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

    # Evaluate a given solution, using original scenarios or out-of-sample
    # assign is an array with the facility assigned to each customers
    def computeScenarioCosts(self, assignment, outSample=False, nOSScen = 0):
        if outSample == False:
            samples = self.demandScen
        else:
            if nOSScen == 0:
                nOSScen = self.nscen
            rng = default_rng()
            samples = ( rng.random((self.numCustomers,nOSScen))
            < np.matmul( self.probs.reshape((self.numCustomers,1)) 
            ,np.ones((1,nOSScen))) )
        nscen = samples.shape[1]
        # Evaluate assignment cost
        scenCost = np.zeros(nscen)
        cumObj = np.zeros(nscen)
        for i in range(self.numFacilities):
            nCustAssigned = np.zeros(nscen, dtype=int)
            costSortedSubID = np.argsort(self.assignCost[i,assignment==i])
            subId = np.argwhere(assignment==i).ravel()
            for j in subId[costSortedSubID]:
                idx = ((nCustAssigned < self.maxCapac[i]) & samples[j,:])
                scenCost += idx * self.assignCost[i,j]
                nCustAssigned += idx
            scenCost += (np.sum(samples[subId,:],axis=0)-nCustAssigned)*self.unSatCost

        return scenCost
    
    def computeTotalCosts(self, outX, assignment, outSample=False, nOSScen = 0):
        scenCost = self.computeScenarioCosts(assignment, outSample, nOSScen)
        totalCost = np.sum(outX * self.openCost)
        totalCost += np.sum(scenCost) / len(scenCost)
        return totalCost

    # compute c^\xi_i breaking cost of the current assignment
    def getBreakCost(self, assignment):
        nStar = np.inf*np.ones((self.numFacilities,self.nscen))
        for i in range(self.numFacilities):
            if np.sum(assignment==i) >= self.maxCapac[i]:
                # sort assigned by cost
                costSortedSubID = np.argsort(self.assignCost[i,assignment==i])
                subId = np.argwhere(assignment==i).ravel()
                sortedAssigned = subId[costSortedSubID]
                #print(i, subId[costSortedSubID], assignCost[i,subId[costSortedSubID]],maxCapac[i] )
                # see demand of assigned in the order
                demandScenInOrder = self.demandScen[sortedAssigned]
                # check where maxCapac is reached
                numCustScen = np.sum(self.demandScen[sortedAssigned], axis=0)
                # note: if numCustScen[s] < maxCapac then all are true and return index 0
                whereReachMax = np.argmin(np.cumsum(self.demandScen[sortedAssigned], axis=0) < self.maxCapac[i], axis=0)
                idMaxReached = (numCustScen >= self.maxCapac[i])
                nStar[i,idMaxReached] = self.assignCost[i,sortedAssigned[whereReachMax[idMaxReached]]]
        return nStar

    def getGammaDual(self, breakcost):
        return np.maximum(self.unSatCost-breakcost,0)

    def getBetaDual(self, assignment, breakCost):
        cie = breakCost[assignment,:]
        cijj = self.assignCost[assignment,np.arange(self.numCustomers)]
        cijjScen = np.hstack([cijj.reshape(self.numCustomers,1)]*self.nscen)
        betaDual = cie - cijjScen
        betaDual[betaDual==np.inf] = self.unSatCost - cijjScen[betaDual==np.inf]
        betaDual[betaDual<0] = 0
        return betaDual

    def computeDualCost(self, gamma, beta):
        # Removed initial cost because it is included on MP objective separately
        dualObj = np.zeros(self.nscen) #self.unSatCost*np.sum(self.demandScen, axis=0)
        dualObj -= np.sum(beta*self.demandScen, axis=0)
        dualObj -= np.matmul(self.maxCapac,gamma)
        return  dualObj


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
            gp.quicksum(self.openCost[i]*X[i] for i in range(self.numFacilities)) + float(1/self.nscen) * gp.quicksum((self.assignCost[i,j] - self.unSatCost)*W[i,j,s] for i in range(self.numFacilities) for j in range(self.numCustomers) for s in range(self.nscen)) +  float(1/self.nscen)*np.sum(self.demandScen)*self.unSatCost, GRB.MINIMIZE)
        print("Updating and solving")
        m.update()
        m.Params.timeLimit = timeLimit
        m.Params.Threads = 4
        self.m = m
        self.m._varX = X
        self.m._varY = Y
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

    def formulateMP(self):
        self.MP =  gp.Model("MasterProblem")       
        #Defining variables
        X = self.MP.addVars(range(self.numFacilities), vtype=GRB.BINARY, name="X")
        Y = self.MP.addVars(range(self.numFacilities), range(self.numCustomers), vtype=GRB.BINARY,name='Y')

        # assign each customer
        self.MP.addConstrs(gp.quicksum(Y[i,j] for i in range(self.numFacilities)) == 1 for j in range(self.numCustomers))

        # only if open facility
        self.MP.addConstrs(Y[i,j] <= X[i] for i in range(self.numFacilities) for j in range(self.numCustomers))

        # min assignment
        self.MP.addConstrs(gp.quicksum(Y[i,j] for j in range(self.numCustomers)) >= self.minCapac[i]*X[i] for i in range(self.numFacilities) )

        theta = self.MP.addVars(range(self.nscen), lb=-1e6, name="theta")
        self.MP.setObjective( gp.quicksum(self.openCost[i]*X[i] for i in range(self.numFacilities)) + float(1/self.nscen)*np.sum(self.demandScen)*self.unSatCost + float(1/self.nscen)* gp.quicksum(theta[s] for s in range(self.nscen)), GRB.MINIMIZE)
        self.MP._varX = X
        self.MP._varY = Y
        self.MP._varTheta = theta
        self.MP._prob = self
        ## set parameters
        self.MP.Params.OutputFlag = 1
        self.MP.Params.Threads = 4

    # Solve master problem
    def MPsolve(self, useMulticuts = True):
        self.useMulticuts = useMulticuts
        self.MP.Params.LazyConstraints = 1
        #self.MP.Params.PreCrush = 0
        self.MP.optimize(singleBenders)
        if self.MP.status == GRB.OPTIMAL:
            solX = np.array(self.MP.getAttr('x',self.MP._varX).values())
#            solY = np.array(self.MP.getAttr('x',self.MP._varY).values())
            solT = np.array(self.MP.getAttr('x',self.MP._varTheta).values())
            solY = np.zeros(self.numCustomers, dtype=int)            
            for j in range(self.numCustomers):
                solY[j] = np.argmax([self.MP._varY[i,j].x for i in range(self.numFacilities)])
            return(self.MP.ObjVal, solX, solY, solT)
        else:
            raise Exception("Gurobi solStatus "+str(self.MP.status))

# %% single Benders
def singleBenders(model, where):
    tol_optcut = 1e-5
    prob = model._prob
    scenarios = prob.demandScen
    nscen = prob.nscen

    if where == GRB.Callback.MIPNODE:
        # At root node with an optimal relaxed solution
        if (model.cbGet(GRB.Callback.MIPNODE_NODCNT) < 1) and (model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
        #if (model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
            solX = np.array(list(model.cbGetNodeRel(model._varX).values())).ravel()
            solY = np.array(list(model.cbGetNodeRel(model._varY).values())).reshape(prob.numFacilities,prob.numCustomers)
            solT = np.array(list(model.cbGetNodeRel(model._varTheta).values())).ravel()

            # nz = np.argwhere(solX > 1e-6).ravel()
            # print([(i, solX[i]) for i in nz])
            # nz = np.argwhere(solY > 1e-6)
            # print([(i, j, solY[i,j]) for (i,j) in nz])
            
            alpha = np.zeros((prob.numFacilities,prob.numCustomers,nscen))
            gamma = np.zeros((prob.numFacilities,nscen))
            for i in range(prob.numFacilities):
                # scenarios where capacity is not reached
                scenNotReach = np.matmul(solY[i,:],scenarios) < (prob.maxCapac[i]*solX[i])
                costSortedID = np.argsort(prob.assignCost[i,:])
                # set duals for not reached
                alpha[i,:,scenNotReach] = (prob.unSatCost-prob.assignCost[i,:])
                # compute for reached capacity
                scen = np.argwhere(scenNotReach==False).ravel()
                if len(scen) > 0:
                    # demand * Y sorted by cost 
                    # demPerY = np.broadcast_to(solY[i,costSortedID].reshape(prob.numCustomers,1),(prob.numCustomers, len(scen))) * scenarios[costSortedID,:][:,scen]
                    demPerY = np.einsum('j,js->js',solY[i,costSortedID], scenarios[costSortedID,:][:,scen] )
                    # compute when reach maxCapac * X and its cost^\xi_i on each scenario
                    idNStar = np.argmin(np.cumsum(demPerY, axis=0) < (prob.maxCapac[i]* solX[i]), axis=0)
                    cie = prob.assignCost[i,costSortedID[idNStar]]
                    # Compute duals on these scenatios
                    gamma[i,scen] = prob.unSatCost-cie 
                    for idS in range(len(scen)):
                        alpha[i,:,scen[idS]] = np.maximum(cie[idS]-prob.assignCost[i,:],0)
            # Compute objective using duals
            # Removed initial cost because it is included on MP objective separately
            # obj = np.zeros(nscen) #prob.unSatCost*np.sum(scenarios, axis=0)
            # obj -= np.matmul(gamma.transpose(), (prob.maxCapac*solX))
            # #print(gamma[:,0], (prob.maxCapac*solX))
            # for s in range(nscen):
            #     obj[s] -= np.sum(np.matmul(alpha[:,:,s]*solY, scenarios[:,s]))
            obj = np.zeros(nscen)
            obj -= np.einsum('is,i,i->s',gamma, prob.maxCapac, solX)
            obj -= np.einsum('ijs,ij,js->s', alpha,solY,scenarios)

            ## Add cuts
            if prob.useMulticuts:
                scenAdd = np.argwhere(obj > (solT + tol_optcut))
                #print("Adding ",len(scenAdd), " user cuts at root node")
                for s in scenAdd.ravel():
                    # if s == 0:
                    #     print("Scenario ",s, "theta=", solT[s], 'dualObj=',obj[s])
                    nZero = np.argwhere(alpha[:,:,s] > 1e-6)
                    exp1 = gp.quicksum(alpha[i,j,s]*scenarios[j,s]*model._varY[i,j] for (i,j) in nZero) #range(prob.numFacilities) for j in range(prob.numCustomers))
                    # exp1E = gp.quicksum(alpha[i,j,s]*scenarios[j,s]*solY[i,j] for (i,j) in nZero)
                    exp2 = gp.quicksum(gamma[i,s]*prob.maxCapac[i]*model._varX[i] for i in range(prob.numFacilities))
                    # exp2E = gp.quicksum(gamma[i,s]*prob.maxCapac[i]*solX[i] for i in range(prob.numFacilities))
                    # if s == 7:
                    #     print(model._varTheta[s] >= -(exp1+exp2))
                    #     print(solT[s], -(exp1E+exp2E))
                    model.cbLazy(model._varTheta[s] >= -(exp1+exp2))
            else:
                if np.sum(obj) > (np.sum(solT) + tol_optcut):
                    coeffX = np.einsum('is,i->i', gamma, prob.maxCapac)
                    coeffY = np.einsum('ijs,js->ij', alpha, scenarios)
                    exp1 = gp.quicksum(coeffY[i,j]*model._varY[i,j] for i in range(prob.numFacilities) for j in range(prob.numCustomers))
                    exp2 = gp.quicksum(coeffX[i]*model._varX[i] for i in range(prob.numFacilities))
                    model.cbLazy(gp.quicksum(model._varTheta[s] for s in range(nscen)) >= - (exp1 + exp2))

    elif where == GRB.Callback.MIPSOL:

        #varX = model.cbGetSolution(model._varX)
        varY = model.cbGetSolution(model._varY)
        varT = model.cbGetSolution(model._varTheta)
        Y = np.zeros(prob.numCustomers, dtype=int)
        for j in range(prob.numCustomers):
            Y[j] = np.argmax([varY[i,j] for i in range(prob.numFacilities)])
        breakcost = prob.getBreakCost(Y)
        gamma = prob.getGammaDual(breakcost)
        beta = prob.getBetaDual(Y,breakcost)
        dCost = prob.computeDualCost(gamma, beta)
        theta = np.array(list(varT.values()))
        if prob.useMulticuts:
            scenAdd = np.argwhere(dCost > (theta + tol_optcut))
            #print("Adding ",len(scenAdd), " lazy cuts at B&B node")
            for s in scenAdd.ravel():
                #gammaMat = np.broadcast_to(gamma[:,s].reshape(prob.numFacilities,1), (prob.numFacilities,prob.numCustomers))
                #betaMat = np.broadcast_to(beta[:,s].reshape(1,prob.numCustomers), (prob.numFacilities,prob.numCustomers))
                #alphaDualScen = np.maximum(prob.unSatCost - prob.assignCost- betaMat - gammaMat, 0)
                alphaDualScen = np.maximum(prob.unSatCost - prob.assignCost- beta[:,s][np.newaxis,:] - gamma[:,s][:,np.newaxis], 0)
                #alphaDualScen = np.maximum(np.einsum('ij,j,i->ij', prob.unSatCost - prob.assignCost, -beta[:,s], -gamma[:,s]),0)
                alphaDualScen[Y,np.arange(prob.numCustomers)] = 0
                exp1 = gp.quicksum(alphaDualScen[i,j]*model._varY[i,j] for i in range(prob.numFacilities) for j in range(prob.numCustomers))
                #exp1E = gp.quicksum(alphaDualScen[i,j]*varY[i,j] for i in range(prob.numFacilities) for j in range(prob.numCustomers))
                exp2 = gp.quicksum((gamma[i,s]*prob.maxCapac[i])*model._varX[i] for i in range(prob.numFacilities))
                #exp2E = gp.quicksum((gammaMat[i,0]*prob.maxCapac[i])*varX[i] for i in range(prob.numFacilities))
                exp3 = np.sum(beta[:,s]*scenarios[:,s])
                #exp3E = np.sum(beta[:,s]*scenarios[:,s])
                # if s == 0:
                #     print(theta[s], varT[s], -(exp1E+exp2E+exp3E), dCost[s])
                #     #print(exp1+exp2+exp3)
                model.cbLazy(model._varTheta[s] >= -(exp1+exp2+exp3))
        else: 
            if np.sum(dCost) > (np.sum(theta) + tol_optcut):
                coeffX = np.einsum('is,i->i', gamma, prob.maxCapac)
                coeffCte = np.einsum('js,js->', beta, scenarios)
                alpha = np.maximum(prob.unSatCost -prob.assignCost[:,:,np.newaxis]-beta[np.newaxis,:,:]-gamma[:,np.newaxis,:],0)
                alpha[Y,np.arange(prob.numCustomers),:]=0
                coeffY = np.sum(alpha, axis=2)
                exp1 = gp.quicksum(coeffY[i,j]*model._varY[i,j] for i in range(prob.numFacilities) for j in range(prob.numCustomers))
                exp2 = gp.quicksum(coeffX[i]*model._varX[i] for i in range(prob.numFacilities))
                model.cbLazy(gp.quicksum(model._varTheta[s] for s in range(nscen)) >= - (exp1 + exp2 + coeffCte))




# %%
datfile = './FLPBD_instances/EJ_p23_4.dat'
prob = SFLBD(datfile)
prob.genScenarios(10)

# %%
obj, X, Y = prob.solveDE()


# %%
prob.formulateMP()
obj2,X2,Y2,Theta2 = prob.MPsolve(useMulticuts=False)

# %%
##%%timeit
breakcost = prob.getBreakCost(Y)
gamma = prob.getGammaDual(breakcost)
beta = prob.getBetaDual(Y,breakcost)
dualCost = prob.computeDualCost(gamma, beta)
primalCost = prob.computeScenarioCosts(Y)
#print(np.max(dualCost - primalCost), np.min(dualCost - primalCost))

# %%
solY = np.array(prob.m.getAttr('x',prob.m._varY.values())).reshape(prob.numFacilities,prob.numCustomers)

# %%
# solucion raiz de './FLPBD_instances/EJ_p23_4.dat' para 100 escenarios
solX = np.zeros(prob.numFacilities)
for (i,val) in [(10, 1.0), (12, 0.1274725274725281), (18, 0.8725274725274721)]:
    solX[i] = val
solY = np.zeros((prob.numFacilities,prob.numCustomers))
for (i,j,val) in [(10, 0, 0.1274725274725279), (10, 1, 1.0), (10, 2, 1.0), (10, 3, 0.1274725274725279), (10, 4, 1.0), (10, 5, 1.0), (10, 6, 1.0), (10, 8, 1.0), (10, 11, 0.8725274725274719), (10, 12, 0.1274725274725279), (10, 13, 1.0), (10, 15, 0.1274725274725279), (10, 18, 0.1274725274725279), (10, 19, 1.0), (10, 20, 0.1274725274725279), (10, 21, 1.0), (10, 23, 1.0), (10, 24, 1.0), (10, 25, 1.0), (10, 26, 0.8725274725274719), (10, 27, 1.0), (10, 28, 1.0), (10, 29, 1.0), (10, 31, 1.0), (10, 32, 1.0), (10, 33, 0.1274725274725279), (10, 34, 0.8725274725274719), (10, 35, 1.0), (10, 37, 1.0), (12, 7, 0.1274725274725279), (12, 9, 0.1274725274725281), (12, 10, 0.1274725274725279), (12, 11, 0.12747252747252813), (12, 14, 0.1274725274725281), (12, 16, 0.1274725274725281), (12, 17, 0.1274725274725281), (12, 22, 0.1274725274725281), (12, 26, 0.1274725274725281), (12, 30, 0.1274725274725279), (12, 34, 0.1274725274725281), (12, 36, 0.1274725274725279), (12, 38, 0.1274725274725279), (12, 39, 0.1274725274725281), (18, 0, 0.8725274725274721), (18, 3, 0.8725274725274721), (18, 7, 0.8725274725274721), (18, 9, 0.8725274725274719), (18, 10, 0.8725274725274721), (18, 12, 0.8725274725274721), (18, 14, 0.8725274725274721), (18, 15, 0.8725274725274721), (18, 16, 0.8725274725274721), (18, 17, 0.8725274725274721), (18, 18, 0.8725274725274721), (18, 20, 0.8725274725274721), (18, 22, 0.8725274725274721), (18, 30, 0.8725274725274721), (18, 33, 0.8725274725274721), (18, 36, 0.8725274725274721), (18, 38, 0.8725274725274721), (18, 39, 0.8725274725274719)]:
    solY[i,j] = val

# %%
%%timeit
alpha = np.zeros((prob.numFacilities,prob.numCustomers,prob.nscen))
gamma = np.zeros((prob.numFacilities,prob.nscen))
for i in range(prob.numFacilities):
    # scenarios where capacity is not reached
    scenNotReach = np.matmul(solY[i,:],prob.demandScen) < (prob.maxCapac[i] * solX[i])
    costSortedID = np.argsort(prob.assignCost[i,:])
    # set duals for not reached
    alpha[i,:,scenNotReach] = (prob.unSatCost-prob.assignCost[i,:])
    # compute for reached capacity
    scen = np.argwhere(scenNotReach==False).ravel()
    if len(scen) > 0:
        # demand * Y sorted by cost 
        demPerY = np.broadcast_to(solY[i,costSortedID].reshape(prob.numCustomers,1),(prob.numCustomers, len(scen))) * prob.demandScen[costSortedID,:][:,scen]
        # compute when reach maxCapac * X and its cost^\xi_i on each scenario
        idNStar = np.argmin(np.cumsum(demPerY, axis=0) < (prob.maxCapac[i]*solX[i]), axis=0) ## * X[i]
        cie = prob.assignCost[i,costSortedID[idNStar]]
        # Compute duals on these scenatios
        gamma[i,scen] = prob.unSatCost-cie 
        for idS in range(len(scen)):
            alpha[i,:,scen[idS]] = np.maximum(cie[idS]-prob.assignCost[i,:],0)

# Compute objective using duals
obj = np.zeros(prob.nscen)
obj -= np.matmul(gamma.transpose(), (prob.maxCapac*solX)) ## * X[i] pero se asume 1
for s in range(prob.nscen):
   obj[s] -= np.sum(np.matmul(alpha[:,:,s]*solY, prob.demandScen[:,s]))

# %%
%%timeit
alpha = np.zeros((prob.numFacilities,prob.numCustomers,prob.nscen))
gamma = np.zeros((prob.numFacilities,prob.nscen))
for i in range(prob.numFacilities):
    # scenarios where capacity is not reached
    scenNotReach = np.matmul(solY[i,:],prob.demandScen) < (prob.maxCapac[i] * solX[i])
    costSortedID = np.argsort(prob.assignCost[i,:])
    # set duals for not reached
    alpha[i,:,scenNotReach] = (prob.unSatCost-prob.assignCost[i,:])
    # compute for reached capacity
    scen = np.argwhere(scenNotReach==False).ravel()
    if len(scen) > 0:
        # demand * Y sorted by cost 
        #demPerY = np.broadcast_to(solY[i,costSortedID].reshape(prob.numCustomers,1),(prob.numCustomers, len(scen))) * prob.demandScen[costSortedID,:][:,scen]
        demPerY = np.einsum('j,js->js',solY[i,costSortedID], prob.demandScen[costSortedID,:][:,scen] )
        # compute when reach maxCapac * X and its cost^\xi_i on each scenario
        idNStar = np.argmin(np.cumsum(demPerY, axis=0) < (prob.maxCapac[i]*solX[i]), axis=0) ## * X[i]
        cie = prob.assignCost[i,costSortedID[idNStar]]
        # Compute duals on these scenatios
        gamma[i,scen] = prob.unSatCost-cie 
        for idS in range(len(scen)):
            alpha[i,:,scen[idS]] = np.maximum(cie[idS]-prob.assignCost[i,:],0)

# Compute objective using duals
obj = np.zeros(prob.nscen)
obj -= np.einsum('is,i,i->s',gamma, prob.maxCapac, solX)
obj -= np.einsum('ijs,ij,js->s', alpha,solY,prob.demandScen)
  
  
# %%
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
