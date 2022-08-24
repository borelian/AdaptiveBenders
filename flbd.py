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
    self.scenProb = 0

  def genScenarios(self, numScenarios, seed=0):
    rng = default_rng(seed=seed)
    self.nscen = numScenarios
    self.demandScen = ( rng.random((self.numCustomers,self.nscen))
        < np.matmul( self.probs.reshape((self.numCustomers,1))
        ,np.ones((1,self.nscen))) )
    self.scenProb = np.ones(self.nscen)*float(1/self.nscen)
    self.currentScen = self.demandScen
    self.currentNScen = self.nscen

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
      #scenCost += (np.sum(samples[subId,:],axis=0)-nCustAssigned)*self.unSatCost

    return scenCost

  def computeTotalCosts(self, outX, assignment, outSample=False, nOSScen = 0):
    scenCost = self.computeScenarioCosts(assignment, outSample, nOSScen)
    totalCost = np.sum(outX * self.openCost)
    totalCost += np.sum(scenCost*self.scenProb)
    return totalCost

  # compute c^\xi_i breaking cost of the current assignment
  def getBreakCost(self, assignment, scenarios):
    nscen = scenarios.shape[1]
    nStar = np.inf*np.ones((self.numFacilities,nscen))
    for i in range(self.numFacilities):
      if np.sum(assignment==i) >= self.maxCapac[i]:
        # sort assigned by cost
        costSortedSubID = np.argsort(self.assignCost[i,assignment==i])
        subId = np.argwhere(assignment==i).ravel()
        sortedAssigned = subId[costSortedSubID]
        #print(i, subId[costSortedSubID], assignCost[i,subId[costSortedSubID]],maxCapac[i] )
        # see demand of assigned in the order
        #demandScenInOrder = self.demandScen[sortedAssigned]
        # check where maxCapac is reached
        numCustScen = np.sum(scenarios[sortedAssigned], axis=0)
        # note: if numCustScen[s] < maxCapac then all are true and return index 0
        whereReachMax = np.argmin(np.cumsum(scenarios[sortedAssigned], axis=0) < self.maxCapac[i], axis=0)
        idMaxReached = (numCustScen >= self.maxCapac[i])
        nStar[i,idMaxReached] = self.assignCost[i,sortedAssigned[whereReachMax[idMaxReached]]]
    return nStar

  def getGammaDual(self, breakcost):
    return np.maximum(self.unSatCost-breakcost,0)

  def getBetaDual(self, assignment, breakCost):
    nscen = breakCost.shape[1]
    cie = breakCost[assignment,:]
    cijj = self.assignCost[assignment,np.arange(self.numCustomers)]
    cijjScen = np.hstack([cijj.reshape(self.numCustomers,1)]*nscen)
    betaDual = cie - cijjScen
    betaDual[betaDual==np.inf] = self.unSatCost - cijjScen[betaDual==np.inf]
    betaDual[betaDual<0] = 0
    return betaDual

  def computeDualCost(self, gamma, beta):
    # Removed initial cost because it is included on MP objective separately
    dualObj = np.zeros(self.currentNScen) #self.unSatCost*np.sum(self.demandScen, axis=0)
    dualObj -= np.sum(beta*self.currentScen, axis=0)
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

    # eliminated constant term
    m.setObjective(
      gp.quicksum(self.openCost[i]*X[i] for i in range(self.numFacilities)) + gp.quicksum(self.scenProb[s]*(self.assignCost[i,j] - self.unSatCost)*W[i,j,s] for i in range(self.numFacilities) for j in range(self.numCustomers) for s in range(self.nscen)) +  np.sum(self.demandScen*self.scenProb)*self.unSatCost*0, GRB.MINIMIZE)
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

    # eliminated constant term
    self.MP.setObjective( gp.quicksum(self.openCost[i]*X[i] for i in range(self.numFacilities)) + np.sum(self.scenProb*self.demandScen)*self.unSatCost*0 + gp.quicksum(self.scenProb[s]*theta[s] for s in range(self.nscen)), GRB.MINIMIZE)
    self.MP._varX = X
    self.MP._varY = Y
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


  # Solve master problem
  def MPsolve(self, useMulticuts = True, useBenders = False):
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
      

    # Solve routine
    start_time = time.time()
    self.MP.optimize(singleBenders)
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
#     solY = np.array(self.MP.getAttr('x',self.MP._varY).values())
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
  scenarios = prob.currentScen
  nscen = prob.currentNScen
  #partition = prob.currentPartition
  if where == GRB.Callback.MIPNODE:
    # At root node with an optimal relaxed solution
    # WARNING! Assuming partition size = 1 or nscen
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
        #scen = np.argwhere(scenNotReach==False).ravel()
        scen = np.flatnonzero(scenNotReach==False)
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
        pTheta = np.zeros(prob.currentNScen)
        for p in range(prob.currentNScen):
          idx = (prob.currentPartition==p)
          pTheta[p] = np.sum(solT[idx])/np.sum(idx)
        #print(pTheta, obj)
        scenAdd = np.flatnonzero(obj > (pTheta + tol_optcut))
        #print("Adding ",len(scenAdd), " user cuts at root node")
        for s in scenAdd: #.ravel():
          nZero = np.argwhere(alpha[:,:,s] > 1e-6)
          exp1 = gp.quicksum(alpha[i,j,s]*scenarios[j,s]*model._varY[i,j] for (i,j) in nZero) #range(prob.numFacilities) for j in range(prob.numCustomers))
          # exp1E = gp.quicksum(alpha[i,j,s]*scenarios[j,s]*solY[i,j] for (i,j) in nZero)
          exp2 = gp.quicksum(gamma[i,s]*prob.maxCapac[i]*model._varX[i] for i in range(prob.numFacilities))
          # exp2E = gp.quicksum(gamma[i,s]*prob.maxCapac[i]*solX[i] for i in range(prob.numFacilities))
          idx = np.flatnonzero(prob.currentPartition==s)
          model.cbLazy(gp.quicksum(model._varTheta[id] for id in idx)*float(1/len(idx)) >= -(exp1+exp2))
          prob.numLazycutsAdded += 1
          #model.cbLazy(model._varTheta[s] >= -(exp1+exp2))

      else:
        #print("Root: ", np.sum(obj*prob.currentProbs), obj, prob.currentProbs, np.sum(solT)*float(1/prob.nscen) )
        if np.sum(obj*prob.currentProbs) > (np.sum(solT)*float(1/prob.nscen) + tol_optcut):
          #coeffX = np.einsum('is,i->i', gamma, prob.maxCapac)
          coeffX = np.einsum('is,i,s->i', gamma, prob.maxCapac,prob.currentProbs)
          #coeffY = np.einsum('ijs,js->ij', alpha, scenarios)
          coeffY = np.einsum('ijs,js,s->ij', alpha, scenarios,prob.currentProbs)
          exp1 = gp.quicksum(coeffY[i,j]*model._varY[i,j] for i in range(prob.numFacilities) for j in range(prob.numCustomers))
          #exp1E = gp.quicksum(coeffY[i,j]*solY[i,j] for i in range(prob.numFacilities) for j in range(prob.numCustomers))
          exp2 = gp.quicksum(coeffX[i]*model._varX[i] for i in range(prob.numFacilities))
          #exp2E = gp.quicksum(coeffX[i]*solX[i] for i in range(prob.numFacilities))
          #model.cbLazy(gp.quicksum(model._varTheta[s] for s in range(nscen)) >= - (exp1 + exp2))
          # OJO: cambie nscen por prob.nscen en la siguiente
          #print("Theta: ",np.sum(solT)*float(1/prob.nscen)," >= ",-(exp1E+exp2E))
          model.cbLazy(gp.quicksum(model._varTheta[s] for s in range(prob.nscen))*float(1/prob.nscen) >= - (exp1 + exp2))
          prob.numLazycutsAdded += 1


  # When a new integer solution has been found
  elif where == GRB.Callback.MIPSOL:
    potentialOptimalFound = False
    varX = model.cbGetSolution(model._varX)
    varY = model.cbGetSolution(model._varY)
    varT = model.cbGetSolution(model._varTheta)
    Y = np.zeros(prob.numCustomers, dtype=int)
    for j in range(prob.numCustomers):
      Y[j] = np.argmax([varY[i,j] for i in range(prob.numFacilities)])
    # Potential loop if refining partition. In single/multi, is a single run
    partitionRefined = True
    while (partitionRefined):
      partitionRefined = False
      breakcost = prob.getBreakCost(Y, scenarios)
      gamma = prob.getGammaDual(breakcost)
      beta = prob.getBetaDual(Y,breakcost)
      dCost = prob.computeDualCost(gamma, beta)
      theta = np.array(list(varT.values()))
      # Compute theta vars for each partition
      pTheta = np.zeros(prob.currentNScen)
      for p in range(prob.currentNScen):
        idx = (prob.currentPartition==p)
        pTheta[p] = np.sum(theta[idx])/np.sum(idx)
      if prob.useMulticuts:
        # Check cut violation
        #scenAdd = np.argwhere(dCost > (pTheta + tol_optcut))
        scenAdd = np.flatnonzero(dCost > (pTheta + tol_optcut))
        #print("Adding ",len(scenAdd), " lazy cuts at B&B node")
        for s in scenAdd: #.ravel():
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
          #idx = np.argwhere(prob.currentPartition==s).reshape(1,-1)[0]
          idx = np.flatnonzero(prob.currentPartition==s)
          model.cbLazy(gp.quicksum(model._varTheta[id] for id in idx)*float(1/len(idx)) >= -(exp1+exp2+exp3))
          prob.numLazycutsAdded += 1
        if len(scenAdd) == 0:
          potentialOptimalFound = True
      else:
        # Potential numerical error? At least in datfile = './FLPBD_instances/EJ_p23_4.dat' with 1000.
        # Dual objective (dCost) omit Y variables because Y=1 => alpha=0 but Y=~0. In multi, this term doesn't affect (because we evaluate each scenario independent) but in single this error sum making feasible an infeasible solution.
        #print("Mipsol: ",np.sum(dCost*prob.currentProbs), dCost, prob.currentProbs, (np.sum(theta)*float(1/prob.nscen)))
        #print(np.sum(dCost), np.sum(theta), np.sum(pTheta))
        if np.sum(dCost*prob.currentProbs) > (np.sum(theta)*float(1/prob.nscen) + tol_optcut):
          #coeffX = np.einsum('is,i->i', gamma, prob.maxCapac)
          coeffX = np.einsum('is,i,s->i', gamma, prob.maxCapac, prob.currentProbs)
          #coeffCte = np.einsum('js,js->', beta, scenarios)
          coeffCte = np.einsum('js,js,s->', beta, scenarios, prob.currentProbs)
          alpha = np.maximum(prob.unSatCost -prob.assignCost[:,:,np.newaxis]-beta[np.newaxis,:,:]-gamma[:,np.newaxis,:],0)
          alpha[Y,np.arange(prob.numCustomers),:]=0
          #coeffY = np.sum(alpha, axis=2)
          coeffY = np.einsum('ijs,s->ij', alpha, prob.currentProbs)
          exp1 = gp.quicksum(coeffY[i,j]*model._varY[i,j] for i in range(prob.numFacilities) for j in range(prob.numCustomers))
          #exp1E = gp.quicksum(coeffY[i,j]*varY[i,j] for i in range(prob.numFacilities) for j in range(prob.numCustomers))
          exp2 = gp.quicksum(coeffX[i]*model._varX[i] for i in range(prob.numFacilities))
          #exp2E = gp.quicksum(coeffX[i]*varX[i] for i in range(prob.numFacilities))
          #print("Theta:",np.sum(theta)*float(1/prob.nscen)," >= ",-(exp1E+exp2E+coeffCte), -(exp2E+coeffCte))
          # for idi in range(prob.numFacilities):
          #   for idj in range(prob.numCustomers):
          #     if varY[idi,idj] > 0:
          #      print(idi,idj,varY[idi,idj], coeffY[idi,idj],varY[idi,idj]*coeffY[idi,idj]) 
          #print("sumDualY ", exp1E)
          #print("Theta >=", - (exp1 + exp2 + coeffCte))
          # thetaVars = gp.LinExpr()
          # for p in range(prob.currentNScen):
          #   #idx = np.argwhere(prob.currentPartition==p).reshape(1,-1)[0]
          #   idx = np.flatnonzero(prob.currentPartition==p)
          #   thetaVars += gp.quicksum(model._varTheta[s] for s in idx)*float(1/len(idx))
          #model.cbLazy(thetaVars >= - (exp1 + exp2 + coeffCte))
          model.cbLazy(gp.quicksum(model._varTheta[s] for s in range(prob.nscen)) >= - (exp1 + exp2 + coeffCte)*prob.nscen)
          prob.numLazycutsAdded += 1
        else:
          potentialOptimalFound = True
      # if no cut added. Refine partition and recheck
      if potentialOptimalFound:
        #print("Checking refinement of previous partition with size", prob.currentNScen)
        breakcost = prob.getBreakCost(Y, prob.demandScen)
        beta = prob.getBetaDual(Y,breakcost)
        partitionId = prob.currentPartition.copy()
        newSizePartition = prob.currentNScen
        for p in range(prob.currentNScen):
          (dualsUnique, inverse, count) = np.unique(beta[:,partitionId==p], axis=1, return_inverse=True, return_counts=True)
          numSubsets = dualsUnique.shape[1]
          if numSubsets > 1:
            #print(p,prob.currentNScen,partitionId,inverse,newSizePartition)
            # we add new elements to the partition
            partitionId[partitionId==p] = (inverse+newSizePartition)
            # but rename the last one as the current one
            partitionId[partitionId==(newSizePartition+numSubsets-1)] = p
            # update probs of each partition
            prob.currentProbs = np.append(prob.currentProbs, count[:-1]/prob.nscen)
            prob.currentProbs[p] = count[-1]/prob.nscen
            newSizePartition += numSubsets -1
            #print("Spliting %d into %d new subsets" % (p,numSubsets))
        # Check
        if np.abs(1-np.sum(prob.currentProbs)) > 1e-6:
          print("WARNING: probs no suman",np.sum(prob.currentProbs)) 
        if newSizePartition != prob.currentNScen:
          print("Partition now has %d elements" % newSizePartition)
          # update info in the problem
          prob.nRefinement += 1
          prob.currentPartition = partitionId
          prob.currentNScen = newSizePartition
          prob.currentScen = np.zeros((prob.numCustomers,newSizePartition))
          for p in range(newSizePartition):
            prob.currentScen[:,p] = np.sum(prob.demandScen[:,partitionId==p], axis=1)/np.sum(partitionId==p)
          scenarios = prob.currentScen
          partitionRefined = True


# # %%
# datfile = './FLPBD_instances/EJ_p23_4.dat'
# prob = SFLBD(datfile)
# prob.genScenarios(1000)

# # %%
# obj, X, Y = prob.solveDE()


# # %%
# prob.formulateMP()
# obj2,X2,Y2,Theta2 = prob.MPsolve(useMulticuts=True, useBenders=True)

# # %%
# ##%%timeit
# breakcost = prob.getBreakCost(Y2, prob.demandScen)
# gamma = prob.getGammaDual(breakcost)
# beta = prob.getBetaDual(Y2,breakcost)
# dualCost = prob.computeDualCost(gamma, beta)
# primalCost = prob.computeScenarioCosts(Y)
# #print(np.max(dualCost - primalCost), np.min(dualCost - primalCost))

# # %%
# solY = np.array(prob.m.getAttr('x',prob.m._varY.values())).reshape(prob.numFacilities,prob.numCustomers)

# # %%
# # solucion raiz de './FLPBD_instances/EJ_p23_4.dat' para 100 escenarios
# solX = np.zeros(prob.numFacilities)
# for (i,val) in [(10, 1.0), (12, 0.1274725274725281), (18, 0.8725274725274721)]:
#     solX[i] = val
# solY = np.zeros((prob.numFacilities,prob.numCustomers))
# for (i,j,val) in [(10, 0, 0.1274725274725279), (10, 1, 1.0), (10, 2, 1.0), (10, 3, 0.1274725274725279), (10, 4, 1.0), (10, 5, 1.0), (10, 6, 1.0), (10, 8, 1.0), (10, 11, 0.8725274725274719), (10, 12, 0.1274725274725279), (10, 13, 1.0), (10, 15, 0.1274725274725279), (10, 18, 0.1274725274725279), (10, 19, 1.0), (10, 20, 0.1274725274725279), (10, 21, 1.0), (10, 23, 1.0), (10, 24, 1.0), (10, 25, 1.0), (10, 26, 0.8725274725274719), (10, 27, 1.0), (10, 28, 1.0), (10, 29, 1.0), (10, 31, 1.0), (10, 32, 1.0), (10, 33, 0.1274725274725279), (10, 34, 0.8725274725274719), (10, 35, 1.0), (10, 37, 1.0), (12, 7, 0.1274725274725279), (12, 9, 0.1274725274725281), (12, 10, 0.1274725274725279), (12, 11, 0.12747252747252813), (12, 14, 0.1274725274725281), (12, 16, 0.1274725274725281), (12, 17, 0.1274725274725281), (12, 22, 0.1274725274725281), (12, 26, 0.1274725274725281), (12, 30, 0.1274725274725279), (12, 34, 0.1274725274725281), (12, 36, 0.1274725274725279), (12, 38, 0.1274725274725279), (12, 39, 0.1274725274725281), (18, 0, 0.8725274725274721), (18, 3, 0.8725274725274721), (18, 7, 0.8725274725274721), (18, 9, 0.8725274725274719), (18, 10, 0.8725274725274721), (18, 12, 0.8725274725274721), (18, 14, 0.8725274725274721), (18, 15, 0.8725274725274721), (18, 16, 0.8725274725274721), (18, 17, 0.8725274725274721), (18, 18, 0.8725274725274721), (18, 20, 0.8725274725274721), (18, 22, 0.8725274725274721), (18, 30, 0.8725274725274721), (18, 33, 0.8725274725274721), (18, 36, 0.8725274725274721), (18, 38, 0.8725274725274721), (18, 39, 0.8725274725274719)]:
#     solY[i,j] = val

# # %%
# coeffXS = np.einsum('is,i->is', gamma, prob.maxCapac)[7,:]
# coeffCteS = np.einsum('js,js->s', beta, prob.currentScen)
# for s in range(prob.currentNScen):
#   print("Agg:",dualCost[s], "Disagg:", np.average(dualCostF[prob.currentPartition==s]), "Comp:",-coeffXS[s]-coeffCteS[s])
# print("Agg:",np.sum(prob.currentProbs*dualCost), "Disagg:", np.average(dualCostF),  "theta:", np.average(Theta2), "obj:", obj2-np.sum(X2*prob.openCost), "comp:", np.sum((-coeffXS-coeffCteS)*prob.currentProbs))

# # Compute objective using duals
# obj = np.zeros(prob.nscen)
# obj -= np.matmul(gamma.transpose(), (prob.maxCapac*solX)) ## * X[i] pero se asume 1
# for s in range(prob.nscen):
#    obj[s] -= np.sum(np.matmul(alpha[:,:,s]*solY, prob.demandScen[:,s]))

# # %%
# %%timeit
# alpha = np.zeros((prob.numFacilities,prob.numCustomers,prob.nscen))
# gamma = np.zeros((prob.numFacilities,prob.nscen))
# for i in range(prob.numFacilities):
#   # scenarios where capacity is not reached
#   scenNotReach = np.matmul(solY[i,:],prob.demandScen) < (prob.maxCapac[i] * solX[i])
#   costSortedID = np.argsort(prob.assignCost[i,:])
#   # set duals for not reached
#   alpha[i,:,scenNotReach] = (prob.unSatCost-prob.assignCost[i,:])
#   # compute for reached capacity
#   scen = np.argwhere(scenNotReach==False).ravel()
#   if len(scen) > 0:
#     # demand * Y sorted by cost
#     #demPerY = np.broadcast_to(solY[i,costSortedID].reshape(prob.numCustomers,1),(prob.numCustomers, len(scen))) * prob.demandScen[costSortedID,:][:,scen]
#     demPerY = np.einsum('j,js->js',solY[i,costSortedID], prob.demandScen[costSortedID,:][:,scen] )
#     # compute when reach maxCapac * X and its cost^\xi_i on each scenario
#     idNStar = np.argmin(np.cumsum(demPerY, axis=0) < (prob.maxCapac[i]*solX[i]), axis=0) ## * X[i]
#     cie = prob.assignCost[i,costSortedID[idNStar]]
#     # Compute duals on these scenatios
#     gamma[i,scen] = prob.unSatCost-cie
#     for idS in range(len(scen)):
#       alpha[i,:,scen[idS]] = np.maximum(cie[idS]-prob.assignCost[i,:],0)

# # Compute objective using duals
# obj = np.zeros(prob.nscen)
# obj -= np.einsum('is,i,i->s',gamma, prob.maxCapac, solX)
# obj -= np.einsum('ijs,ij,js->s', alpha,solY,prob.demandScen)

# # %%
# # test primera particion

# bc = prob.getBreakCost(Y2, prob.demandScen)
# beta = prob.getBetaDual(Y2,bc)

# prob.currentPartition = np.zeros(prob.nscen)
# newSizePartition=1
# p = 0
# partitionId = prob.currentPartition
# (dualsUnique, inverse) = np.unique(beta, axis=1, return_inverse=True)
# numSubsets = dualsUnique.shape[1]
# if numSubsets > 1:
#   # we add new elements to the partition
#   partitionId[partitionId==p] = (inverse+newSizePartition)
#   # but rename the last one as the current one
#   partitionId[partitionId==(newSizePartition+numSubsets-1)] = p
#   newSizePartition += numSubsets -1
#   #print("Spliting %d into %d new subsets" % (p,numSubsets))
# print("Partition now has %d elements" % newSizePartition)
# prob.currentNScen = newSizePartition
# prob.currentScen = np.zeros((prob.numCustomers,newSizePartition))
# for p in range(newSizePartition):
#   prob.currentScen[:,p] = np.sum(prob.demandScen[:,partitionId==p], axis=1)/np.sum(partitionId==p)

