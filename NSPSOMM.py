import math
import copy

import matplotlib.pyplot as plt
import numpy as np

import Solution as sl
import Problem
import ParetoUtil as pu
import Metrics as met
import MRML as mm


class NSPSOMM():
    def __init__(self, problem, popSize: int):
        super().__init__()
        self.problem = problem
        self.popSize = popSize


    def Evolve(self, genertionCount: int, initPopulation: np.array, FitnessInd, GDInd, IGDInd,EIInd, HVInd):
        # 1, 2
        PSOList = copy.deepcopy(initPopulation)
        for i in range(PSOList.shape[0]):
            PSOList[i].velocity = self.problem.createGene()
            if(np.random.rand() < .5):
                PSOList[i].velocity *= -1
            PSOList[i].Pbest = np.copy(PSOList[i].gene)
        nonDomPSOList = None
        #Vmax = self.problem.variableUpperBound
        
        

        result = np.full([genertionCount,7],10,dtype=float)

        for g in range(genertionCount):   
            # 3 Add non-dominated front of population to nonDomPSOList
            PSOListFronts = pu.fastNonDominatedSord(PSOList)
            nonDomPSOList = PSOListFronts[0]            
            #4,5
            
            #6
            nextPopList = np.empty(PSOList.shape[0] * 2,dtype=sl.solution)            
            #w was gradually decreased from 1.0 to 0.4.
            w = 1 - (g/(genertionCount - 1)) * .6
            
            i = 0            
            for f in range(PSOListFronts.shape[0]): 
                prevFrontIndex = max(0, f-1)
                prevFront = pu.sortSolutionsByManifoldCrowdingDist (PSOListFronts[prevFrontIndex])
                randGbestIndex = np.random.randint(0,max(1,int(prevFront.shape[0]/100*5)+1), size=(1))[0]
                randGbest = prevFront[randGbestIndex]

                #get pairs with low manifold difference
                pairList = mm.MattingSelection2(PSOListFronts[f])               
                for pair in pairList:
                    R1, R2 = np.random.rand(), np.random.rand()  
                    C1,C2 = 2,2 

                    isTwinPair = True
                    if pair[1] is None:
                        pair = (pair[0],pair[0]) 
                        isTwinPair = False

                    newSl0 = sl.solution(self.problem)
                    newSl1 = sl.solution(self.problem)
                    
                    # b) calculate new velocity   
                    
                    newSl0.velocity  = (w * pair[0].velocity) +\
                                                (C1 * R1 * (pair[0].Pbest - pair[0].gene)) +\
                                                (C2 * R2 * (randGbest.gene - pair[0].gene))
                    newSl1.velocity  = (w * pair[1].velocity) +\
                                                (C1 * R1 * (pair[1].Pbest - pair[1].gene)) +\
                                                (C2 * R2 * (randGbest.gene - pair[1].gene))                    
                    self.ApplyVelocityConstrains(newSl0.velocity )
                    self.ApplyVelocityConstrains(newSl1.velocity )
                    
                    # calculate new location
                    meanV = (newSl0.velocity + newSl1.velocity)/2
                    randCo =np.random.choice([1,-1],1)[0]
                    newVol0 = newSl0.velocity + meanV * randCo
                    newVol1 = newSl1.velocity + meanV * -randCo  
                    self.ApplyVelocityConstrains(newVol0 )
                    self.ApplyVelocityConstrains(newVol1 )                  
                    
                    newSl0.gene = pair[0].gene + newVol0
                    newSl1.gene = pair[1].gene + newVol0
                    newSl0.ApplyVarialbleConstrains() 
                    newSl1.ApplyVarialbleConstrains()
                    
                    
                    # update pbest
                    newSl0.Pbest = newSl0.gene
                    newSl1.Pbest = newSl1.gene
                    NonDominatedResult = pu.fastNonDominatedSord(np.array([pair[0],pair[1],newSl0,newSl1]))[0]
                    if NonDominatedResult.shape[0] > 1:                        
                        localbest = pu.sortSolutionsByManifoldCrowdingDist(NonDominatedResult)[0]
                    else:
                        localbest = NonDominatedResult[0]
                    newSl0.Pbest = localbest.gene
                    #pair[0].Pbest = localbest.gene                                           
                    newSl1.Pbest = localbest.gene
                    #pair[1].Pbest = localbest.gene
                        
                                    
                    
                    nextPopList[2*i] = pair[0]
                    nextPopList[2*i + 1] = newSl0
                    i +=1
                    if isTwinPair:
                        nextPopList[2*i] = pair[1]
                        nextPopList[2*i + 1] = newSl1
                        i +=1

            #7 
            # modified dont copy repeated to nextPopListRest
            F = pu.fastNonDominatedSord(nextPopList)
            
            nonDomPSOList = F[0]
            
            nextPopListRest = None
            for el in F[1:]:
                for iel in el:
                    nextPopListRest = sl.solution.AddPopulations(nextPopListRest,iel)            

            
            #8
            PSOList = None
            # 9 
            np.random.shuffle(nonDomPSOList)
            PSOList = nonDomPSOList[: min(nonDomPSOList.shape[0] , self.popSize)]

            #10            
            while (PSOList.shape[0] < self.popSize): # and nextPopListRest is not None:
                #a
                nextNonDomList = pu.fastNonDominatedSord(nextPopListRest)[0]
                #b                
                PSOList = sl.solution.AddPopulations(PSOList,nextNonDomList[: min(nextNonDomList.shape[0],(self.popSize - PSOList.shape[0]))])
                if (PSOList.shape[0] == self.popSize):
                    break;
                #c
                nextPopListRestCopy = np.copy(nextPopListRest)
                nextPopListRest = None
                #d
                Fd = pu.fastNonDominatedSord(nextPopListRestCopy)[1:]
                nextPopListRest = None
                
                for el in Fd:
                    for iel in el:
                        nextPopListRest = sl.solution.AddPopulations(nextPopListRest,iel)
                
                
            #11            
            locations = pu.GetPopulationLocations(PSOList)
            result[g] = np.array([g,FitnessInd.compute(locations), GDInd.compute(locations), IGDInd.compute(locations), EIInd.compute(locations), HVInd.compute(locations),met.Delta(locations,self.problem)])
            print(type(self).__name__ + "generation" + str(g))

        
        # get the final Pareto front        
        return [PSOList,result]
    
    def ApplyVelocityConstrains(self,velocity):
            # x0 Constrains
            velocity[0] = np.clip(velocity[0], -self.problem.x0UpperBound, self.problem.x0UpperBound)
            # xi Constrains
            velocity[1:] = np.clip(velocity[1:], -self.problem.variableUpperBound, self.problem.variableUpperBound)
