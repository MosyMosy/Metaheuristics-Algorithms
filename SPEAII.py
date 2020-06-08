import Problem
import numpy as np
import Solution as sl
import math
import matplotlib.pyplot as plt
import NSGAII
import copy
import ParetoUtil as pu
import sys
import SPEA
import MRML
import Metrics as met


class SPEAII:
    def __init__(self, problem, popSize: int, eliteSize: int):
        super().__init__()
        self.problem = problem
        self.popSize = popSize
        self.eliteSize = int(eliteSize)
        self.mattingPoolSize = int(popSize / 2)

    def Evolve(
        self, genertionCount: int, initPopulation: np.array, mutationRate, crossoverRate,useMRML, FitnessInd, GDInd, IGDInd,EIInd, HVInd
    ):
        # 1
        population = copy.deepcopy(initPopulation)
        elites = None
        
        result = np.full([genertionCount,7],10,dtype=float)

        for g in range(genertionCount):
            newElits = None
            #2 Fitness Assignment
            allPop = sl.solution.AddPopulations(population,elites)
            SPEAII.seFitness(allPop)

            #3 Environmental selection
            
            allPop = SPEAII.SortByFitness(allPop)
            # add non-dominated to newElites
            for i in range(np.size(allPop)):
                if  allPop[i].fitness < 1:
                    newElits = sl.solution.AddPopulations(newElits,allPop[i])
            if  np.size(newElits)< self.eliteSize:
                newElits = sl.solution.AddPopulations(newElits,allPop[np.size(newElits):np.size(newElits) + (self.eliteSize - np.size(newElits))])
            elif np.size(newElits)> self.eliteSize:
                newElits = SPEAII.Truncation(newElits,self.eliteSize)
            
            # 4 get the final Pareto front
            
            locations = pu.GetPopulationLocations(newElits)
            result[g] = np.array([g,FitnessInd.compute(locations), GDInd.compute(locations), IGDInd.compute(locations), EIInd.compute(locations), HVInd.compute(locations),met.Delta(locations,self.problem)])
            if g == genertionCount - 1:
                return [newElits,result]
            
            #5 binary tournoment in size population size
            mattingPool = SPEA.SPEA.binaryTournament(newElits, None, int((self.eliteSize/3)*2))
            
            # 5,6 New population by apply matting on matting pool
            if useMRML:            
                population = MRML.matting(
                    mattingPool, self.popSize, self.problem, mutationRate, crossoverRate
                )
                elites = np.copy(newElits)
                print("SPEAII-MRML generation" + str(g))
            else:
                
                population = NSGAII.NSGAII.matting(
                    mattingPool, self.popSize, self.problem, mutationRate, crossoverRate
                )
                elites = np.copy(newElits)                
                print("SPEAII generation" + str(g))
            

        
       


    @staticmethod
    def seFitness(allPop):
        for i in range(np.size(allPop)):
            allPop[i].dominationCoun = 0
            allPop[i].dominatedSolutions = None
            
        k = int(math.sqrt(np.size(allPop)))
        Density = np.zeros_like(allPop,dtype=float)
        distance = np.full((np.size(allPop),np.size(allPop)),1000,dtype=float)
        # set S
        for i in range(np.size(allPop)):
            
            allPop[i].fitness = 0
            for j in range(i):
                if (allPop[i].Dominate(allPop[j])):
                    allPop[i].dominationCount += 1
                    allPop[i].dominatedSolutions= sl.solution.AddPopulations(allPop[i].dominatedSolutions,allPop[j])
                if (allPop[j].Dominate(allPop[i])):
                    allPop[j].dominationCount += 1
                    allPop[j].dominatedSolutions= sl.solution.AddPopulations(allPop[j].dominatedSolutions,allPop[i])
                distance[i,j] = allPop[i].EuclideanDist(allPop[j])
                distance[j,i] = distance[i,j]

            # ste R raw fitness use the fitness as temporal R holder
            if allPop[i].dominatedSolutions is not None:
                for p in allPop[i].dominatedSolutions:
                    p.fitness += allPop[i].dominationCount
        
        distance = np.sort(distance,axis=1)
        # calculate final fitness
        for i in range(np.size(allPop)):            
            Density[i] = 1/(distance[i][k]+2)
            allPop[i].fitness += Density[i] 
             
    
    @staticmethod
    def SortByFitness(pop):
        newpop = np.ndarray.tolist(pop)
        newpop.sort(key=lambda x: x.fitness, reverse=False)
        return np.asarray(newpop, sl.solution)
    
    @staticmethod
    def Truncation(pop,eliteSize):
        
        while np.size(pop) > eliteSize:
            distMat = np.full((np.size(pop),np.size(pop)),sys.float_info.max,dtype=float)
            indexToRemove = -1
            for i in range(np.size(pop)):
                for j in range(np.size(pop)):
                    if i!= j:
                        distMat[i,j] = pop[i].EuclideanDist(pop[j])
            
            distMat.sort(axis=1)
            for j in range(np.size(pop)):
                Cj = np.copy(distMat[:,j])
                Cjtemp = np.copy(Cj)
                Cjtemp.sort()
                indexes = np.where(Cj == Cjtemp[0])
                if  np.size(indexes) == 1:
                    indexToRemove = indexes[0]
            
            pop = np.delete(pop,indexToRemove)
        return pop


        