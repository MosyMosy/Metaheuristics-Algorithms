import Problem
import numpy as np
import Solution as sl
import math
import matplotlib.pyplot as plt
import copy
import ParetoUtil as pu
import MRML
import Metrics as met


class NSGAII():
    def __init__(self, problem, popSize: int):
        super().__init__()
        self.problem = problem
        self.popSize = popSize

    def Evolve(self, genertionCount: int, initPopulation: np.array, mutationRate, crossoverRate,useMRML, FitnessInd, GDInd, IGDInd,EIInd, HVInd):
        # initiate early sets
        pop = copy.deepcopy(initPopulation)
        Q = NSGAII.matting(pop, self.popSize, self.problem, mutationRate, crossoverRate)
        R = sl.solution.AddPopulations(pop,Q)
        F =  pu.fastNonDominatedSord(R)

        result = np.full([genertionCount,7],10,dtype=float)
        
        for g in range(genertionCount):            
            pop = None
            i = 0            
            #Make new population
            # while |Fi| + |Pt+1| <= popSize
            
            while (np.size(F[i]) + (np.size(pop) if (type(pop).__module__ == np.__name__) else 0)) <= self.popSize:
                # ignor crowding distance calculation for all
                pop =sl.solution.AddPopulations(pop,F[i])
                
                i += 1
            F[i] = pu.sortSolutionsByCrowdingDist(F[i])
            pop = sl.solution.AddPopulations(pop,F[i][0:(self.popSize - np.size(pop))])
            # Make new Q
            if useMRML:
                Q = MRML.matting(pop, self.popSize, self.problem, mutationRate, crossoverRate)
            else:
                Q = NSGAII.matting(pop, self.popSize, self.problem, mutationRate, crossoverRate)
            R = sl.solution.AddPopulations(pop,Q)
            F =  pu.fastNonDominatedSord(R)
            
            if useMRML:
                print("NSGAII-MRML generation" + str(g))
            else:
                print("NSGAII generation" + str(g))            
            
            locations = pu.GetPopulationLocations(pop)
            result[g] = np.array([g,FitnessInd.compute(locations), GDInd.compute(locations), IGDInd.compute(locations), EIInd.compute(locations), HVInd.compute(locations),met.Delta(locations,self.problem)])

        # get the final Pareto front
        return [pop,result]
        

    
    @staticmethod
    def matting(population, generationSize, problem, mutationRate, crossoverRate):
        newPopulation = np.empty([1], dtype=sl.solution)
        np.random.shuffle(population)

        mutationCount = int((np.size(population)) * mutationRate)
        crossoverCount = int((np.size(population)) * crossoverRate)

        difference = generationSize - (mutationCount + crossoverCount)
        for i in range(0, difference, -1):
            mutationCount -= 1
            crossoverCount -= 1

        # Make crossoverCount to even
        crossoverCount = crossoverCount if(
            crossoverCount % 2 == 0) else crossoverCount-1

        # Filling newPopulation with mutate crossover and new
        for i in range(0, mutationCount):
            newPopulation = np.append(newPopulation, population[i].Mutate())
        for i in range(mutationCount, (mutationCount + crossoverCount), 2):
            newPopulation = np.append(
                newPopulation, np.c_[population[i].CrossoverWith(population[1])])

        difference = generationSize - (mutationCount + crossoverCount)
        for i in range(difference):
            newPopulation = np.append(newPopulation, sl.solution(problem))

        return newPopulation[1:]
