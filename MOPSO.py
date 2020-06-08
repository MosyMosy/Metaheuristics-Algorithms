import math
import copy

import matplotlib.pyplot as plt
import numpy as np

import Solution as sl
import Problem
import ParetoUtil as pu


class MOPSO():
    def __init__(self, problem, popSize: int, repSize: int):
        super().__init__()
        self.problem = problem
        self.popSize = popSize
        self.repSize = int(repSize)
        self.numberOfHyperCubesPerRow = 5

    def Evolve(self, genertionCount: int, initPopulation: np.array, mutationRate, crossoverRate):
        # 1, 2, 3, 6
        population = copy.deepcopy(initPopulation)
        repository = None

        # 4 Add non-dominated front of population to repository
        repository = pu.fastNonDominatedSord(population)[0] 

        # remove dominated ones from repository
        repository = pu.fastNonDominatedSord(repository)[0]        

        # remove extra based on reverse hypercube crowdness
        if(np.size(repository)) > self.repSize:
            repository = MOPSO.sortBasedOnHyperCubeCrowd(
                repository, self.numberOfHyperCubesPerRow)
            repository = repository[0: self.repSize]

        for g in range(genertionCount):
            # calculate the velocity
            w = .4
            for i in range(self.popSize):
                R1, R2 = np.random.rand(), np.random.rand()
                h = MOPSO.getSolutionIndexFromCrowdCube(
                    repository, self.numberOfHyperCubesPerRow)
                population[i].velocity = (w * population[i].velocity) +\
                                         (R1 * (population[i].Pbest.gene - population[i].gene)) +\
                                         (repository[h].gene - population[i].gene)

            # calculate new location
            for i in range(self.popSize):
                population[i].gene = population[i].gene + \
                    population[i].velocity
                # Apply varialble constrain after position update
                population[i].ApplyVarialbleConstrains()
                # update Pbest
                if (population[i].Dominate(population[i].Pbest)):
                    population[i].Pbest=copy.deepcopy(population[i].gene)

            # Add non-dominated front of population to repository with duplicate remove
            nonDominatedPop = pu.fastNonDominatedSord(population)[0]
            for i in range(np.size(nonDominatedPop)):
                isRepeated = False
                for j in range(np.size(repository)):
                    if nonDominatedPop[i] is repository[j]: isRepeated = True
                if isRepeated == False: 
                    repository=sl.solution.AddPopulations(repository, nonDominatedPop[i])


            # remove dominated ones from repository
            repository= pu.fastNonDominatedSord(repository)[0]
            
            # remove extra based on reverse hyperCube crowdness
            if(np.size(repository)) > self.repSize:
                repository=MOPSO.sortBasedOnHyperCubeCrowd(
                    repository, self.numberOfHyperCubesPerRow)
                repository=repository[0:self.repSize]

            print("MSPSO generation" + str(g))

        
        # get the final Pareto front        
        return repository


    @staticmethod
    def assignSolutionToHyperCube(solution, numberOfHyperCubesPerRow):
        hyperCubeWidth=1 / numberOfHyperCubesPerRow
        f1Cordination=int(solution.GetObjective(0) / hyperCubeWidth)
        f2Cordination=int(solution.GetObjective(1) / hyperCubeWidth)

        return (f2Cordination * numberOfHyperCubesPerRow) + f1Cordination

    @staticmethod
    def assignPopulationToHyperCube(population, numberOfHyperCubesPerRow):
        popSize=np.size(population)
        hyperCubeLocation=np.empty([popSize], dtype=int)
        for i in range(popSize):
            hyperCubeLocation[i]=MOPSO.assignSolutionToHyperCube(
                population[i], numberOfHyperCubesPerRow)
        return hyperCubeLocation

    @staticmethod
    def getSolutionIndexFromCrowdCube(population, numberOfHyperCubesPerRow):
        hyperCubeLocation=MOPSO.assignPopulationToHyperCube(
            population, numberOfHyperCubesPerRow)
        HyperCubeCrowdnessProbability=np.empty(
            [np.size(hyperCubeLocation)], dtype=float)
        # assign crowdness Probability
        for i in range(np.size(hyperCubeLocation)):
            HyperCubeCrowdnessProbability[i]=np.size(
                np.where(hyperCubeLocation == hyperCubeLocation[i]))
        sumOfCrowdness=np.sum(HyperCubeCrowdnessProbability)
        HyperCubeCrowdnessProbability=HyperCubeCrowdnessProbability/sumOfCrowdness
        # simulate the roulette wheel selection
        return np.random.choice(np.size(hyperCubeLocation), 1, p=HyperCubeCrowdnessProbability)[0]

    @staticmethod
    def sortBasedOnHyperCubeCrowd(population, numberOfHyperCubesPerRow):
        hyperCubeLocation=MOPSO.assignPopulationToHyperCube(
            population, numberOfHyperCubesPerRow)
        # find crowdest location
        largestCrowd=0
        crowdestLocation=0
        for i in range(np.size(population)):
            population[i].fitness=np.size(np.where(hyperCubeLocation == hyperCubeLocation[i]))

        _list = np.ndarray.tolist(population)
        _list.sort(key=lambda x: x.fitness, reverse=False)
        return np.asarray(_list, sl.solution)
