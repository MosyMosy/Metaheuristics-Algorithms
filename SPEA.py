import Problem
import numpy as np
import Solution as sl
import math
import matplotlib.pyplot as plt
import NSGAII
import copy
import ParetoUtil as pu


class SPEA:
    def __init__(self, problem, popSize: int, eliteSize: int):
        super().__init__()
        self.problem = problem
        self.popSize = popSize
        self.eliteSize = int(eliteSize)
        self.mattingPoolSize = int(popSize / 2)

    def Evolve(
        self, genertionCount: int, initPopulation: np.array, mutationRate, crossoverRate
    ):
        # 1
        population = copy.deepcopy(initPopulation)
        elites = None

        for g in range(genertionCount):

            # 2 add non dominated front of population to elites
            nonDominatedPop = pu.fastNonDominatedSord(population)[0]
            dominatedpop = np.asarray([i for i in population if i not in nonDominatedPop])
            elites = sl.solution.AddPopulations(elites, nonDominatedPop)

            # remove dominated solutions from Elites
            elites = pu.fastNonDominatedSord(elites)[0]

            # reduce the eliets size to max eliteSize
            if np.size(elites) > self.eliteSize:
                # do clustering
                elites = SPEA.Cluster(elites, self.eliteSize)
            """elif np.size(elites) < self.eliteSize:
                # add random solutions from population front > 0
                np.random.shuffle(dominatedpop)
                elites = sl.solution.AddPopulations(
                    elites, dominatedpop[0 : (self.eliteSize - np.size(elites))]
                )"""

            # 3 assign fitness to elites
            elites = SPEA.setSfitness(elites, population)

            # assign fitness to population
            population = SPEA.setFfitness(population, elites)

            # 4 binary tournoment in size population size
            mattingPool = SPEA.binaryTournament(population, elites, self.popSize)

            # 5,6 New population by apply matting on matting pool
            population = NSGAII.NSGAII.matting(
                mattingPool, self.popSize, self.problem, mutationRate, crossoverRate
            )
            print("SPEA generation" + str(g))

        # 7 get the final Pareto front
        population = pu.fastNonDominatedSord(elites)[0]
        return population

    @staticmethod
    def binaryTournament(population, elites, mattingPoolSize:int):
        candidatePool = sl.solution.AddPopulations(population, elites)
        mattingPool = None
        mattingIndex = 0
        while True:
            index1 = np.random.randint(0, np.size(candidatePool))
            index2 = np.random.randint(0, np.size(candidatePool))

            candidateSolution = None
            if candidatePool[index1].fitness <= candidatePool[index2].fitness:
                candidateSolution = candidatePool[index1]
            else:
                candidateSolution = candidatePool[index2]
            if sl.solution.isRepeated(mattingPool, candidateSolution) == False:
                if type(mattingPool).__module__ == np.__name__:
                    mattingPool = sl.solution.AddPopulations(
                        mattingPool, candidateSolution
                    )
                else:
                    mattingPool = np.array([candidateSolution])
            mattingIndex += 1
            if np.size(mattingPool) == mattingPoolSize:
                break

        return mattingPool

    
    @staticmethod
    def setSfitness(elites, population):
        for e in elites:
            e.fitness = -1
            dominated = 0
            for p in population:
                if e.Dominate(p):
                    dominated += 1
            e.fitness = dominated / (np.size(population) + 1)
        return elites

    @staticmethod
    def setFfitness(population, elites):
        for p in population:
            p.fitness
            sSum = 0
            for e in elites:
                if e.Dominate(p):
                    sSum += e.fitness
            p.fitness = 1 + sSum
        return population

    @staticmethod
    def Cluster(elites, finalSize):
        while np.size(elites, 0) > finalSize:
            index1, index2 = SPEA.NearClusters(elites)
            elites[index1] = np.append(elites[index1], elites[index2])
            elites = [elites[i] for i in range(np.size(elites, 0)) if i != index2]

        result = np.empty([finalSize], dtype=sl.solution)
        for i in range(finalSize):
            result[i] = SPEA.clusterCenter(elites[i])
        return result

    @staticmethod
    def clusterDistance(cluster1, cluster2):
        c1_cardinality = np.size(cluster1)  # cluster 1 cardinality
        c2_cardinality = np.size(cluster2)  # cluster 2 cardinality
        sum = 0.0
        for i in range(c1_cardinality):
            for j in range(c2_cardinality):
                soluton1 = cluster1 if (type(cluster1) == sl.solution) else cluster1[i]
                soluton2 = cluster2 if (type(cluster2) == sl.solution) else cluster2[j]
                sum += soluton1.EuclideanDist(soluton2)

        constant = 1.0 / (c1_cardinality * c2_cardinality)
        return sum * constant

    @staticmethod
    def NearClusters(population):
        min_distance = None
        nearest_clusters = None
        penultimate = np.size(population, 0) - 1
        for i in range(penultimate):
            for j in range(i + 1, np.size(population, 0)):
                d = SPEA.clusterDistance(population[i], population[j])
                if min_distance is None:
                    min_distance = d
                    nearest_clusters = np.array([i, j])
                elif min_distance is not None and (d < min_distance):
                    min_distance = d
                    nearest_clusters = np.array([i, j])
        return nearest_clusters[0], nearest_clusters[1]

    @staticmethod
    def clusterCenter(cluster):
        if type(cluster) == sl.solution:
            return cluster

        min_avg_dist = None
        index = None
        for i in range(np.size(cluster)):
            dist_sum = 0.0
            for j in range(np.size(cluster)):
                dist_sum += cluster[i].EuclideanDist(cluster[j])
            avg_dist = dist_sum / (np.size(cluster) - 1)
            if i == 0:
                min_avg_dist = avg_dist
                index = i
            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                index = i
        centroid = cluster[index]
        return centroid
