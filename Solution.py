import Problem
import numpy as np
import copy
import math
import random


class solution:
    def __init__(self, problem: Problem.ProblemBase):
        self.fitness = -1  # for SPEA algorithm
        self.crowdingDistance = -1
        self.ManifoldDistance = -1
        self.gene = problem.createGene()
        self.velocity = None # for MOPSO algorithm
        self.Pbest = None
        self.dominationCount = 0
        self.dominatedSolutions = None
        self.dominanceRank = 100
        self.problem = problem
        self.objectives = []
        self.isPosChanged = True
    
    def GetObjective(self,index = -1):
        if self.isPosChanged:
            self.objectives = [self.problem.objectives[i](self.gene) for i in range(len(self.problem.objectives))]
            self.isPosChanged = False
        if index == -1:            
            return self.objectives
        else:
            return self.objectives[index]
        

        

    def Mutate(self):
        return self.PolynomialMutation(1)
        temp = solution(self.problem)
        temp.gene = copy.deepcopy(self.gene)
        index1 = np.random.randint(1, self.problem.geneSize)
        temp.gene[index1] = np.random.uniform(
            self.problem.variableLowerBound, self.problem.variableUpperBound, size=(1)
        )
        return temp

    # inspired from https://github.com/DEAP/deap/blob/master/deap/tools/mutation.py
    def PolynomialMutation(self, indpb):
        eta = 1 
        temp = solution(self.problem)
        temp.gene = copy.deepcopy(self.gene)

        size = len(temp.gene)
        low = [self.problem.variableLowerBound] * size
        up = [self.problem.variableUpperBound] * size
        low[0] = self.problem.x0LowerBound       
        up[0] = self.problem.x0UpperBound

        for i, xl, xu in zip(range(size), low, up):
            if random.random() <= indpb:
                x = temp.gene[i]
                delta_1 = (x - xl) / (xu - xl)
                delta_2 = (xu - x) / (xu - xl)
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.0)

                if rand < 0.5:
                    xy = 1.0 - delta_1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                    delta_q = 1.0 - val ** mut_pow

                x = x + delta_q * (xu - xl)
                x = min(max(x, xl), xu)
                temp.gene[i] = x
        temp.ApplyVarialbleConstrains()
        return temp

    def CrossoverWith(self, nextParent):
        return self.SBX(nextParent)
        child1 = solution(self.problem)
        child1.gene = copy.deepcopy(self.gene)

        child2 = solution(self.problem)
        child2.gene = copy.deepcopy(nextParent.gene)

        for i in range(self.problem.geneSize):
            if np.random.rand() > 0.5:
                child1.gene[i] = self.gene[i]
                child2.gene[i] = nextParent.gene[i]
            else:
                child1.gene[i] = nextParent.gene[i]
                child2.gene[i] = self.gene[i]

        return child1.Mutate(), child2.Mutate()

    # simulated binary crossover inspired from https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py
    def SBX(self, nextParent):
        """Crowding degree of the crossover. A high eta will produce
        children resembling to their parents, while a small eta will
        produce solutions much more different."""
        eta = 1
        child1 = solution(self.problem)
        child1.gene = copy.deepcopy(self.gene)

        child2 = solution(self.problem)
        child2.gene = copy.deepcopy(nextParent.gene)

        for i, (x1, x2) in enumerate(zip(child1.gene, child2.gene)):
            rand = random.random()
            if rand <= 0.5:
                beta = 2.0 * rand
            else:
                beta = 1.0 / (2.0 * (1.0 - rand))
            beta **= 1.0 / (eta + 1.0)
            child1.gene[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))
            child2.gene[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))

        child1.ApplyVarialbleConstrains()
        child2.ApplyVarialbleConstrains()
        return child1, child2

    def Dominate(self, solution):
        _fistcons = True
        _secocons = False
        selfObjectives = self.GetObjective()
        solutionObjectives = solution.GetObjective()
        for i in range(len(selfObjectives)):        
            if (selfObjectives[i] > solutionObjectives[i]):
                _fistcons = False
            if (selfObjectives[i] < solutionObjectives[i]):
                _secocons = True

        return (_fistcons and _secocons)

    def ApplyVarialbleConstrains(self):
        # x0 Constrains
        self.gene[0] = np.clip(self.gene[0], self.problem.x0LowerBound, self.problem.x0UpperBound)
        # xi Constrains
        self.gene[1:] = np.clip(self.gene[1:], self.problem.variableLowerBound, self.problem.variableUpperBound)
        self.isPosChanged = True
    
    
    def EuclideanDist(self, solution):
        point1 = self.GetObjective()
        point2 = solution.GetObjective()
        """return np.linalg.norm(point1 - point2)"""
        dist = [(a - b) ** 2 for a, b in zip(point1, point2)]
        dist = math.sqrt(sum(dist))
        return dist

    def EuclideanDistInSearchSpace(self, solution):
        return np.linalg.norm(self.gene - solution.gene)

    @staticmethod
    def CreatePopulation(popSize: int, problem):
        temp = np.empty([popSize], dtype=solution)
        for i in range(popSize):
            temp[i] = solution(problem)
        return temp

    @staticmethod
    def AddPopulations(pop1, pop2):
        if pop1 is None:
            if type(pop2).__module__ == np.__name__:
                return pop2
            else:
                return np.array([pop2])
        if pop2 is None:
            if type(pop1).__module__ == np.__name__:
                return pop1
            else:
                return np.array([pop1])
        return np.append(pop1, pop2)
    
    @staticmethod
    def isRepeated(population: np.array, sl):
        if np.size(population) == 1:
            if population == None:
                return False

        for i in range(np.size(population)):
            if sl is population[i]:
                return True
        return False
