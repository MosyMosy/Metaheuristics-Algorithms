import numpy as np
import Solution as sl
import sys


def fastNonDominatedSord(population):
    fronts = []
    fronts.append([]) 
    for solution in population:
        solution.dominationCount = 0
        solution.dominatedSolutions = set()
        
        for other_individual in population:
            if solution.Dominate(other_individual):
                solution.dominatedSolutions.add(other_individual)
            elif other_individual.Dominate(solution):
                solution.dominationCount += 1
        if solution.dominationCount == 0:
            fronts[0].append(solution)
            solution.dominanceRank = 0
    i = 0
    while True:
        temp = []
        for solution in fronts[i]:
            for other_individual in solution.dominatedSolutions:
                other_individual.dominationCount -= 1
                if other_individual.dominationCount == 0:
                    other_individual.dominanceRank = i+1
                    temp.append(other_individual)
        i = i+1
        if len(temp)>0:
            fronts.append(temp)
        else:
            break
    npFront = []
    for j in range(np.size(fronts,0)):
        npFront.append(np.asarray(fronts[j]))
    return np.asarray(npFront)




def getFront(population, level: int):
    _list = []
    for i in range(np.size(population)):
        if population[i].dominated == level:
            _list.append(population[i])
    return np.asarray(_list, dtype=sl.solution)


def GetPopulationLocations(population):
    if (type(population).__module__ == np.__name__):
        dimentions = len(population[0].problem.objectives)
        locations = np.empty([np.size(population), dimentions], dtype=float)
        for i in range(np.size(population)):
            locations[i] = np.asarray(population[i].GetObjective())# np.c_[population[i].F1, population[i].F2]
        return locations
    else:
        return None

def GetPopulationDecisionLocations(population):
    if (type(population).__module__ == np.__name__):
        dimentions = population[0].gene.shape[0]
        locations = np.empty([np.size(population), dimentions])
        for i in range(np.size(population)):
            locations[i] = np.asarray(population[i].gene)# np.c_[population[i].F1, population[i].F2]
        return locations
    else:
        return None
    

def sortSolutionsByCrowdingDist(pop):
    front = np.ndarray.tolist(pop)
    if len(front) > 0:
        solutions_num = len(front)
        for individual in front:
            individual.crowdingDistance = 0
        objectiveCount = len(front[0].problem.objectives)
        for i in range(objectiveCount):
            # for F1
            front.sort(key=lambda x: x.GetObjective(i), reverse=False)
            FMin = front[0].GetObjective(i)
            FMax = front[solutions_num-1].GetObjective(i)
            front[0].crowdingDistance = sys.maxsize
            front[solutions_num-1].crowdingDistance = sys.maxsize
            for index in range(1,solutions_num-1):
                front[index].crowdingDistance += abs(front[index+1].GetObjective(i) - front[index-1].GetObjective(i)) / ((FMax - FMin) + sys.float_info.epsilon)

    for f in front:
        f.crowdingDistance = min(f.crowdingDistance,sys.maxsize)

    front.sort(key=lambda x: x.crowdingDistance, reverse=True)
    return np.asarray(front, sl.solution)


def sortSolutionsByManifoldCrowdingDist(pop):
    front = np.ndarray.tolist(pop)
    if len(front) > 0:
        solutions_num = len(front)
        for i in range(len(front)):
            front[i].ManifoldDistance = 0
        
        objectiveCount = len(front[0].problem.objectives)        
        for i in range(objectiveCount):
            # for F1
            front.sort(key=lambda x: x.GetObjective(i), reverse=False)
            FMin = front[0].GetObjective(i)
            FMax = front[solutions_num-1].GetObjective(i)
            front[0].ManifoldDistance = sys.maxsize
            front[solutions_num-1].ManifoldDistance = sys.maxsize
            for index in range(1,solutions_num-1):
                front[index].ManifoldDistance += front[index+1].EuclideanDistInSearchSpace(front[index-1]) / ((FMax - FMin) + sys.float_info.epsilon)
            
    for f in front:
        f.ManifoldDistance = min(f.ManifoldDistance,sys.maxsize)

               
    front.sort(key=lambda x: x.ManifoldDistance, reverse=True)
    return np.asarray(front, sl.solution)



    
                        
               