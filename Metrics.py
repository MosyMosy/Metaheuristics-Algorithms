import numpy as np
from scipy import spatial
import ParetoUtil  as pu

def GD( population):
    if type(population).__module__ == np.__name__:
        if np.size(population)>0:
            trueLocations = np.asarray(population[0].problem.perfect_pareto_front())
            solutionsLocations = pu.GetPopulationLocations(population)
            distances = spatial.distance.cdist(solutionsLocations, trueLocations)
            return np.mean(np.min(distances, axis=1))
    raise Exception 

def IGD( population):
    if type(population).__module__ == np.__name__:
        if np.size(population)>0:
            trueLocations = np.asarray(population[0].problem.perfect_pareto_front())
            solutionsLocations = pu.GetPopulationLocations(population)
            distances = spatial.distance.cdist(trueLocations, solutionsLocations)
            return np.mean(np.min(distances, axis=1))
    raise Exception 

def FitnessValue( population): 
    if type(population).__module__ == np.__name__:
        if np.size(population)>0:
            solutionsLocations = pu.GetPopulationLocations(population)   
            return np.mean(solutionsLocations)
    raise Exception

def Delta(population,problem):
    if type(population).__module__ == np.__name__:
        if np.size(population)>0:
            trueLocations = problem.perfect_pareto_front()
            solutionsLocations = population
            manDistances = spatial.distance.cdist(trueLocations, solutionsLocations,metric='cityblock')
            manDistances.sort(axis=1)
            
            # dem is the distancebetween the extreme solutions of the true Pareto-optimal set P∗ and Q on the m-th objective
            d_me = manDistances[:,1].sum()

            distances = spatial.distance.cdist(solutionsLocations, solutionsLocations)
            mean = np.mean(distances)

            # distance between two neighbouring solutions in the nondominated solution set
            d_i = np.sum(distances - mean)

            return (d_me + d_i)/(d_me + (mean * solutionsLocations.shape[0]))



