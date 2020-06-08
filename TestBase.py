
import numpy as np

import NSGAII
import Solution as sl
import SPEAII
import NSPSO
import Problem
import ParetoUtil as pu
import NSPSOMM
import JmetalIndicators as ind
import TestUtil as tu



def main():

    problemlist = [Problem.UF1(),Problem.UF2(),Problem.UF3(),Problem.UF4(), Problem.ZDT1(), Problem.ZDT2(), Problem.ZDT3(), Problem.ZDT4()]
    populationSizelist = [12, 40, 60, 100, 1000]
    generationCountlist = [50, 100, 500, 1000]
    mutationRate = 0.2
    corssoverRate = 0.7
    initSolution = None
    markerAlpha = .3

    # Resultplot("",True,problem,initSolution)
    
    for populationSize in populationSizelist:
        for generationCount in generationCountlist:
            for problem in problemlist:
                locations = []
                locationsinDecision = []
                Fitness = []
                GD = []
                IGD = []
                EI = []
                HV = []
                Delta = []

                locations.append((problem.perfect_pareto_front(),"","-","",1))
                
                prefectFront = problem.perfect_pareto_front()
                FitnessInd = ind.FitnessValue(prefectFront)
                GDInd = ind.GenerationalDistance(prefectFront)
                IGDInd= ind.InvertedGenerationalDistance(prefectFront)
                EIInd= ind.EpsilonIndicator(prefectFront)
                HVInd= ind.HyperVolume(reference_point=[1, 1])

                plotTitle = "Population Size %d, Generation Count %d, Function %s" % (
                    populationSize,
                    generationCount,
                    type(problem).__name__,
                )
                
                initSolution = sl.solution.CreatePopulation(populationSize, problem)
                
                temp = SPEAII.SPEAII(
                    problem, popSize=populationSize, eliteSize=populationSize).Evolve(
                        generationCount, initSolution, mutationRate, corssoverRate,False,FitnessInd, GDInd, IGDInd,EIInd, HVInd)
                locations.append((pu.GetPopulationLocations(temp[0]),"SPEAII","","1",markerAlpha))
                locationsinDecision.append((pu.GetPopulationDecisionLocations(temp[0]),"SPEAII","","1",markerAlpha))
                Fitness.append((temp[1][:,[0,1]],"SPEAII","-","",markerAlpha))
                GD.append((temp[1][:,[0,2]],"SPEAII","-","",markerAlpha))
                IGD.append((temp[1][:,[0,3]],"SPEAII","-","",markerAlpha))
                EI.append((temp[1][:,[0,4]],"SPEAII","-","",markerAlpha))
                HV.append((temp[1][:,[0,5]],"SPEAII","-","",markerAlpha))
                Delta.append((temp[1][:,[0,6]],"SPEAII","-","",markerAlpha))


                temp = SPEAII.SPEAII(
                    problem, popSize=populationSize, eliteSize=populationSize).Evolve(
                        generationCount, initSolution, mutationRate, corssoverRate,True,FitnessInd, GDInd, IGDInd,EIInd, HVInd)
                locations.append((pu.GetPopulationLocations(temp[0]),"SPEAII-MM","","2",markerAlpha))
                locationsinDecision.append((pu.GetPopulationDecisionLocations(temp[0]),"SPEAII-MM","","1",markerAlpha))
                Fitness.append((temp[1][:,[0,1]],"SPEAII-MM","-","",markerAlpha))
                GD.append((temp[1][:,[0,2]],"SPEAII-MM","-","",markerAlpha))
                IGD.append((temp[1][:,[0,3]],"SPEAII-MM","-","",markerAlpha))
                EI.append((temp[1][:,[0,4]],"SPEAII-MM","-","",markerAlpha))
                HV.append((temp[1][:,[0,5]],"SPEAII-MM","-","",markerAlpha))
                Delta.append((temp[1][:,[0,6]],"SPEAII-MM","-","",markerAlpha))


                temp = NSGAII.NSGAII(
                    problem, popSize=populationSize).Evolve(
                        generationCount, initSolution, mutationRate, corssoverRate,False,FitnessInd, GDInd, IGDInd,EIInd, HVInd)
                locations.append((pu.GetPopulationLocations(temp[0]),"NSGAII","","x",markerAlpha))
                locationsinDecision.append((pu.GetPopulationDecisionLocations(temp[0]),"NSGAII","","1",markerAlpha))
                Fitness.append((temp[1][:,[0,1]],"NSGAII","-","",markerAlpha))
                GD.append((temp[1][:,[0,2]],"NSGAII","-","",markerAlpha))
                IGD.append((temp[1][:,[0,3]],"NSGAII","-","",markerAlpha))
                EI.append((temp[1][:,[0,4]],"NSGAII","-","",markerAlpha))
                HV.append((temp[1][:,[0,5]],"NSGAII","-","",markerAlpha))
                Delta.append((temp[1][:,[0,6]],"NSGAII","-","",markerAlpha))                    
                

                temp = NSGAII.NSGAII(
                    problem, popSize=populationSize).Evolve(
                        generationCount, initSolution, mutationRate, corssoverRate,True,FitnessInd, GDInd, IGDInd,EIInd, HVInd)
                locations.append((pu.GetPopulationLocations(temp[0]),"NSGAII-MM","",".",markerAlpha))
                locationsinDecision.append((pu.GetPopulationDecisionLocations(temp[0]),"NSGAII-MM","","1",markerAlpha))
                Fitness.append((temp[1][:,[0,1]],"NSGAII-MM","-","",markerAlpha))
                GD.append((temp[1][:,[0,2]],"NSGAII-MM","-","",markerAlpha))
                IGD.append((temp[1][:,[0,3]],"NSGAII-MM","-","",markerAlpha))
                EI.append((temp[1][:,[0,4]],"NSGAII-MM","-","",markerAlpha))
                HV.append((temp[1][:,[0,5]],"NSGAII-MM","-","",markerAlpha))
                Delta.append((temp[1][:,[0,6]],"NSGAII-MM","-","",markerAlpha))
                
                
                temp = NSPSO.NSPSO(
                    problem, popSize=populationSize
                ).Evolve(generationCount, initSolution,FitnessInd, GDInd, IGDInd,EIInd, HVInd)                  
                locations.append((pu.GetPopulationLocations(temp[0]),"NSPSO","",".",markerAlpha))
                locationsinDecision.append((pu.GetPopulationDecisionLocations(temp[0]),"NSPSO","","1",markerAlpha))
                Fitness.append((temp[1][:,[0,1]],"NSPSO","-","",markerAlpha))
                GD.append((temp[1][:,[0,2]],"NSPSO","-","",markerAlpha))
                IGD.append((temp[1][:,[0,3]],"NSPSO","-","",markerAlpha))
                EI.append((temp[1][:,[0,4]],"NSPSO","-","",markerAlpha))
                HV.append((temp[1][:,[0,5]],"NSPSO","-","",markerAlpha))
                Delta.append((temp[1][:,[0,6]],"NSPSO","-","",markerAlpha))


                temp = NSPSOMM.NSPSOMM(
                    problem, popSize=populationSize
                ).Evolve(generationCount, initSolution,FitnessInd, GDInd, IGDInd,EIInd, HVInd)                  
                locations.append((pu.GetPopulationLocations(temp[0]),"NSPSO-MM","","*",markerAlpha))
                locationsinDecision.append((pu.GetPopulationDecisionLocations(temp[0]),"NSPSO-MM","","1",markerAlpha))
                Fitness.append((temp[1][:,[0,1]],"NSPSO-MM","-","",markerAlpha))
                GD.append((temp[1][:,[0,2]],"NSPSO-MM","-","",markerAlpha))
                IGD.append((temp[1][:,[0,3]],"NSPSO-MM","-","",markerAlpha))
                EI.append((temp[1][:,[0,4]],"NSPSO-MM","-","",markerAlpha))
                HV.append((temp[1][:,[0,5]],"NSPSO-MM","-","",markerAlpha))
                Delta.append((temp[1][:,[0,6]],"NSPSO-MM","-","",markerAlpha))


                
                tu.saveAsText(plotTitle,FitnessInd.get_short_name(),Fitness)
                tu.saveAsText(plotTitle,GDInd.get_short_name(),GD)
                tu.saveAsText(plotTitle,IGDInd.get_short_name(),IGD)
                tu.saveAsText(plotTitle,EIInd.get_short_name(),EI)
                tu.saveAsText(plotTitle,HVInd.get_short_name(),HV)
                tu.saveAsText(plotTitle,"Delta",Delta)

                tu.saveAsText(plotTitle,"Locations",locations)
                tu.saveAsText(plotTitle,"LocationsInDecision",locationsinDecision)
                    
                tu.Resultplot(plotTitle  + " " + FitnessInd.get_short_name() , False, Fitness)
                tu.Resultplot(plotTitle  + " " + GDInd.get_short_name() , False, GD)
                tu.Resultplot(plotTitle  + " " + IGDInd.get_short_name() , False, IGD)
                tu.Resultplot(plotTitle  + " " + EIInd.get_short_name() , False, EI)
                tu.Resultplot(plotTitle  + " " + HVInd.get_short_name() , False, HV)
                tu.Resultplot(plotTitle  + " " + "Delta"               , False, Delta)
                
                tu.Resultplot(plotTitle, False, locations)



if __name__ == "__main__":
    main()
