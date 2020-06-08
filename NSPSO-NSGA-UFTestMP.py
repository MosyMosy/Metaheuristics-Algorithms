"""import matplotlib.pyplot as plt
import numpy as np

import NSGAII
import Solution as sl
import SPEAII
import NSPSO
import Problem
import ParetoUtil as pu
import NSPSOMM
import concurrent.futures



def main():

    problemlist = [Problem.UF1(),Problem.UF2(),Problem.UF3(),Problem.UF4()] #Problem.ZDT1(), Problem.ZDT2(), Problem.ZDT3(), Problem.ZDT4(),
    populationSizelist = [40]#[12, 40, 60, 100, 1000]
    generationCountlist = [100]#[50, 100, 500, 1000]
    mutationRate = 0.2
    corssoverRate = 0.7
    initSolution = None

    # Resultplot("",True,problem,initSolution)

    for populationSize in populationSizelist:
        for generationCount in generationCountlist:
            
            for i in range(1): 
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    NSGAMMResult = []
                    SPEAMMResult = []
                    NSPSOResult = []
                    NSPSOMMResult = []
                    
                    initSolution = sl.solution.CreatePopulation(populationSize, problemlist[0])                    
                    futureNSGAMM1 = executor.submit(NSGAII.NSGAII(problemlist[0], popSize=populationSize).Evolve, generationCount, initSolution, mutationRate, corssoverRate,True)
                    futureSPEAMM1 = executor.submit(SPEAII.SPEAII(problemlist[0], popSize=populationSize, eliteSize=populationSize).Evolve, generationCount, initSolution, mutationRate, corssoverRate,True)
                    futureNSPSO1 = executor.submit(NSPSO.NSPSO(problemlist[0], popSize=populationSize).Evolve, generationCount, initSolution)
                    futureNSPSOMM1 = executor.submit(NSPSOMM.NSPSOMM(problemlist[0], popSize=populationSize).Evolve, generationCount, initSolution)
                    
                    initSolution = sl.solution.CreatePopulation(populationSize, problemlist[1])  
                    futureNSGAMM2 = executor.submit(NSGAII.NSGAII(problemlist[1], popSize=populationSize).Evolve, generationCount, initSolution, mutationRate, corssoverRate,True)
                    futureSPEAMM2 = executor.submit(SPEAII.SPEAII(problemlist[1], popSize=populationSize, eliteSize=populationSize).Evolve, generationCount, initSolution, mutationRate, corssoverRate,True)
                    futureNSPSO2 = executor.submit(NSPSO.NSPSO(problemlist[1], popSize=populationSize).Evolve, generationCount, initSolution)
                    futureNSPSOMM2 = executor.submit(NSPSOMM.NSPSOMM(problemlist[1], popSize=populationSize).Evolve, generationCount, initSolution)
                    
                    initSolution = sl.solution.CreatePopulation(populationSize, problemlist[2])  
                    futureNSGAMM3 = executor.submit(NSGAII.NSGAII(problemlist[2], popSize=populationSize).Evolve, generationCount, initSolution, mutationRate, corssoverRate,True)
                    futureSPEAMM3 = executor.submit(SPEAII.SPEAII(problemlist[2], popSize=populationSize, eliteSize=populationSize).Evolve, generationCount, initSolution, mutationRate, corssoverRate,True)
                    futureNSPSO3 = executor.submit(NSPSO.NSPSO(problemlist[2], popSize=populationSize).Evolve, generationCount, initSolution)
                    futureNSPSOMM3 = executor.submit(NSPSOMM.NSPSOMM(problemlist[2], popSize=populationSize).Evolve, generationCount, initSolution)
                    
                    initSolution = sl.solution.CreatePopulation(populationSize, problemlist[3])  
                    futureNSGAMM4 = executor.submit(NSGAII.NSGAII(problemlist[3], popSize=populationSize).Evolve, generationCount, initSolution, mutationRate, corssoverRate,True)
                    futureSPEAMM4 = executor.submit(SPEAII.SPEAII(problemlist[3], popSize=populationSize, eliteSize=populationSize).Evolve, generationCount, initSolution, mutationRate, corssoverRate,True)
                    futureNSPSO4 = executor.submit(NSPSO.NSPSO(problemlist[3], popSize=populationSize).Evolve, generationCount, initSolution)
                    futureNSPSOMM4 = executor.submit(NSPSOMM.NSPSOMM(problemlist[3], popSize=populationSize).Evolve, generationCount, initSolution)
                    
                    NSGAMMResult.append(futureNSGAMM1.result())
                    SPEAMMResult.append(futureSPEAMM1.result())
                    NSPSOResult.append(futureNSPSO1.result())
                    NSPSOMMResult.append(futureNSPSOMM1.result())

                    NSGAMMResult.append(futureNSGAMM2.result())
                    SPEAMMResult.append(futureSPEAMM2.result())
                    NSPSOResult.append(futureNSPSO2.result())
                    NSPSOMMResult.append(futureNSPSOMM2.result())

                    NSGAMMResult.append(futureNSGAMM3.result())
                    SPEAMMResult.append(futureSPEAMM3.result())
                    NSPSOResult.append(futureNSPSO3.result())
                    NSPSOMMResult.append(futureNSPSOMM3.result())

                    NSGAMMResult.append(futureNSGAMM4.result())
                    SPEAMMResult.append(futureSPEAMM4.result())
                    NSPSOResult.append(futureNSPSO4.result())
                    NSPSOMMResult.append(futureNSPSOMM4.result())
                        
                for i in range(len(problemlist)):
                    locations =[]
                    Delta = []
                    GD = []
                    IGD = []

                    locations.append((problemlist[i].perfect_pareto_front(),"","-","",1))
                    plotTitle = "Population Size %d, Generation Count %d, Function %s" % (
                        populationSize,
                        generationCount,
                        type(problemlist[i]).__name__,
                    )
                    with open( "NSPSO-NSGA-UFTest.txt", "a") as text_file:
                        print(plotTitle, file=text_file)
                    
                    locations.append((pu.GetPopulationLocations(NSGAMMResult[i][0]),"NSGAIIMM","","1",1))
                    Delta.append((NSGAMMResult[i][1][:,[0,1]],"NSGAIIMM","-","",1))
                    GD.append((NSGAMMResult[i][1][:,[0,2]],"NSGAIIMM","-","",1))
                    IGD.append((NSGAMMResult[i][1][:,[0,3]],"NSGAIIMM","-","",1))

                    locations.append((pu.GetPopulationLocations(SPEAMMResult[i][0]),"SPEAIIMM","","P",1))
                    Delta.append((SPEAMMResult[i][1][:,[0,1]],"SPEAIIMM","-","",1))
                    GD.append((SPEAMMResult[i][1][:,[0,2]],"SPEAIIMM","-","",1))
                    IGD.append((SPEAMMResult[i][1][:,[0,3]],"SPEAIIMM","-","",1))

                    locations.append((pu.GetPopulationLocations(NSPSOResult[i][0]),"NSPSO","","x",1))
                    Delta.append((NSPSOResult[i][1][:,[0,1]],"NSPSO","-","",1))
                    GD.append((NSPSOResult[i][1][:,[0,2]],"NSPSO","-","",1))
                    IGD.append((NSPSOResult[i][1][:,[0,3]],"NSPSO","-","",1))

                    locations.append((pu.GetPopulationLocations(NSPSOMMResult[i][0]),"NSPSOMM","",".",1))
                    Delta.append((NSPSOMMResult[i][1][:,[0,1]],"NSPSOMM","-","",1))
                    GD.append((NSPSOMMResult[i][1][:,[0,2]],"NSPSOMM","-","",1))
                    IGD.append((NSPSOMMResult[i][1][:,[0,3]],"NSPSOMM","-","",1))

                    with open("NSPSO-NSGA-UFTest.txt", "a") as text_file:
                        for i in range(len(Delta)):
                            print("Metric {0}\t{1}\t{2}\t{3}".format(Delta[i][1],Delta[i][0][-1,1],GD[i][0][-1,1],IGD[i][0][-1,1]), file =text_file )
            
                    Resultplot(plotTitle + ' - Delta' , False, Delta)
                    Resultplot(plotTitle + ' - GD' , False, GD)
                    Resultplot(plotTitle + ' - IGD' , False, IGD)
                    
                    Resultplot(plotTitle, False, locations)
                        


def Resultplot(title, showPlot, dataList):
    fig = plt.figure()
    plt.title(title)

    for dataV in dataList:
        plt.plot(
            dataV[0][:, 0],
            dataV[0][:, 1],
            linestyle=dataV[2],
            marker=dataV[3],
            label=dataV[1],
            alpha=dataV[4])
    plt.legend()
    plt.grid()
    if showPlot:
        plt.show()
    else:
        plt.savefig(title + ".pdf")



if __name__ == "__main__":
    main()
"""