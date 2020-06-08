import matplotlib.pyplot as plt
import numpy as np

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
        plt.savefig("results/"+title + ".pdf")

def saveAsText(title,indicatorName, dataList):
    for dataV in dataList:
        fname = "results/"+title  + " " + indicatorName + " " + dataV[1] + ".csv"        
        np.savetxt(fname,dataV[0],delimiter=',',fmt='%s')