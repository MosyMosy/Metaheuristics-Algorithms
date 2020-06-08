import numpy as np
import Solution as sl
import random
import sys
import ParetoUtil as pu


"""Algorithm 1 The Framework of the Proposed Mating Restriction Strategy"""
def matting(population, Size, problem, mutationRate, crossoverRate):
    generationSize = 2 * (int(Size/2))
    # Non-Dominate Sorting
    F =  pu.fastNonDominatedSord(population)

    pairList = []
    
    # /*The manifold distance based mating selection */
    for Fi in F:  
        if np.size(Fi) >1:   
            pairList += MattingSelection(Fi,(generationSize/2 - len(pairList)))
    
    # /*Copy the pairs until the number of mating pairs is equal to N/2*/
    while len(pairList)<generationSize/2:
        for Fi in F: 
            #/*Copy the pairs until the number of mating pairs is equal to N/2*/  
            if len(pairList)<generationSize/2:         
                sec = SecondarySelection(pairList,Fi,generationSize)           
                pairList +=  sec
    
    # reproduct by selected parent pairs
    return Reproduct(pairList,generationSize,problem, mutationRate, crossoverRate)
    

"""Algorithm 2 M ating selection(P Fi, N)"""
def MattingSelection(Fi,remainingSpace):
    # current Front
    Pc = np.copy(Fi)
    # Calculate the manifold distances of Pc by (5) as an |Pc|*|Pc| array
    MD = CalculateManifoldDistance(np.ndarray.tolist(Pc))
    #/*Calculate the range of the neighborhood*/
    R = np.max(MD)/(np.size(Pc) - 1)

    pairList = []
    while (len(pairList) < remainingSpace):        
        # Randomly index a solution in Pc
        s = random.randint(0,(np.size(Pc)-1))
        while (Pc[s] == None):
            s = random.randint(0,(np.size(Pc)-1))
        # Pair1j+1 ← s
        Pairnew = (Pc[s],None)
        
        # /*The neighborhood of s*/ Pn ← {q | q ∈ Pc ∧ MDs,q < R}
        Pn_indexes = [i for i in range(len(MD[s])) if  (MD[s][i] < R) and MD[s][i] > 0]
        Pn_indexes.sort(key=lambda x: MD[s][x], reverse=False)
        if  len(Pn_indexes)>0:
            # /*Select the solution that has minimal manifold distance with the solution s in its neighborhood*/
            Pairnew = (Pairnew[0], Pc[Pn_indexes[0]])
            #Pc ← Pc \ Pn
            for i in range(len(Pn_indexes)):
                Pc[i] = None
                MD[i,:] = sys.float_info.max
                MD[:,i] = sys.float_info.max
        # remove index s from Pc
        Pc[s] = None     
        MD[s,:] = sys.float_info.max
        MD[:,s] = sys.float_info.max  
        
        pairList.append(Pairnew)
        # is pc empty
        if np.size(np.where(Pc == None))== np.size(Pc):
            break  
    return pairList

def MattingSelection2(Fi):
    # current Front
    Pc = np.copy(Fi)
    
    #/*Calculate the range of the neighborhood*/
    pairCount = int(Pc.shape[0]/2) 

    pairList = []
    selectedIndexes = []
    if Pc.shape[0]%2 == 1:
        Pairnew = (Pc[2* pairCount],None)
        pairList.append(Pairnew)
        Pc = Pc[:2* pairCount]
    # Calculate the manifold distances of Pc by (5) as an |Pc|*|Pc| array
    MD = CalculateManifoldDistance(np.ndarray.tolist(Pc))
    MDSortedIndexes = np.argsort(MD,1)

    while (len(pairList) <= pairCount ):        
        # Randomly index a solution in Pc
        indexes =np.arange(0,(Pc.shape[0]))
        if len(selectedIndexes) > 0:
            indexes =indexes[np.isin(indexes,np.array(selectedIndexes),invert=True)]
       
        if indexes.shape[0] == 0:
            break           
        s = np.random.choice(indexes)
        selectedIndexes.append(s)
        # Pair1j+1 ← s
        tergetIndex =MDSortedIndexes[s][np.isin(MDSortedIndexes[s] , np.array(selectedIndexes),invert=True)]
        Pairnew = (Pc[s],Pc[tergetIndex[0]])
        pairList.append(Pairnew)
        
        selectedIndexes.append(tergetIndex[0])
        
    return pairList



"""Algorithm 3 Secondary Selection(P air, P Fi, N)"""
def SecondarySelection(pairList,Fi,generationSize):
    
    if len(pairList)<generationSize/2:
        newPairs = []
        Nu = int(generationSize/2-len(pairList))
        #Find all the pairs with one member in front i
        onePairs = [pairList[i] for i in range(len(pairList)) if  pairList[i][1]  == None]
        for op in onePairs:
            if  op[0] in Fi:
                newPairs.append(op)
        
        if len(newPairs) == 0:
            #Find all the pairs in front i /*The pairs with one member and the pairs with two members*/
            for op in onePairs:
                if  (op[0] in Fi) and (op[1] in Fi):
                    newPairs.append(op)
        
        if len(newPairs)> Nu:
            # Randomly select nu pairs in P air′
            randomIndexes = np.random.choice(range(len(newPairs)), Nu, replace=False)
            newPairs =[newPairs[i] for i in randomIndexes]  #[newPairs[i] for i in randomIndexes]
        
        return newPairs

# Algorithm 4 Reproduction(P air, N)
def Reproduct(pairList,generationSize,problem, mutationRate, crossoverRate):
    newPop = np.empty(generationSize,dtype=sl.solution)
    """Using polynomial mutation in which the probability
    of mutation is fixed to 1 to generate a new parent for
    each single parent"""
    for i in range(int(generationSize/2)):
        if pairList[i][1] == None:
            pairList[i] = (pairList[i][0], pairList[i][0].PolynomialMutation(1))
    
    for i in range(0,int(generationSize/2)):
        # /*Each pair generates two offsprings through simulated binary crossover*/
        newPop[2*i],newPop[2*i + 1] = pairList[i][0].SBX(pairList[i][1])
    
    for i in range(generationSize):
        newPop[i] = newPop[i].PolynomialMutation(0.5)
    
    return newPop

    



def CalculateManifoldDistance(Fc):
    Fc.sort(key=lambda x: x.GetObjective(0), reverse=False)
    D = [Fc[i].EuclideanDistInSearchSpace(Fc[i + 1]) for i in range(len(Fc)-1)]
    
    MD = np.full([len(Fc),len(Fc)],-1,dtype=float)
    for i in range(len(Fc)):
        for j in range(i,len(Fc)):
            cumulativeDist = 0
            for c in range(i,j):
                cumulativeDist += D[c] 
            MD[i,j] = cumulativeDist
            MD[j,i] = MD[i,j]

    return MD



    