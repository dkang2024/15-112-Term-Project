from Vectors import *

@ti.func 
def selectionSort(leaves: ti.template()): #type: ignore
    numberLeaves = leaves.shape[0]
    for i in range(numberLeaves):
        minIndex = i
        for j in range(i + 1, numberLeaves):
            if leaves[j].mortonCode < leaves[minIndex].mortonCode:
                minIndex = j
        leaves[i], leaves[minIndex] = leaves[minIndex], leaves[i]