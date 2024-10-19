from Vectors import *

@ti.func 
def selectionSort(leaves):
    numLeaves = leaves.shape[0]
    for i in leaves:
        minIndex = i 
        for j in ti.ndrange((i + 1, numLeaves)):
            if leaves.mortonCode[j] < leaves.mortonCode[minIndex]:
                minIndex = j 
        objectIndex, mortonCode, boundingBox = leaves[i].objectIndex, leaves[i].mortonCode, leaves[i].boundingBox
        leaves[i] = leaves[minIndex]
        leaves[minIndex].objectIndex = objectIndex
        leaves[minIndex].mortonCode = mortonCode 
        leaves[minIndex].boundingBox = boundingBox
