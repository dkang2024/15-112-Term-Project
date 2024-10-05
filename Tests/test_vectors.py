from Utils.Vectors import *
import pytest 
import random as rand

def genRand(lowRange = -100, highRange = 100):
    return rand.randint(lowRange, highRange)

def generateTestArray(numTests):
    testArray = []
    for _ in range(numTests):
        posTuple1, posTuple2 = [genRand() for _ in range(3)], [genRand() for _ in range(3)]
        addTuple = [pos1 + pos2 for pos1, pos2 in zip(posTuple1, posTuple2)]
        testArray.append(jnp.array(posTuple1), jnp.array(posTuple2), jnp.array(addTuple))
    return testArray 

def testFunction():
    assert vector3D(1, 2, 3) == vector3D(1, 2, 3)
