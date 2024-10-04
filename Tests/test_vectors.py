from Utils.Vectors import *
import pytest 
import random as rand

def genRand(lowRange = -100, highRange = 100):
    return rand.randint(lowRange, highRange)

def generateAddVectors(numTests):
    return [(jax.asarray([x1, y1, z1]), jax.asarray([x2, y2, z2]), jax.asarray([x1 + x2, y1 + y2, z1 + z2])) for _ in range(numTests) for x1, y1, z1, x2, y2, z2 in [genRand() for _ in range(6)]]

def testFunction():
    print(vector3D(1, 2, 3))
    assert False == False

print(generateAddVectors(2))
testFunction()