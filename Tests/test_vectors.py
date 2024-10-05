from Utils.Vectors import *
import pytest 
import random as rand

def genRand(lowRange = -100, highRange = 100):
    return rand.randint(lowRange, highRange)

def testFunction():
    assert vector3D(1, 2, 3) == vector3D(1, 2, 3)
    assert False == False

testFunction()