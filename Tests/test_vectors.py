from Utils.Vectors import *
import pytest 

x, y, z = 1, 3, 4
posVector, negVector = vector3D(x, y, z), vector3D(-x, -y, -z)

@pytest.mark.parametrize('vector1, vector2, isEqual', [
    (posVector, posVector, True),
    (posVector, -negVector, True),
    (posVector, -posVector, False),
    (posVector.scalarMultiply(0), negVector.scalarMultiply(0), True)   
])
def testEquality(vector1, vector2, isEqual):
    assert (vector1 == vector2) == isEqual
    
