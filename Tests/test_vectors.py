from Utils.Vectors import *
import pytest 

@pytest.mark.parametrize('bool1, bool2, answer', [
    (True, True, True),
    (True, False, False),
    (False, True, False),
    (False, False, True)
])
def testPytest(bool1, bool2, answer):
    assert (bool1 == bool2) == answer 

def generateProdTests(x, y, func):
    return [
    (vec3(*x), vec3(*y), func(x, y)),
    (-vec3(*x), vec3(*y), func([-v for v in x], y)),
    (vec3(*x), -vec3(*y), func(x, [-v for v in y])),
    (-vec3(*x), -vec3(*y), func([-v for v in x], [-v for v in y])),
    (0 * vec3(*x), vec3(*y), func([0, 0, 0], y)),
    (vec3(*x), 0 * vec3(*y), func(x, [0, 0, 0]))
    ]

def listDotProduct(l1: list, l2: list) -> float: 
    if len(l1) != len(l2):
        raise Exception(f'The length of the vectors should be equal. L1: {len(l1)}. L2: {len(l2)}')
    
    dotProd = 0
    for value1, value2 in zip(l1, l2):
        dotProd += value1 * value2 
    return dotProd

x, y = [2, 0, 3], [-2, 5, 1]

@pytest.mark.parametrize('v1, v2, dotProd', generateProdTests(x, y, listDotProduct))
def testDotProduct(v1, v2, dotProd):
    assert dotProduct(v1, v2) == dotProd

def listCrossProduct(l1: list, l2: list) -> vec3: #type: ignore 
    if len(l1) != len(l2):
        raise Exception(f'The length of the vectors should be equal. L1: {len(l1)}. L2: {len(l2)}')
    
    crossProd = []
    for i, j in zip((1, 2, 0), (2, 0, 1)):
        crossProd.append(l1[i] * l2[j] - l1[j] * l2[i])
    return vec3(*crossProd) 

@pytest.mark.parametrize('v1, v2, crossProd', generateProdTests(x, y, listCrossProduct))
def testCrossProduct(v1, v2, crossProd):
    assert crossProduct(v1, v2) == crossProd