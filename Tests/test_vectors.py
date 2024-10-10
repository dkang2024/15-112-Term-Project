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

@ti.kernel 
def createTestRandomVector() -> vec3: #type: ignore
    return randomVectorOnUnitSphere()

def testUnitCircle():
    for _ in range(250):
        assert abs(magnitude(createTestRandomVector()) - 1) < 1e2