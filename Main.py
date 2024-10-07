from Utils import *

v1 = vec3(2, 3, 4)
v2 = ray3(v1, v1)

@ti.kernel 
def determinePointOnRay(t: float) -> vec3: #type: ignore 
    return v2.pointOnRay(t)

print(determinePointOnRay(1))