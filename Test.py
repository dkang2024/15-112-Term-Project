import taichi as ti 
import taichi.math as tm 
import time

ti.init(ti.gpu)

vec3 = tm.vec3 

@ti.kernel 
def subtract(vec1: vec3, vec2: vec3) -> vec3: #type: ignore 
    return vec2 - vec1

start = time.time()
subtract(vec3(2, 1, 0), vec3(5, 4, 2))
print(vec3(1, 2, 3))
end = time.time()
print(end - start)
