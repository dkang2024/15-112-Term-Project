import taichi as ti 
import taichi.math as tm 

ti.init(ti.gpu)
vec3 = tm.vec3

@ti.kernel 
def dotProduct(v1: vec3, v2: vec3) -> float: #type: ignore
    '''
    Allows calling tm.dot within Python's scope
    '''
    return tm.dot(v1, v2)

@ti.kernel 
def crossProduct(v1: vec3, v2: vec3) -> vec3: #type: ignore 
    '''
    Allows calling tm.cross within Python's scope 
    '''
    return tm.cross(v1, v2)

@ti.kernel 
def normalize(v: vec3) -> vec3: #type: ignore 
    '''
    Allows calling tm.normalize within Python's scope
    '''
    return tm.normalize(v)

@ti.kernel 
def magnitude(v: vec3) -> float: #type: ignore
    '''
    Returns the magnitude of a vector
    '''
    return tm.dot(v, v) ** 0.5