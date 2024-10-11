import taichi as ti 
import taichi.math as tm 
import random as rand

ti.init(ti.gpu, offline_cache = True, random_seed = rand.randint(0, 10000))
vec3 = tm.vec3

@ti.func 
def randomRange(minValue, maxValue):
    '''
    Return a random value in the range [minValue, maxValue)
    '''
    return ti.random() * (maxValue - minValue) + minValue 

@ti.func 
def randVectorRange(minValue, maxValue):
    '''
    Return a vector with values in the range
    '''
    return vec3(*[randomRange(minValue, maxValue) for _ in range(3)])

@ti.func 
def getX(vector):
    return vector[0]

@ti.func 
def getY(vector):
    return vector[1]

@ti.func 
def getZ(vector):
    return vector[2]

@ti.func 
def getRandomValueWithR(rSquared):
    '''
    Get a random value for a dimension of the vector for the random vector on the unit sphere
    '''
    r = rSquared ** 0.5
    value = randomRange(-r, r)
    rSquared -= value ** 2
    return value, rSquared

@ti.func
def chooseX(rSquared):
    '''
    Choose between a negative or positive x
    '''
    x = rSquared ** 0.5
    if ti.random() >= 0.5: 
        x *= -1
    return x 

@ti.func 
def randomVectorOnUnitSphere(rSquared = 1.0):
    z, rSquared = getRandomValueWithR(rSquared)
    y, rSquared = getRandomValueWithR(rSquared)
    x = chooseX(rSquared)
    return vec3(x, y, z)

@ti.func 
def randomInNormalDirection(normalVector):
    unitRayDir = randomVectorOnUnitSphere()
    if tm.dot(unitRayDir, normalVector) < 0.0: 
        unitRayDir *= -1.0 
    return unitRayDir

@ti.kernel 
def magnitude(v: vec3) -> float: #type: ignore 
    return tm.dot(v, v) ** 0.5