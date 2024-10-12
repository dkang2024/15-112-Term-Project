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
    '''
    Create a random vector on the unit sphere for Lambertian reflfection
    '''
    z, rSquared = getRandomValueWithR(rSquared)
    y, rSquared = getRandomValueWithR(rSquared)
    x = chooseX(rSquared)
    return vec3(x, y, z)

@ti.kernel 
def magnitude(v: vec3) -> float: #type: ignore 
    return tm.dot(v, v) ** 0.5

@ti.func 
def nearZero(v):
    '''
    Checks whether all elements of the vector are near zero
    '''
    epsilon = 1e-5
    return getX(v) < epsilon and getY(v) < epsilon and getZ(v) < epsilon

@ti.func 
def reflect(v, n):
    '''
    Reflect a vector back according to the normal vector of a surface
    '''
    return v - 2 * tm.dot(v, n) * n

@ti.func 
def refract(v, n, etaRatio, cosTheta):
    '''
    Refract a vector through an object
    '''
    rPerp = etaRatio * (v + cosTheta * n)
    rParallel = -ti.sqrt(ti.abs(1.0 - tm.dot(rPerp, rPerp))) * n 
    return rPerp + rParallel