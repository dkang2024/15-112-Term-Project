from Rays import *
from Interval import *

@ti.func
def initDefaultHitRecord(tInterval):
    '''
    Initializes the default state of a hit record with maximal ray distance
    '''
    return hitRecord(defaultVec(), defaultVec(), True, defaultVec(), defaultRay(), defaultVec(), tInterval, True)

@ti.func 
def copyHitRecord(record):
    '''
    Copies over the values of a hitRecord
    '''
    return hitRecord(record.pointHit, record.initRayDir, record.didRayScatter, record.rayColor, record.rayScatter, record.normalVector, record.tInterval, record.frontFace)

@ti.dataclass 
class hitRecord: 
    '''
    Records important values with ray hits for recording and storing data efficiently (without having a ton of parameters and return values)
    '''

    pointHit: vec3 #type: ignore 
    initRayDir: vec3 #type: ignore
    didRayScatter: bool #type: ignore
    rayColor: vec3 #type: ignore
    rayScatter: ray3 #type: ignore
    normalVector: vec3 #type: ignore
    tInterval: interval #type: ignore
    frontFace: bool #type: ignore

    @ti.func
    def isFrontFace(self, ray):
        '''
        Check if the ray is front face or not using the dot product because my implementation for the normal vector always points against the ray. ALso resets the normal vector to point against the ray if the normal vector points with the ray
        '''
        frontFace = tm.dot(ray.direction, self.normalVector) < 0
        if not frontFace:
            self.normalVector = -self.normalVector
        return frontFace

    @ti.func 
    def t(self):
        '''
        Get the max t value of the interval (which would be the intersection of the ray)
        '''
        return self.tInterval.maxValue