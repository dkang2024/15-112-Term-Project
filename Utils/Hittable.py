from Vectors import * 
from Interval import *

@ti.func
def initDefaultHitRecord(tInterval):
    '''
    Initializes the default state of a hit record with maximal ray distance
    '''
    return hitRecord(vec3(0, 0, 0), vec3(0, 0, 0), tInterval, True)

@ti.func 
def copyHitRecord(record):
    '''
    Copies over the values of a hitRecord
    '''
    return hitRecord(record.pointHit, record.normalVector, record.tInterval, record.frontFace)

@ti.dataclass 
class hitRecord: 
    pointHit: vec3 #type: ignore 
    normalVector: vec3 #type: ignore
    tInterval: interval #type: ignore
    frontFace: bool #type: ignore

    @ti.func
    def isFrontFace(self, ray):
        '''
        Check if the ray is inside or outside the object using the dot product because my implementation for the normal vector always points outwards from the object
        '''
        isOutsideObject = True
        if tm.dot(ray.direction, self.normalVector) > 0:
            isOutsideObject = False
        return isOutsideObject

    @ti.func 
    def t(self):
        '''
        Get the max t value of the interval (which would be the intersection of the ray)
        '''
        return self.tInterval.maxValue