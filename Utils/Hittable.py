from Vectors import * 
from Interval import *

@ti.func
def initDefaultHitRecord(tMax):
    '''
    Initializes the default state of a hit record with maximal ray distance
    '''
    return hitRecord(vec3(0, 0, 0), vec3(0, 0, 0), tMax, True)

@ti.func 
def copyHitRecord(record):
    '''
    Copies over the values of a hitRecord
    '''
    return hitRecord(record.pointHit, record.normalVector, record.t, record.frontFace)

@ti.dataclass 
class hitRecord: 
    pointHit: vec3 #type: ignore 
    normalVector: vec3 #type: ignore
    t: float #type: ignore 
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
    
@ti.kernel 
def test() -> hitRecord: #type: ignore 
    return initDefaultHitRecord(1e10)

@ti.kernel 
def copyTest(record: hitRecord) -> hitRecord: #type: ignore 
    return copyHitRecord(record)