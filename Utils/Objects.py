from Vectors import * 
from Materials import *
from Hittable import * 

@ti.func 
def simplifiedDiscriminant(a, c, h):
    '''
    Returns the discriminant given a simplified sphere equation 
    '''
    return h ** 2 - a * c 

@ti.func 
def simplifiedQuadFormula(a, h, discriminant, sign):
    '''
    Returns the quadratic formula given a simplified sphere equation 
    '''
    return (h + sign * discriminant ** 0.5) / a 

@ti.func 
def findSphereNormalVector(ray, t, center):
    '''
    Find the sphere's normal vector
    '''
    return tm.normalize(ray.pointOnRay(t) - center)

@ti.func 
def checkSphereIntersection(a, h, discriminant, tInterval):
    '''
    Check whether the ray-sphere intersection occurs within the range of tMin and tMax. Unfortunately cannot do this with a loop because Taichi disallows looping over anything else than its own values (so can't use a tuple to vary the sign and then use break).
    '''
    t = simplifiedQuadFormula(a, h, discriminant, -1.0)
    if not tInterval.surrounds(t):
        t = simplifiedQuadFormula(a, h, discriminant, 1.0)
        if not tInterval.surrounds(t):
            t = -1.0
    return t >= 0, t

@ti.data_oriented
class sphere3(): 
    '''
    Class for a sphere and its ray intersections
    '''
    def __init__(self, center, radius, material):
        self.center, self.radius, self.material = center, radius, material

    @ti.func
    def hit(self, ray, tempHitRecord): 
        '''
        Check whether a ray intersects with a sphere and return t = -1.0 if it doesn't
        '''
        rayToSphereCenter = self.center - ray.origin
        a, h, c = tm.dot(ray.direction, ray.direction), tm.dot(ray.direction, rayToSphereCenter), tm.dot(rayToSphereCenter, rayToSphereCenter) - self.radius ** 2
        discriminant = simplifiedDiscriminant(a, c, h)
        
        hitSphere = False 
        if discriminant >= 0:
            hitSphere, tempHitRecord.tInterval.maxValue = checkSphereIntersection(a, h, discriminant, tempHitRecord.tInterval)
            if hitSphere:
                tempHitRecord.pointHit = ray.pointOnRay(tempHitRecord.t())
                tempHitRecord.initRayDir = ray.direction
                tempHitRecord.normalVector = findSphereNormalVector(ray, tempHitRecord.t(), self.center)
                tempHitRecord.frontFace = tempHitRecord.isFrontFace(ray)
                tempHitRecord.rayScatter, tempHitRecord.rayColor = self.material.scatter(tempHitRecord)
        
        return hitSphere, tempHitRecord
