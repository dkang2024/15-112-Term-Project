from Vectors import * 

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
def checkOutBounds(tMin, tMax, t):
    '''
    Check if t is outside of tMin and tMax
    '''
    return t < tMin or t > tMax 

@ti.func 
def isSphereFrontFace(ray, normalVector):
    '''
    Check if the ray is inside or outside the sphere using the dot product because my implementation for the normal vector of a sphere always points outwards from the center of the sphere
    '''
    isOutsideSphere = True
    if tm.dot(ray.direction, normalVector) > 0:
        isOutsideSphere = False
    return isOutsideSphere    

@ti.func 
def checkSphereIntersection(a, h, discriminant, tMin, tMax):
    '''
    Check whether the ray-sphere intersection occurs within the range of tMin and tMax. Unfortunately cannot do this with a loop because Taichi disallows looping over anything else than its own values (so can't use a tuple to vary the sign and then use break).
    '''
    t = simplifiedQuadFormula(a, h, discriminant, -1.0)
    if checkOutBounds(tMin, tMax, t):
        t = simplifiedQuadFormula(a, h, discriminant, 1.0)
        if checkOutBounds(tMin, tMax, t):
            t = -1.0
    return t >= 0, t

@ti.data_oriented
class sphere3(): 
    '''
    Class for a sphere and its ray intersections
    '''
    def __init__(self, center, radius):
        self.center, self.radius = center, radius

    @ti.func
    def hit(self, tMin, tMax, ray): 
        '''
        Check whether a ray intersects with a sphere and return t = -1.0 if it doesn't
        '''
        rayToSphereCenter = self.center - ray.origin
        a, h, c = tm.dot(ray.direction, ray.direction), tm.dot(ray.direction, rayToSphereCenter), tm.dot(rayToSphereCenter, rayToSphereCenter) - self.radius ** 2
        discriminant = simplifiedDiscriminant(a, c, h)
        
        hitSphere, t, normalVector, frontFace = False, -1.0, vec3(0, 0, 0), False
        if discriminant >= 0:
            hitSphere, t = checkSphereIntersection(a, h, discriminant, tMin, tMax)
            if hitSphere:
                normalVector = findSphereNormalVector(ray, t, self.center)
                frontFace = isSphereFrontFace(ray, normalVector)

        return hitSphere, t, normalVector, frontFace
