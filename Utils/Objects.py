from Vectors import * 

@ti.func 
def calculateDiscriminant(a, b, c):
    '''
    Returns the discriminant of the quadratic given by coefficients a, b, and c
    '''
    return b ** 2 - 4 * a * c      

@ti.func 
def quadFormula(a, b, c):
    '''
    Perform the quadratic formula on the quadratic given by coefficients a, b, and c
    '''
    return (-b - calculateDiscriminant(a, b, c) ** 0.5) / (2 * a)

@ti.func 
def simplifiedDiscriminant(a, c, h):
    '''
    Returns the discriminant given a simplified sphere equation 
    '''
    return h ** 2 - a * c 

@ti.func 
def simplifiedQuadFormula(a, h, discriminant):
    '''
    Returns the quadratic formula given a simplified sphere equation 
    '''
    return (h - discriminant ** 0.5) / a 

@ti.dataclass 
class sphere3: 
    center: vec3 #type:ignore 
    radius: float 

@ti.func
def intersectSphere(rayStart, rayDirection, sphereStart, sphereRadius): 
    '''
    Check whether a ray intersects with a sphere and return t = -1.0 if it doesn't
    '''
    rayToSphereCenter = sphereStart - rayStart 
    a, h, c = tm.dot(rayDirection, rayDirection), tm.dot(rayDirection, rayToSphereCenter), tm.dot(rayToSphereCenter, rayToSphereCenter) - sphereRadius ** 2
    discriminant = simplifiedDiscriminant(a, c, h)
    
    t = -1.0
    if discriminant >= 0:
        t = simplifiedQuadFormula(a, h, discriminant)
    return t
