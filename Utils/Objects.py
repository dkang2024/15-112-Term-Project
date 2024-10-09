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

@ti.dataclass 
class sphere3: 
    center: vec3 #type:ignore 
    radius: float 

@ti.func
def intersectSphere(rayStart, rayDirection, sphereStart, sphereRadius): 
    rayToSphereCenter = sphereStart - rayStart 
    a, b, c = tm.dot(rayDirection, rayDirection), tm.dot(-2 * rayDirection, rayToSphereCenter), tm.dot(rayToSphereCenter, rayToSphereCenter) - sphereRadius ** 2
    return calculateDiscriminant(a, b, c) >= 0
