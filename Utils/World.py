from Objects import * 

@ti.data_oriented 
class World: 
    '''
    Sets the world scene for all hittable objects
    '''
    def __init__(self):
        self.hittable = []

    def addHittable(self, hittableObject):
        '''
        Add a hittable object and its classification 
        '''
        self.hittable.append(hittableObject)

    @ti.func
    def hitObjects(self, tMin, tMax, ray):
        '''
        Iterate through the hittable objects list and check the smallest t that it intersects with to get the closest possible object
        '''
        hitAnything, t, normalVector, frontFace = False, tMax, vec3(0, 0, 0), False
        for i in ti.static(range(len(self.hittable))):
            objectHit, objectT, objectNormal, objectFrontFace = self.hittable[i].hit(tMin, tMax, ray)
            if objectHit and objectT < t: 
                hitAnything, t, normalVector, frontFace = objectHit, objectT, objectNormal, objectFrontFace

        return hitAnything, t, normalVector, frontFace
    
