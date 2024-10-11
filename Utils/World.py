from Objects import * 
import copy

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
    def hitObjects(self, tInterval, ray, rayHitRecord):
        '''
        Iterate through the hittable objects list and check the smallest t that it intersects with to get the closest possible object
        '''
        hitAnything = False
        for i in ti.static(range(len(self.hittable))):
            objectHit, tempHitRecord = self.hittable[i].hit(tInterval, ray, initDefaultHitRecord(tInterval.max))
            if objectHit and tempHitRecord.t < rayHitRecord.t: 
                hitAnything = objectHit 
                rayHitRecord = copyHitRecord(tempHitRecord)

        return hitAnything, rayHitRecord
    
