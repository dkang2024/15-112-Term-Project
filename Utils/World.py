from Objects import * 

@ti.data_oriented 
class World: 
    '''
    Sets the world scene for all hittable objects
    '''
    def __init__(self):
        self.hittable = []
    
    @ti.kernel 
    def addHittable(self, hittableObject: ti.template()): #type: ignore
        '''
        Add a hittable object and its classification 
        '''
        self.hittable.append(hittableObject)
    
    @ti.func
    def hitObjects(self, ray, rayHitRecord):
        '''
        Iterate through the hittable objects list and check the smallest t that it intersects with to get the closest possible object
        '''
        for i in ti.static(range(len(self.hittable))):
            tempHitRecord = self.hittable[i].hit(ray, initDefaultHitRecord(rayHitRecord.tInterval))
            if tempHitRecord.hitAnything:
                rayHitRecord = copyHitRecord(tempHitRecord)

        return rayHitRecord
    
