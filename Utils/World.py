from Objects import * 

@ti.data_oriented


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
    def hitObjects(self, rayStart, rayDir):
        t, center = 0.0, vec3(0, 0, 0)
        for i in ti.static(range(len(self.hittable))):
            hittableObject = self.hittable[i]
            t, center = hittableObject.hit(rayStart, rayDir), hittableObject.center
        return t, center
    
