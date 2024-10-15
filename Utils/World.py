from Objects import * 

@ti.data_oriented 
class World: 
    '''
    Sets the world scene for all hittable objects
    '''
    def __init__(self):
        self.hittable = []
        self.boundingBox = ti.Struct.field({
            'x': interval, 
            'y': interval, 
            'z': interval
        }, shape = ())
    
    @ti.kernel 
    def addHittable(self, hittableObject: ti.template()): #type: ignore
        '''
        Add a hittable object and its classification 
        '''
        self.hittable.append(hittableObject)

    @ti.kernel 
    def createBoundingBox(self): 
        boundingBox = self.hittable[0].boundingBox 
        for i in ti.static(range(1, len(self.hittable))):
            boundingBox.addBoundingBox(self.hittable[i].boundingBox)
        self.boundingBox[None] = boundingBox

    def returnBoundingBox(self):
        return self.boundingBox[None]

    @ti.func
    def hitObjects(self, ray, rayHitRecord):
        '''
        Iterate through the hittable objects list and check the smallest t that it intersects with to get the closest possible object
        '''
        hitAnything = False #Set defaults for these variables
        for i in ti.static(range(len(self.hittable))):
            objectHit, tempHitRecord = self.hittable[i].hit(ray, initDefaultHitRecord(rayHitRecord.tInterval))
            if objectHit:
                hitAnything = objectHit
                rayHitRecord = copyHitRecord(tempHitRecord)

        return hitAnything, rayHitRecord
    
