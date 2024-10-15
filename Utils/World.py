from Objects import * 
from BVH import *

@ti.data_oriented 
class World: 
    '''
    Sets the world scene for all hittable objects
    '''
    def __init__(self):
        self.hittable, self.BVHTree = [], []
    
    @ti.kernel 
    def addHittable(self, hittableObject: ti.template()): #type: ignore
        '''
        Add a hittable object and its classification 
        '''
        self.hittable.append(hittableObject)
    
    @ti.kernel 
    def constructBVHTree(self):
        self.BVHTree.append(BVHNode(self.hittable))
    
    def returnBVHTree(self):
        return self.BVHTree[0]

    @ti.func
    def hitObjects(self, ray, rayHitRecord):
        '''
        Iterate through the hittable objects list and check the smallest t that it intersects with to get the closest possible object
        '''
        hitAnything, BVHRoot, hitObject = False, self.BVHTree[0], False #Set defaults for these variables
        while not hitObject:
            hitAnything, BVHRoot, isObject = BVHRoot.hit(ray, rayHitRecord)
        for i in ti.static(range(len(self.hittable))):
            objectHit, tempHitRecord = self.hittable[i].hit(ray, initDefaultHitRecord(rayHitRecord.tInterval))
            if objectHit:
                hitAnything = objectHit
                rayHitRecord = copyHitRecord(tempHitRecord)

        return hitAnything, rayHitRecord
    
