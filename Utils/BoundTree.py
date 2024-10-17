from BoundBox import * 
from Objects import * 
from Morton import *

import warnings
warnings.filterwarnings("ignore") #Taichi throws warnings because list methods are used (and Taichi doesn't handle these but Python does). We want to ignore these warnings (the classes are specifically designed to allow taichi to work)

@ti.data_oriented 
class World: 
    '''
    Sets the world scene for all hittable objects
    '''
    def __init__(self):
        self.hittableList, self.centroidScale = [], ti.Vector.field(3, float, shape = (2,))
    
    @ti.kernel 
    def addHittable(self, hittableObject: ti.template()): #type: ignore
        '''
        Add a hittable object and its classification 
        '''
        self.hittableList.append(hittableObject)

    @ti.func
    def compileMinAndMaxCentroid(self):
        centroids = [self.hittableList[i].boundingBox.centroid() for i in ti.static(range(len(self.hittableList)))]
        self.centroidScale[0], self.centroidScale[1] = ti.min(*centroids), ti.max(*centroids)
    
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

@ti.data_oriented 
class BVHTree:
    def __init__(self, hittableList, centroidScale):
        self.numLeaves = len(hittableList)
        self.minCentroidVec, self.maxCentroidVec = self.getMinCentroidVec(centroidScale), self.getMaxCentroidVec(centroidScale)
        self.centroidDivider = self.createDivisor()
        self.leaves = []

    @ti.kernel 
    def getMinCentroidVec(self, centroidScale: ti.template()) -> vec3: #type: ignore
        return centroidScale[0]
    
    @ti.kernel 
    def getMaxCentroidVec(self, centroidScale: ti.template()) -> vec3: #type: ignore 
        return centroidScale[1]

    @ti.func 
    def valueNearZero(self, x):
        '''
        Checks whether a value is near zero within a range epsilon (to deal with floating point inaccuracies)
        '''
        epsilon = 1e-5
        return x < epsilon

    @ti.func 
    def createDivisor(self):
        '''
        Make sure the divisor near divides by zero to ensure that the scaling bounds from [0, 1] so that Morton codes don't throw an error
        '''
        divisor = self.maxCentroidVec - self.minCentroidVec
        for i in ti.static(range(len(divisor))):
            if self.valueNearZero(divisor[i]):
                divisor[i] = 1
        return 1 / divisor

@ti.dataclass 
class BVHLeaf:
    def __init__(self, hittable, minCentroidVec, divisor):
        self.object, self.boundingBox = hittable, hittable.boundingBox 
        self.centroid = self.scaleCentroid(minCentroidVec, divisor)
        self.mortonCode = mortonEncode(self.centroid)    

    @ti.kernel 
    def scaleCentroid(self, minCentroidVec: vec3, divisor: vec3) -> vec3: #type: ignore
        '''
        Scale the bounding box centroid vector to [0, 1] for determining morton codes
        '''
        return (self.boundingBox.centroid() - minCentroidVec) * divisor
    

newWorld = World()

materialLeft = dielectricMaterial(1.0 / 1.3)
materialCenter = lambertianMaterial(vec3(0.1, 0.2, 0.5))

newWorld.addHittable(sphere3(vec3(0, 0, -1), 0.5, materialCenter))
newWorld.addHittable(sphere3(vec3(-1, 0, -1), 0.5, materialLeft))

@ti.kernel 
def testing():
    newWorld.compileMinAndMaxCentroid()

testing()
newNode = BVHLeaf(newWorld.hittableList[0], vec3(0, 0, 0), vec3(0, 1, 0))
print(newNode.object)
newTree = BVHTree(newWorld.hittableList)