from Objects import * 
from Morton import *
import copy

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
        for i in ti.static(range(len(self.hittableList))):
            tempHitRecord = self.hittableList[i].hit(ray, initDefaultHitRecord(rayHitRecord.tInterval))
            if tempHitRecord.hitAnything:
                rayHitRecord = copyHitRecord(tempHitRecord)

        return rayHitRecord

@ti.dataclass 
class BVHLeaf:
    '''
    This is the leaf node for the BVH Tree 
    '''
    objectIndex: int 
    boundingBox: aabb 
    mortonCode: int 

@ti.data_oriented 
class BVHTree:
    def __init__(self, hittableList, centroidScale):
        self.numLeaves, self.hittableList = len(hittableList), hittableList
        self.minCentroidVec, self.maxCentroidVec = self.getMinCentroidVec(centroidScale), self.getMaxCentroidVec(centroidScale)
        self.centroidDivisor = self.createDivisor()

        self.leaves = ti.Struct.field({
            'Leaf': BVHLeaf
        }, shape = (self.numLeaves,))
        self.fillLeaves()

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

    @ti.kernel 
    def createDivisor(self) -> vec3: #type: ignore
        '''
        Make sure the divisor near divides by zero to ensure that the scaling bounds from [0, 1] so that Morton codes don't throw an error
        '''
        divisor = self.maxCentroidVec - self.minCentroidVec
        for i in ti.static(range(len(divisor))):
            if self.valueNearZero(divisor[i]):
                divisor[i] = 1
        return 1 / divisor
    
    @ti.func 
    def createMorton(self, boundingBox): #type: ignore
        '''
        Create and return a morton code for the BVH leaf
        '''
        return mortonEncode(self.scaleCentroid(boundingBox))

    @ti.func 
    def scaleCentroid(self, boundingBox) -> vec3: #type: ignore
        '''
        Scale the bounding box centroid vector to [0, 1] for determining morton codes
        '''
        return (boundingBox.centroid() - self.minCentroidVec) * self.centroidDivisor
    
    @ti.func 
    def initLeaf(self, i: int, hittable: ti.template()) -> BVHLeaf: #type: ignore 
        '''
        Create and return a BVH leaf
        '''
        return BVHLeaf(i, hittable.boundingBox, self.createMorton(hittable.boundingBox))
    
    @ti.kernel 
    def fillLeaves(self):
        '''
        Fill the structure containing the leaves
        '''
        for i in ti.static(range(len(self.hittableList))):
            self.leaves.Leaf[i] = self.initLeaf(i, self.hittableList[i])

newWorld = World()

materialLeft = dielectricMaterial(1.0 / 1.3)
materialCenter = lambertianMaterial(vec3(0.1, 0.2, 0.5))

newWorld.addHittable(sphere3(vec3(0, 0, -1), 0.5, materialCenter))
newWorld.addHittable(sphere3(vec3(-1, 0, -1), 0.5, materialLeft))

@ti.kernel 
def testing():
    newWorld.compileMinAndMaxCentroid()

testing()
newTree = BVHTree(newWorld.hittableList, newWorld.centroidScale)

testLeaf = newTree.leaves.Leaf[0].mortonCode
print(testLeaf)