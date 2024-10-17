from Objects import * 
from Morton import *
import time 
from taichi.algorithms import parallel_sort

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
    This is the leaf for the BVH Tree 
    '''
    objectIndex: int 
    boundingBox: aabb 

@ti.data_oriented 
class BVHTree:
    def __init__(self, hittableList, centroidScale):
        self.numLeaves, self.hittableList = len(hittableList), hittableList
        self.minCentroidVec, self.maxCentroidVec = self.getMinCentroidVec(centroidScale), self.getMaxCentroidVec(centroidScale)
        self.centroidDivisor = self.createDivisor()

        self.leaves = ti.Struct.field({
            'Leaf': BVHLeaf
        }, shape = (self.numLeaves,))
        self.mortonCodes = ti.field(ti.uint32, shape = (self.numLeaves,))
        self.fillLeaves()
        self.sortLeaves()

        self.nodes = ti.Vector.field(4, int, shape = (self.numLeaves - 1,)) #Keep track of nodes in the tree through keeping track of 
        self.generateNodes()

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
    def initLeaf(self, i: int, hittable: ti.template()): #type: ignore 
        '''
        Create and return a BVH leaf
        '''
        return BVHLeaf(i, hittable.boundingBox), self.createMorton(hittable.boundingBox)
    
    @ti.kernel 
    def fillLeaves(self):
        '''
        Fill the structure containing the leaves
        '''
        for i in ti.static(range(len(self.hittableList))):
            self.leaves.Leaf[i], self.mortonCodes[i] = self.initLeaf(i, self.hittableList[i])

    def sortLeaves(self):
        '''
        Sort the leaves according to morton code
        '''
        parallel_sort(self.mortonCodes, self.leaves) 

    @ti.func 
    def countLeadingZeros(self, num):
        '''
        Count the number of leading zeros in an unsigned binary representation of a number. I should get no credit for this (I was really clueless on how to do this until I found: https://github.com/archibate/ptina/blob/master/ptina/tree/lbvh.py [it has really really really awful naming conventions though good luck reading through it])
        '''
        count = 0 
        while True: 
            leftBit = num >> (31 - count) #Assumption being that we're using a 32 bit unsigned integer, and we're finding the leftmost bit by shifting it over and incrementing count. Note that the leftover values from the left of the bitwise shift left don't matter because the algorithm checks if they're 0 anyway (what a genius algorithm that person on Github came up with)
            if leftBit == 1 or count == 31:
                count += 1
                break 
            count += 1
        return count 

    @ti.func 
    def findSplit(self, firstIndex, lastIndex):
        '''
        Find the split for the LBVH. Thanks to https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/ (lifesaver). I translated the code over to Taichi Python
        '''
        firstCode, lastCode = self.mortonCodes[firstIndex], self.mortonCodes[lastIndex]
        splitIndex = 0 

        if firstCode == lastCode: 
            splitIndex = ti.cast((firstCode + lastCode), int) >> 1 #If the code at the first sorted morton code is equal to the code at the last sorted morton code, just take the midpoint (bitwise shift right by 1 for efficiency sake [fancy? :)])
        else: 
            commonPrefix = self.countLeadingZeros(firstCode ^ lastCode) #This is the number of bits that the first Morton code and the last Morton code share 

            # We now perform binary search to find where the next bit differs and we return the split index that splits this difference
            split = firstIndex #Start the split at the first possible index
            step = lastIndex - firstIndex #Init the step 

            while step > 1:
                step = (step + 1) >> 1 #This is the step for binary search
                newSplit = split + step 
                
                if newSplit < lastIndex: #Check whether the split is a valid split
                    splitCode = self.mortonCodes[newSplit]
                    splitPrefix = self.countLeadingZeros(firstCode ^ splitCode)
                    if splitPrefix > commonPrefix:
                        split = newSplit 
        
        return splitIndex
    
    @ti.func
    def determineRange(self, i):
        '''
        Determine the range for firstIndex and lastIndex. PLEASE NOTE that I should get absolutely no credit for this. I could not (for the life of me) understand the algorithm so I copy pasted https://github.com/archibate/ptina/blob/master/ptina/tree/lbvh.py.
        '''
        l, r = 0, self.numLeaves - 1

        if i != 0:
            ic = self.mortonCodes[i]
            lc = self.mortonCodes[i - 1]
            rc = self.mortonCodes[i + 1]

            if lc == ic == rc:
                l = i
                while i < self.numLeaves - 1:
                    i += 1
                    if i >= self.numLeaves - 1:
                        break
                    if self.mortonCodes[i] != self.mortonCodes[i + 1]:
                        break
                r = i

            else:
                ld = self.countLeadingZeros(ic ^ lc)
                rd = self.countLeadingZeros(ic ^ rc)

                d = -1
                if rd > ld:
                    d = 1
                delta_min = ti.min(ld, rd)
                lmax = 2
                delta = -1
                itmp = i + d * lmax
                if 0 <= itmp < self.numLeaves:
                    delta = self.countLeadingZeros(ic ^ self.mortonCodes[itmp])
                while delta > delta_min:
                    lmax <<= 1
                    itmp = i + d * lmax
                    delta = -1
                    if 0 <= itmp < self.numLeaves:
                        delta = self.countLeadingZeros(ic ^ self.mortonCodes[itmp])
                s = 0
                t = lmax >> 1
                while t > 0:
                    itmp = i + (s + t) * d
                    delta = -1
                    if 0 <= itmp < self.numLeaves:
                        delta = self.countLeadingZeros(ic ^ self.mortonCodes[itmp])
                    if delta > delta_min:
                        s += t
                    t >>= 1

                l, r = i, i + s * d
                if d < 0:
                    l, r = r, l

        return l, r

    @ti.kernel
    def generateNodes(self):
        '''
        Generate the nodes in the tree
        '''
        for i in ti.static(range(self.numLeaves - 1)):
            firstIndex, lastIndex = self.determineRange(i)

            split = self.findSplit(firstIndex, lastIndex)
            
            leftSplit = split 
            if leftSplit == firstIndex:
                leftSplit += self.numLeaves #Add the number of leaves to make the split out of index of the Morton Codes to indicate that the child is a leaf 
            
            rightSplit = split + 1
            if rightSplit == lastIndex:
                rightSplit += self.numLeaves #Add the number of leaves to make the split out of index of the Morton Codes to indicate that the child is a leaf
            
            self.nodes[i][0] = firstIndex 
            self.nodes[i][1] = lastIndex 
            self.nodes[i][2] = leftSplit 
            self.nodes[i][3] = rightSplit

    @ti.func 
    def generateBoundingBox(self, startIndex, endIndex):
        '''
        Generate a bounding box given the hittable list start index and end index (needed for transversal of the BVH Tree)
        '''
        boundingBox = self.hittableList[startIndex].boundingBox 
        for i in ti.static(range(startIndex + 1, endIndex)):
            boundingBox.addBoundingBox(self.hittableList[i].boundingBox)
        return boundingBox 
    
    @ti.func 
    def walkTree(self, ray, rayHitRecord):
        '''
        Walk the tree to determine if the ray hit any bounding box
        '''
        nodeIndex, isRoot = 0, True 
        while True: 
            pass 

camera = World()

materialGround = lambertianMaterial(vec3(0.8, 0.8, 0.0))
materialCenter = lambertianMaterial(vec3(0.1, 0.2, 0.5))
materialLeft = dielectricMaterial(1.0 / 1.3)
materialRight = reflectiveMaterial(vec3(0.8, 0.6, 0.2), 0.5)
materialFront = reflectiveMaterial(vec3(0.8, 0.8, 0.8), 0.2)

camera.addHittable(sphere3(vec3(0, 0, -1), 0.5, materialCenter))
camera.addHittable(sphere3(vec3(0, -100.5, -1), 100, materialGround))
camera.addHittable(sphere3(vec3(-1, 0, -1), 0.5, materialLeft))
camera.addHittable(sphere3(vec3(1, 0, -1), 0.5, materialRight))
camera.addHittable(sphere3(vec3(0, 0, 0), 0.5, materialFront))

@ti.kernel 
def testing():
    camera.compileMinAndMaxCentroid()

testing()

start = time.perf_counter()
newTree = BVHTree(camera.hittableList, camera.centroidScale)
end = time.perf_counter()

def printValuables():
    node = newTree.nodes[0]
    print(node[0], node[1], node[2], node[3])

printValuables()
print(newTree.mortonCodes, newTree.leaves.Leaf[1].objectIndex, newTree.nodes[0], end - start)