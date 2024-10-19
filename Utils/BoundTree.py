from Objects import * 
from Morton import *
from Sort import *
import time 

import warnings
warnings.filterwarnings("ignore") #Taichi throws warnings because list methods are used (and Taichi doesn't handle these but Python does). We want to ignore these warnings (the classes are specifically designed to allow taichi to work)

@ti.data_oriented 
class BVHTree:
        
    @ti.func 
    def valueNearZero(self, x):
        '''
        Checks whether a value is near zero within a range epsilon (to deal with floating point inaccuracies)
        '''
        epsilon = 1e-5
        return x < epsilon

    @ti.func 
    def createDivisor(self): #type: ignore
        '''
        Make sure the divisor near divides by zero to ensure that the scaling bounds from [0, 1] so that Morton codes don't throw an error
        '''
        divisor = self.centroidScale[1] - self.centroidScale[0]
        for i in ti.static(range(len(divisor))):
            if self.valueNearZero(divisor[i]):
                divisor[i] = 1
        self.divisor[None] = 1 / divisor
    
    @ti.func 
    def createMorton(self, hittable): #type: ignore
        '''
        Create and return a morton code for the BVH leaf
        '''
        return mortonEncode(self.scaleCentroid(hittable))

    @ti.func 
    def scaleCentroid(self, hittable) -> vec3: #type: ignore
        '''
        Scale the bounding box centroid vector to [0, 1] for determining morton codes
        '''
        return (hittable.center - self.centroidScale[0]) * self.divisor[None]
    
    @ti.func 
    def fillLeaves(self): #type: ignore
        '''
        Fill the structure containing the leaves
        '''
        self.createDivisor()
        for i in ti.static(ti.ndrange(self.numLeaves)):
            self.mortonCodes[i] = self.createMorton(self.hittableList[i])

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
            splitIndex = ti.cast((firstIndex + lastIndex), int) >> 1 #If the code at the first sorted morton code is equal to the code at the last sorted morton code, just take the midpoint (bitwise shift right by 1 for efficiency sake [fancy? :)])
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

    @ti.func 
    def generateBoundingBox(self, leftIndex, rightIndex):
        '''
        Generate a bounding box given the hittable list start index and end index (needed for transversal of the BVH Tree)
        '''
        boundingBox = self.boundingBoxes[leftIndex].boundingBox.returnCopy()
        for i in ti.ndrange((leftIndex + 1, rightIndex + 1)):
            boundingBox.addBoundingBox(self.boundingBoxes[i].boundingBox)
        return boundingBox

    @ti.func
    def sortLeaves(self):
        '''
        Sort the leaves in ascending order based on their Morton codes
        '''
        sort(self.mortonCodes)

    @ti.func  
    def generateNodes(self):
        '''
        Generate the nodes in the tree
        '''

        self.fillLeaves()
        self.sortLeaves()

        for i in ti.static(range(self.numLeaves - 1)):
            firstIndex, lastIndex = self.determineRange(i)
            split = self.findSplit(firstIndex, lastIndex)
            print(split, split + 1, firstIndex, lastIndex)
            leftSplit = split 
            if leftSplit == firstIndex:
                leftSplit += self.numLeaves #Add the number of leaves to make the split out of index of the Morton Codes to indicate that the child is a leaf 
            
            rightSplit = split + 1
            if rightSplit == lastIndex:
                rightSplit += self.numLeaves #Add the number of leaves to make the split out of index of the Morton Codes to indicate that the child is a leaf
            
            self.nodes[i].boundingBox = self.generateBoundingBox(firstIndex, lastIndex)
            self.nodes[i].leftChild = leftSplit 
            self.nodes[i].rightChild = rightSplit 
    
    @ti.func 
    def convertChildIndex(self, childIndex):
        '''
        Take care of the case that the child is a leaf 
        '''
        isLeaf = True 
        if childIndex >= self.numLeaves:
            isLeaf = False 
            childIndex -= self.numLeaves
        return isLeaf, childIndex

    @ti.func 
    def checkChild(self, isLeaf, childIndex, ray, rayHitRecord):
        '''
        Check if the child is hit and return the resulting hit record
        '''
        boundingBox = self.boundingBoxes.boundingBox[childIndex]
        if not isLeaf:
            boundingBox = self.nodes.boundingBox[childIndex]
        return boundingBox.hit(ray, initDefaultHitRecord(rayHitRecord.tInterval))
    
    @ti.func 
    def sortWithBothChildrenHit(self, leftIsLeaf, rightIsLeaf, leftNodeIndex, rightNodeIndex, rayHitRecord, leftChildHitRecord, rightChildHitRecord):
        '''
        Sort with both children being hit by the ray. Adding rayHitRecord as a parameter is required for Taichi to not throw an error because it has to be defined before the if statements 
        '''
        isLeaf, nodeIndex = False, leftNodeIndex
        if leftChildHitRecord.t() <= rightChildHitRecord.t():
            rayHitRecord = copyHitRecord(leftChildHitRecord)
            if leftIsLeaf: 
                isLeaf = True 
        else: 
            rayHitRecord, nodeIndex = copyHitRecord(rightChildHitRecord), rightNodeIndex
            if rightIsLeaf:
                isLeaf = True 
        return isLeaf, nodeIndex, rayHitRecord

    @ti.func 
    def walkTree(self, ray, rayHitRecord):
        '''
        Walk the tree to determine if the ray hit any bounding box
        '''
        nodeIndex, isLeaf = 0, False
        while True: 
            if nodeIndex == 0:
                boundingBox = self.nodes[nodeIndex].boundingBox 
                rayHitRecord = boundingBox.hit(ray, rayHitRecord)
            if rayHitRecord.hitAnything: 
                leftIsLeaf, leftChild = self.convertChildIndex(self.nodes[nodeIndex].leftChild)
                rightIsLeaf, rightChild = self.convertChildIndex(self.nodes[nodeIndex].rightChild)

                print(leftIsLeaf, rightIsLeaf, leftChild, rightChild)
                leftChildHitRecord = self.checkChild(leftIsLeaf, leftChild, ray, rayHitRecord)
                rightChildHitRecord = self.checkChild(rightIsLeaf, rightChild, ray, rayHitRecord)

                if leftChildHitRecord.hitAnything and rightChildHitRecord.hitAnything: #If ray hits both children, sort by which tInterval is lesser 
                    isLeaf, nodeIndex, rayHitRecord = self.sortWithBothChildrenHit(leftIsLeaf, rightIsLeaf, leftChild, rightChild, rayHitRecord, leftChildHitRecord, rightChildHitRecord)
                    print(isLeaf, nodeIndex, leftChild)
                elif leftChildHitRecord.hitAnything: 
                    isLeaf, nodeIndex, rayHitRecord = leftIsLeaf, leftChild, leftChildHitRecord 
                elif rightChildHitRecord.hitAnything:
                    isLeaf, nodeIndex, rayHitRecord = rightIsLeaf, rightChild, rightChildHitRecord 
                else: 
                    break 
                
                if isLeaf: 
                    break
                    
        return nodeIndex, rayHitRecord

@ti.data_oriented 
class World(BVHTree): 
    '''
    Sets the world scene for all hittable objects
    '''
    def __init__(self):
        self.hittableList = []
        self.divisor, self.centroidScale = ti.Vector.field(3, float, shape = ()), ti.Vector.field(3, float, shape = (2,))
        
    def addHittable(self, hittableObject): #type: ignore
        '''
        Add a hittable object and its classification 
        '''
        self.hittableList.append(hittableObject)

    def compileFields(self):
        '''
        Compile the fields that the BVH Tree needs to use based on the length of the hittable list (YOU MUST CALL THIS IN PYTHON SCOPE BEFORE ACTUALLY COMPILING THE TREE)
        '''
        self.numLeaves = len(self.hittableList)
        self.mortonCodes = ti.field(int, shape = (self.numLeaves,))
        self.nodes = ti.Struct.field({
            'boundingBox': aabb, 
            'leftChild': int, 
            'rightChild': int 
        }, shape = (self.numLeaves - 1)) #Keep track of nodes in the tree through keeping track of their child ranges and children split (note that if child split is greater than)
        self.boundingBoxes = ti.Struct.field({
            'boundingBox': aabb
        }, shape = (self.numLeaves,))
        self.fillBoundingBoxes()

    @ti.kernel 
    def fillBoundingBoxes(self):
        '''
        Fill the structure field with the bounding boxes for compiling 
        '''
        for i in ti.static(range(self.numLeaves)):
            self.boundingBoxes[i].boundingBox = self.hittableList[i].boundingBox

    @ti.func 
    def compileMinAndMaxCentroid(self): #type: ignore
        '''
        Return the minimum and maximum values for all the centers of all the objects in order to rescale the centers to [0, 1]
        '''
        centroids = [self.hittableList[i].center for i in ti.static(range(len(self.hittableList)))]
        self.centroidScale[0], self.centroidScale[1] = ti.min(*centroids), ti.max(*centroids)

    @ti.func 
    def compileTree(self):
        '''
        Compile the BVH Tree for the world
        '''
        self.compileMinAndMaxCentroid()
        self.generateNodes()
    
    @ti.func
    def hitObjects(self, ray, rayHitRecord):
        '''
        Iterate through the hittable objects list and check the smallest t that it intersects with to get the closest possible object
        '''
        for i in ti.static(range(self.numLeaves)):
            tempHitRecord = self.hittableList[i].hit(ray, initDefaultHitRecord(rayHitRecord.tInterval))
            if tempHitRecord.hitAnything:
                rayHitRecord = copyHitRecord(tempHitRecord)

        return rayHitRecord

@ti.kernel 
def testCompile():
    camera.compileTree()

start = time.perf_counter()
camera = World()
materialCenter = lambertianMaterial(vec3(0.1, 0.2, 0.5))
materialGround = lambertianMaterial(vec3(0.8, 0.8, 0.0))
camera.addHittable(sphere3(vec3(0, 0, -1), 0.5, materialCenter))
camera.addHittable(sphere3(vec3(0, -100.5, -1), 100, materialGround))
camera.compileFields()
testCompile()
end = time.perf_counter()
print(end - start)




materialLeft = dielectricMaterial(1.0 / 1.3)
materialRight = reflectiveMaterial(vec3(0.8, 0.6, 0.2), 0.5)
materialFront = reflectiveMaterial(vec3(0.8, 0.8, 0.8), 0.2)



camera.addHittable(sphere3(vec3(-1, 0, -1), 0.5, materialLeft))
camera.addHittable(sphere3(vec3(1, 0, -1), 0.5, materialRight))
camera.addHittable(sphere3(vec3(0, 0, 0), 0.5, materialFront))

start = time.perf_counter()
camera.compileFields()
end = time.perf_counter()
print(f'Init Starting: {end - start}')



start = time.perf_counter()
testCompile()
end = time.perf_counter()
print(f'In Kernel With Compile: {end - start}')

start = time.perf_counter()
testCompile()
end = time.perf_counter()
print(f'In Kernel Out of Compile: {end - start}')

camera.addHittable(sphere3(vec3(0, 0, 0), 0.5, materialFront))

start = time.perf_counter()
camera.compileFields()
end = time.perf_counter()
print(f'Init Starting 2: {end - start}')

start = time.perf_counter()
testCompile()
end = time.perf_counter()
print(f'In Kernel Out of Compile 2: {end - start}')


@ti.kernel 
def testTreeWalk():
    nodeIndex, nodeHitRecord = camera.walkTree(ray3(vec3(0, 0, -2), vec3(0, 0, -1)), initDefaultHitRecord(interval(0.001, 1e10)))
    print(nodeIndex)

start = time.perf_counter()
#testTreeWalk()
end = time.perf_counter()
print(f'Walk Tree: {end - start}')

start = time.perf_counter()
#testTreeWalk()
end = time.perf_counter()
print(f'Walk Tree 2: {end - start}')

