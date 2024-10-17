from BoundBox import * 
import copy 
from Objects import * 
import time

import warnings
warnings.filterwarnings("ignore") #Taichi throws warnings because classes are used in ti.kernel. We want to ignore these warnings (the classes are specifically designed to allow taichi to work)

@ti.kernel 
def returnIntervalMin(hittable: ti.template(), axisIndex: int) -> float: #type: ignore 
    '''
    Get the bounding box interval's minimum value for comparison in the BVH Tree
    '''
    intervalMin, boundingBox = 0.0, hittable.boundingBox
    if axisIndex == 0:
        intervalMin = boundingBox.x.minValue
    elif axisIndex == 1:
        intervalMin = boundingBox.y.minValue 
    else: 
        intervalMin = boundingBox.z.minValue 
    return intervalMin

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
        hitAnything = False 
        for i in ti.static(range(len(self.hittable))):
            objectHit, tempHitRecord = self.hittable[i].hit(ray, initDefaultHitRecord(rayHitRecord.tInterval))
            if objectHit:
                hitAnything = objectHit
                rayHitRecord = copyHitRecord(tempHitRecord)

        return hitAnything, rayHitRecord

@ti.data_oriented
class BVHNode:
    '''
    Node on the BVH Tree
    '''
    def __init__(self, treeLength, childIndex, hittableList):
        self.hittableList = copy.copy(hittableList)

        self.isLeaf = False 
        if len(self.hittableList) == 1:
            self.isLeaf = True 

        self.nodeIndex = treeLength
        self.leftChild, self.rightChild = treeLength + childIndex + 1, treeLength + childIndex + 2
        if self.isLeaf: 
            self.leftChild, self.rightChild = -1, -1

        self.boundingBox = self.initBoundingBox()

    @ti.kernel 
    def initBoundingBox(self) -> aabb: 
        '''
        Create the bounding box for this BVH Node
        '''
        boundingBox = self.hittableList[0].boundingBox
        for i in ti.static(range(1, len(self.hittableList))):
            boundingBox.addBoundingBox(self.hittableList[i].boundingBox)
        return boundingBox

    @ti.kernel 
    def hit(self, ray: ti.template(), rayHitRecord: ti.template()) -> hitRecord: #type: ignore
        '''
        Check whether the ray hits this bounding box
        '''
        return self.boundingBox.hit(ray, initDefaultHitRecord(rayHitRecord.tInterval))

@ti.data_oriented 
class BVHTree:
    '''
    Create the BVH Tree and allow easy access to go througb it
    '''
    def __init__(self, hittableList):
        self.nodes, self.hittableList = [BVHNode(1, 0, hittableList)], hittableList

        self.hittableLeft, self.hittableRight = [], []

    def addHittable(self, hittable):
        self.hittableList.append(hittable)

    @ti.func 
    def createBoundingBox(self, hittableList):
        '''
        Create the bounding box for the hittable list for testing SAH
        '''
        boundingBox = hittableList[0].boundingBox
        for i in ti.static(range(1, len(hittableList))):
            boundingBox.addBoundingBox(hittableList[i].boundingBox)
        return boundingBox
    
    def sortHittable(self, hittableList, axisIndex):
        '''
        Sort the hittable list according to the minimal values on the axis (required for both SAH and longest axis split)
        '''
        return sorted(hittableList, key = lambda hittable: returnIntervalMin(hittable, axisIndex))

    @ti.kernel 
    def calculateCost(self) -> float:
        '''
        Calculate the SAH cost for the new hittable left and right
        '''
        return self.createBoundingBox(self.hittableLeft).area() * len(self.hittableLeft) + self.createBoundingBox(self.hittableRight).area() * len(self.hittableRight)

    def updateHittableLists(self, hittableLeft, hittableRight):
        '''
        Update the tree's hittable lists to allow calculate cost to be a taichi kernel (massive speedup)
        '''
        self.hittableLeft.clear()
        self.hittableLeft.extend(hittableLeft)
        self.hittableRight.clear()
        self.hittableRight.extend(hittableRight)

    def splitHittable(self, hittableList):
        '''
        Split the hittable list along the most optimal path according to the SAH
        '''
        minCost, bestHittableLeft, bestHittableRight = 0, None, None
        for axisIndex in range(3):
            hittableListAlongAxis = self.sortHittable(hittableList, axisIndex)
            for splitIndex in range(1, len(hittableListAlongAxis)):
                hittableLeft, hittableRight = hittableListAlongAxis[:splitIndex], hittableListAlongAxis[splitIndex:]
                
                self.updateHittableLists(hittableLeft, hittableRight)
                costSplit = self.calculateCost()
                
                if costSplit < minCost or bestHittableLeft == None: 
                    minCost, bestHittableLeft, bestHittableRight = costSplit, hittableLeft, hittableRight 
        return bestHittableLeft, bestHittableRight
    
    @ti.kernel 
    def findAddIndex(self, lenLeft: int, lenRight: int) -> ti.types.vector(2, int): #type: ignore
        addIndexLeft, addIndexRight = 1, 2
        if lenLeft == 1 and lenRight == 1:
            addIndexLeft, addIndexRight = 0, 0 
        elif lenLeft == 1 and lenRight > 1:
            addIndexLeft, addIndexRight = 0, 0 
        elif lenLeft > 1 and lenRight == 1:
            addIndexLeft, addIndexRight = 1, 1 
        return ti.Vector([addIndexLeft, addIndexRight], int)

    def buildTree(self):
        self.nodes.clear()
        self.nodes.append(BVHNode(1, 0, self.hittableList))
        i = 0
        while i < len(self.nodes):
            currentNode = self.nodes[i]

            if not currentNode.isLeaf:
                nodeHittableList = currentNode.hittableList
                hittableLeft, hittableRight = self.splitHittable(nodeHittableList)
                    
                addIndexLeft, addIndexRight = self.findAddIndex(len(self.hittableLeft), len(self.hittableRight))
                self.nodes.append(BVHNode(len(self.nodes), addIndexLeft, hittableLeft))
                self.nodes.append(BVHNode(len(self.nodes), addIndexRight, hittableRight))
            i += 1 

    def walkTree(self, ray, rayHitRecord):
        i = 0
        while True: 
            currentNode = self.nodes[i]
            rayHitRecord = currentNode.hit(ray, rayHitRecord)
            if currentNode.isLeaf:
                return True, currentNode.hittableList[0]
            elif not rayHitRecord.hitAnything:
                return False, 0 
            leftChildIndex, rightChildIndex = currentNode.leftChild, currentNode.rightChild 

            leftChild, rightChild = self.nodes[leftChildIndex], self.nodes[rightChildIndex]
            leftChildHitRecord, rightChildHitRecord = leftChild.hit(ray, rayHitRecord), rightChild.hit(ray, rayHitRecord)
            if leftChildHitRecord.tInterval.maxValue < rightChildHitRecord.tInterval.maxValue:
                i = leftChildIndex
            else:
                i = rightChildIndex
    
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

testTree = BVHTree(camera.hittable)
testNode = testTree.nodes[0]

@ti.kernel 
def testHit():
    didHit, rayHitRecord = testNode.hit(ray3(vec3(0, 0, 1), vec3(0, 0, -1)), initDefaultHitRecord(interval(0.001, 1e10)))
    print(didHit)

def testSplit():
    newList = testTree.sortHittable(testTree.hittableList, 0)
    for hittable in newList:
        print(hittable.boundingBox.x.minValue)

def testSAH():
    hittableLeft, hittableRight = testTree.splitHittable(testTree.hittableList)
    print(hittableLeft, hittableRight)

def testTreeBuild():
    start = time.perf_counter()
    testTree.buildTree()
    end = time.perf_counter()
    print(end - start)

newTest = aabb(interval(0, 5), interval(-2, 2), interval(-3, 2))

@ti.kernel 
def testArea():
    print(newTest.area())

#testArea()
#testHit()
#testSplit()
#testSAH()
testTreeBuild()
testTree.addHittable(sphere3(vec3(0, 0, 0), 0.5, materialFront))
testTreeBuild()

globalHitRecord = hitRecord(False, vec3(0, 0, 0), vec3(0, 0, 0), True, vec3(0, 0, 0), ray3(vec3(0, 0, 0), vec3(0, 0, 0)), vec3(0, 0, 0), interval(0, 1e10), True)

start = time.perf_counter()
testTree.walkTree(ray3(vec3(0, 0, 0), vec3(0, 0, -1)), globalHitRecord)
end = time.perf_counter()
print(end - start)
print([node.isLeaf for node in testTree.nodes].count(True))