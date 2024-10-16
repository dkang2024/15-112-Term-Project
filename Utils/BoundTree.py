from BoundBox import * 
import copy 
from Objects import * 

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
    def __init__(self, treeLength, hittableList):
        self.hittableList = copy.copy(hittableList)
        
        self.isLeaf = False 
        if len(self.hittableList) == 1:
            self.isLeaf = True 
        
        self.nodeIndex = treeLength - 1
        self.leftChild, self.rightChild = treeLength, treeLength + 1  
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

    @ti.func 
    def hit(self, ray, rayHitRecord):
        '''
        Check whether the ray hits this bounding box
        '''
        didHit, rayHitRecord.tInterval = self.boundingBox.hit(ray, copyInterval(rayHitRecord.tInterval))
        return didHit, rayHitRecord

@ti.data_oriented 
class BVHTree:
    '''
    Create the BVH Tree and allow easy access to go througb it
    '''
    def __init__(self, hittableList):
        self.nodes, self.hittableList = [BVHNode(1, hittableList)], hittableList

        self.hittableLeft, self.hittableRight = [], []

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
    
    def buildTree(self):
        i = 0
        while i < len(self.nodes):
            currentNode = self.nodes[i]

            if not currentNode.isLeaf:
                nodeHittableList = currentNode.hittableList
                hittableLeft, hittableRight = self.splitHittable(nodeHittableList)
                self.nodes.append(BVHNode(len(self.nodes), hittableLeft))
                self.nodes.append(BVHNode(len(self.nodes), hittableRight))
            i += 1 

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
    testTree.buildTree()

newTest = aabb(interval(0, 5), interval(-2, 2), interval(-3, 2))

@ti.kernel 
def testArea():
    print(newTest.area())

#testArea()
#testHit()
#testSplit()
#testSAH()
testTreeBuild()
print([node.isLeaf for node in testTree.nodes].count(True))