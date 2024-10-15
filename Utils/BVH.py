from BoundBox import * 
import copy 

@ti.data_oriented
class BVHNode:
    '''
    Implements a node for a BVH
    '''
    def __init__(self, hittableList):
        hittableList = copy.copy(hittableList)

        if len(hittableList) == 1:
            self.boundingBox, self.isObject, self.leftChild = hittableList[0].boundingBox, True, hittableList[0]
        else:
            midIndex = len(hittableList) // 2
            self.leftChild, self.rightChild = BVHNode(hittableList[:midIndex]), BVHNode(hittableList[midIndex:])  
            self.boundingBox = addTwoBoundingBoxes(self.leftChild.boundingBox, self.rightChild.boundingBox)
            self.isObject = False 
        
    @ti.func 
    def hit(self, ray, tempHitRecord):
        didHit, childHit = False, self.leftChild
        if self.boundingBox.hit(ray, tempHitRecord.tInterval):
            didHit = True 
            if self.rightChild.boundingBox.hit(ray, tempHitRecord.tInterval) and not self.isObject:
                childHit = self.rightChild 
        
        return didHit, childHit, self.isObject

