from Rays import * 
from Interval import *

def setInterval(x1, x2):
    return interval(min(x1, x2), max(x1, x2))

def createBoundingBox(p1, p2):
    x = setInterval(getKernelX(p1), getKernelX(p2))
    y = setInterval(getKernelY(p1), getKernelY(p2))
    z = setInterval(getKernelZ(p1), getKernelZ(p2))
    return aabb(x, y, z)

@ti.dataclass
class aabb:
    '''
    Axis aligned bounding box for BVH ray tracing optimizations
    '''
    x: interval 
    y: interval 
    z: interval 
    
    @ti.func
    def addBoundingBox(self, bb2: ti.template()): #type: ignore
        '''
        Add the interval of another bounding box to create a larger bounding box 
        '''
        return aabb(self.x.addInterval(bb2.x), self.y.addInterval(bb2.y), self.z.addInterval(bb2.z))
    
    @ti.func
    def getIntervalWithIndex(self, index):
        '''
        Return the correct interval utilizing an index (useful for the for loop because of Taichi limitations)
        '''
        pointInterval = self.x 
        if index == 1: 
            pointInterval = self.y 
        elif index == 2: 
            pointInterval = self.z 
        return pointInterval

    @ti.func 
    def calculateIntersection(self, axisValue, rayOrigin, rayDirection):
        return (axisValue - rayOrigin) / rayDirection

    @ti.func 
    def hit(self, ray, tInterval):
        '''
        Check whether the ray hits the axis aligned bounding box. 
        '''
        wasHit = True
        for i in range(3):
            axisInterval = self.getIntervalWithIndex(i)

            t0 = self.calculateIntersection(axisInterval.minValue, ray.origin[i], ray.direction[i])
            t1 = self.calculateIntersection(axisInterval.maxValue, ray.origin[i], ray.direction[i])
            t0, t1 = ti.min(t0, t1), ti.max(t0, t1)

            if t0 > tInterval.minValue:
                tInterval.minValue = t0 
            if t1 < tInterval.maxValue:
                tInterval.maxValue = t1 

            if tInterval.maxValue <= tInterval.minValue:
                wasHit = False 
                break
        return wasHit
        