from Rays import * 
from Interval import *

@ti.dataclass 
class aabb:
    '''
    Axis aligned bounding box for BVH ray tracing optimizations
    '''
    def __init__(self, p1, p2):
        self.x = self.setInterval(getX(p1), getX(p2))
        self.y = self.setInterval(getY(p1), getY(p2))
        self.z = self.setInterval(getZ(p1), getZ(p2))
        
    def setInterval(self, x1, x2):
        '''
        Correctly set intervals with min values on the left and max values on the right
        '''
        return interval(min(x1, x2), max(x1, x2))
    
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
        Check whether the ray hits the axis aligned bounding box
        '''
        wasHit = True
        for i in ti.static(range(3)):
            axisInterval = self.getIntervalWithIndex(i)

            t0 = self.calculateIntersection(axisInterval.minValue, ray.origin[i], ray.direction[i])
            t1 = self.calculateIntersection(axisInterval.maxValue, ray.origin[i], ray.direction[i])
            t0, t1 = ti.min(t0, t1), ti.max(t0, t1)

            if t0 > tInterval.minValue:
                tInterval.minValue = t0 
            elif t1 < tInterval.maxValue:
                tInterval.maxValue = t1 

            if tInterval.maxValue <= tInterval.minValue:
                wasHit = False 
                break
        return wasHit
        