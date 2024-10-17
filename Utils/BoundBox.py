from Rays import * 
from Interval import *

@ti.dataclass
class aabb:
    '''
    Axis aligned bounding box for BVH ray tracing optimizations
    '''
    x: interval 
    y: interval 
    z: interval 
    
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
    def calculateIntersection(self, axisValue, rayOrigin, inverseRayDirection):
        '''
        Calculate the value for the parametric of the ray for when it intersects with the bounding box (slab method)
        '''
        return (axisValue - rayOrigin) * inverseRayDirection

    @ti.func 
    def addBoundingBox(self, bb2):
        '''
        Add another bounding box to this bounding box by adding the intervals
        '''
        self.x = addIntervals(self.x, bb2.x)
        self.y = addIntervals(self.y, bb2.y)
        self.z = addIntervals(self.z, bb2.z)

    @ti.func
    def area(self) -> float:
        '''
        Calculate the bounding box's area
        '''
        xLength, yLength, zLength = self.x.length(), self.y.length(), self.z.length()
        return 2 * xLength * yLength + 2 * xLength * zLength + 2 * yLength * zLength 
    
    @ti.func 
    def centroid(self) -> vec3: #type: ignore
        '''
        Return the position of the centroid of the bounding Box
        '''
        return vec3(self.x.minValue, self.y.minValue, self.z.minValue) + vec3(self.x.length(), self.y.length(), self.z.length()) * 0.5

    @ti.func 
    def hit(self, ray, tempHitRecord):
        '''
        Check whether the ray hits the axis aligned bounding box. 
        '''
        tInterval = tempHitRecord.tInterval
        for i in ti.static(range(3)):
            axisInterval, inverseRayDirection = self.getIntervalWithIndex(i), 1 / ray.direction[i]

            t0 = self.calculateIntersection(axisInterval.minValue, ray.origin[i], inverseRayDirection)
            t1 = self.calculateIntersection(axisInterval.maxValue, ray.origin[i], inverseRayDirection)
            t0, t1 = ti.min(t0, t1), ti.max(t0, t1)

            tInterval.minValue = ti.max(t0, tInterval.minValue)
            tInterval.maxValue = ti.min(t1, tInterval.maxValue)

        tempHitRecord.hitAnything = tInterval.maxValue > tInterval.minValue
        tempHitRecord.tInterval = copyInterval(tInterval)
        return tempHitRecord
        
@ti.func 
def setInterval(x1, x2):
    return interval(ti.min(x1, x2), ti.max(x1, x2))

@ti.kernel 
def createBoundingBox(p1: vec3, p2: vec3) -> aabb: #type: ignore
    '''
    Create a bounding box given two points indicating the minimum possible value and the maximal possible value of the bounding box.
    '''
    x = setInterval(getX(p1), getX(p2))
    y = setInterval(getY(p1), getY(p2))
    z = setInterval(getZ(p1), getZ(p2))
    return aabb(x, y, z)