from Vectors import * 

@ti.dataclass
class interval: 
    '''
    Interval class to make dealing with intervals a lot easier. Note that this is an exclusive bounds interval (not that it really matters all too much)
    ''' 
    minValue: float
    maxValue: float
    
    @ti.func 
    def surrounds(self, value):
        '''
        Check if the interval surrounds the value
        '''
        return self.minValue < value and value < self.maxValue
    
    @ti.func 
    def expand(self, delta):
        '''
        Add padding to the sides of the interval
        '''
        padding = delta / 2
        return interval(self.minValue - padding, self.maxValue + padding)
    
@ti.func 
def overlaps(interval1, interval2):
    '''
    Check whether intervals overlap for checking bounding boxes
    '''
    tMin = ti.max(interval1.minValue, interval2.minValue)
    tMax = ti.min(interval1.maxValue, interval2.maxValue)
    return tMin < tMax