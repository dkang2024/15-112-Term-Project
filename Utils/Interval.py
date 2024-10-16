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
    def length(self):
        '''
        Return the interval's length
        '''
        return self.maxValue - self.minValue
    
@ti.func 
def copyInterval(intervalValue):
    return interval(intervalValue.minValue, intervalValue.maxValue)

@ti.func 
def addIntervals(interval1, interval2):
    '''
    Add two intervals to create a new interval encapsulating both
    '''
    minValue = ti.min(interval1.minValue, interval2.minValue)
    maxValue = ti.max(interval1.maxValue, interval2.maxValue)
    return interval(minValue, maxValue)