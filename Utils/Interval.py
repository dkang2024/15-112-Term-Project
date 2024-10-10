from Vectors import * 

@ti.data_oriented 
class Interval: 
    '''
    Interval class to make dealing with intervals a lot easier 
    ''' 

    def __init__(self, minValue, maxValue):
        self.min, self.max = minValue, maxValue 

    @ti.func 
    def size(self):
        return self.max - self.min 
    
    @ti.func 
    def contains(self, value):
        return self.min <= value and value <= self.max 
    
    @ti.func 
    def surrounds(self, value):
        return self.min < value and value < self.max 