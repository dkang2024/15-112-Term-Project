from Vectors import * 

@ti.dataclass
class interval: 
    '''
    Interval class to make dealing with intervals a lot easier 
    ''' 
    minValue: float
    maxValue: float
    
    @ti.func 
    def surrounds(self, value):
        return self.minValue < value and value < self.maxValue