from Vectors import *

@ti.data_oriented
class ray3:    
    def __init__(self, origin: vec3, direction: vec3): #type: ignore
        self.origin, self.direction = origin, direction 

    @ti.func 
    def pointOnRay(self, t: float) -> float: 
        '''
        Get a point on the ray using the parametric equation
        '''
        return self.origin + self.direction * t