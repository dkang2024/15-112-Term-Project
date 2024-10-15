from Vectors import *

@ti.dataclass
class ray3:
    '''
    Class for a ray starting from an origin and heading into a direction given by another vector
    '''
    origin: vec3 #type: ignore 
    direction: vec3 #type: ignore

    @ti.func 
    def pointOnRay(self, t: float) -> float: 
        '''
        Get a point on the ray using the parametric equation
        '''
        return self.origin + self.direction * t
    
@ti.func 
def defaultRay():
    return ray3(defaultVec(), defaultVec())