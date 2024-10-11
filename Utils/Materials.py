from Rays import *
from Hittable import *

@ti.dataclass 
class lambertianMaterial:

    @ti.func 
    def scatter(ray, hitRecord, color):
        return False 