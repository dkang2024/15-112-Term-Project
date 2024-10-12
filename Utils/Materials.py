from Rays import *
from Hittable import *

@ti.func 
def defaultMaterial():
    '''
    Initialize the default material 
    '''
    return lambertianMaterial(1.0, 1.0)

@ti.dataclass 
class lambertianMaterial:
    '''
    Class for Lambertian materials and dealing with ray scattering for them 
    '''
    albedo: float 

    @ti.func 
    def scatter(self, rayHitRecord):
        '''
        Scatter rays with a lambertian material 
        '''
        scatteredRay = ray3(rayHitRecord.pointHit, rayHitRecord.normalVector + randomVectorOnUnitSphere())

        if nearZero(scatteredRay.direction): #Deal with the possibility of returning a 0 direction vector scattering
            scatteredRay.direction = rayHitRecord.normalVector

        return scatteredRay, self.albedo