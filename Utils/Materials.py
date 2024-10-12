from Rays import *
from Hittable import *

@ti.dataclass 
class lambertianMaterial:
    '''
    Class for Lambertian materials and dealing with ray scattering for them 
    '''
    color: vec3 #type: ignore

    @ti.func 
    def scatter(self, rayHitRecord):
        '''
        Scatter rays with a lambertian material 
        '''
        scatteredRay = ray3(rayHitRecord.pointHit, rayHitRecord.normalVector + randomVectorOnUnitSphere())

        if nearZero(scatteredRay.direction): #Deal with the possibility of returning a 0 direction vector scattering
            scatteredRay.direction = rayHitRecord.normalVector

        return True, scatteredRay, self.color
    
@ti.dataclass 
class reflectiveMaterial:
    '''
    Class for materials that include reflections
    '''
    color: vec3 #type: ignore 
    fuzz: float #type: ignore

    @ti.func 
    def scatter(self, rayHitRecord):
        '''
        Scatter rays with a reflective material
        '''
        reflectDir = reflect(rayHitRecord.initRayDir, rayHitRecord.normalVector)
        reflectDir = tm.normalize(reflectDir) + self.fuzz * randomVectorOnUnitSphere()
        scatteredRay = ray3(rayHitRecord.pointHit, reflectDir)
        return tm.dot(reflectDir, rayHitRecord.normalVector) > 0, scatteredRay, self.color