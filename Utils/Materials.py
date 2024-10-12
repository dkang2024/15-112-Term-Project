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
    
@ti.dataclass 
class dielectricMaterial:
    '''
    Class for dielectric materials
    '''
    refractionIndex: float #type: ignore

    @ti.func 
    def scatter(self, rayHitRecord):
        '''
        Scatter rays with a dielectric material
        '''
        etaRatio = self.refractionIndex
        if rayHitRecord.frontFace:
            etaRatio = 1.0 / self.refractionIndex
        
        unitDirection = tm.normalize(rayHitRecord.initRayDir)
        cosTheta = tm.min(tm.dot(-unitDirection, rayHitRecord.normalVector), 1.0)
        sinTheta = tm.sqrt(1 - cosTheta ** 2)
        
        rayDir = defaultVec()
        if etaRatio * sinTheta > 1.0 or self.reflectance(cosTheta, etaRatio) > ti.random(): #Dealing with Total Internal Reflection
            rayDir = reflect(unitDirection, rayHitRecord.normalVector)
        else:
            rayDir = refract(unitDirection, rayHitRecord.normalVector, etaRatio, cosTheta)

        scatteredRay = ray3(rayHitRecord.pointHit, rayDir)
        return True, scatteredRay, vec3(1.0, 1.0, 1.0)
    
    @ti.func 
    def reflectance(self, cosTheta, etaRatio): 
        '''
        Use Schlick's approximation to get the reflectance
        '''
        r0 = (1 - etaRatio) / (1 + etaRatio)
        r0 = r0 ** 2
        return r0 + (1 - r0) * (1 - cosTheta) ** 5

