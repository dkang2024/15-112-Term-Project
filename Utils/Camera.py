from Vectors import *
from Rays import *
from Objects import *
from World import *
from Interval import *
from Hittable import * 

import warnings
warnings.filterwarnings("ignore") #Taichi throws warnings because classes are used in ti.kernel. We want to ignore these warnings (the classes are specifically designed to allow taichi to work)

@ti.kernel
def calculateImageHeight(imageWidth: int, aspectRatio: float) -> int:
    '''
    Calculates image height given the image's width and aspect ratio
    '''
    return ti.ceil(imageWidth / aspectRatio, int) #Ceiling ensures imageHeight >= 1

@ti.kernel
def calculateViewportWidth(viewportHeight: float, imageWidth: int, imageHeight: int) -> float: 
    '''
    Calcuates viewport width given the viewport height, image width, and image height
    '''
    return viewportHeight * (imageWidth / imageHeight)

@ti.kernel 
def calculateViewportHeight(fov: float, focalLength: float) -> float: 
    '''
    Calculates the viewport height given the camera's vertical FOV and focal length 
    '''
    tanTheta = ti.tan(tm.radians(fov) / 2)
    return ti.abs(2 * tanTheta * focalLength) 

@ti.kernel
def calculatePixelDelta(viewportVector: vec3, imageLen: float) -> vec3: #type: ignore 
    '''
    Calculates the displacement vector in between pixels on the image in the viewport. Note that this only calculates in one dimension because Taichi kernels can only return one value. 
    '''
    return viewportVector / imageLen

@ti.kernel
def calculateFirstPixelPos(cameraPos: vec3, viewportWidthVector: vec3, viewportHeightVector: vec3, pixelDX: vec3, pixelDY: vec3, focalLength: float, k: vec3) -> vec3: #type: ignore
    '''
    Calculates the position of the first pixel on the viewport in terms of the world's coordinate system
    '''
    viewportBottomLeftPos = cameraPos - focalLength * k - (viewportWidthVector + viewportHeightVector) / 2
    return viewportBottomLeftPos + (pixelDX + pixelDY) / 2

@ti.kernel 
def calculateCameraK(cameraPos: vec3, lookAt: vec3) -> vec3: #type: ignore
    '''
    Calculate the camera's unit vector in the +z direction
    '''
    return tm.normalize(cameraPos - lookAt)

@ti.kernel 
def calculateCameraI(vectorUp: vec3, k: vec3) -> vec3: #type: ignore 
    '''
    Calculate the camera's unit vector in the +x direction
    '''
    crossProduct = tm.cross(vectorUp, k)
    if nearZero(crossProduct):
        crossProduct = tm.cross(vec3(0, 0, 1), k)
    return tm.normalize(crossProduct)

@ti.kernel 
def calculateCameraJ(i: vec3, k: vec3) -> vec3: #type: ignore
    '''
    Calculate the camera's unit vector in the +y direction
    '''
    return tm.cross(k, i)

@ti.data_oriented 
class Camera(World): 
    '''
    Class for a camera with render capabilities. Add on the world list to the camera for ease of use (Taichi kernels don't accept classes as arguments)
    '''
    def __init__(self, cameraPos: vec3, imageWidth: int, fov: float, lookAt: vec3, aspectRatio: float, tMin: float, tMax: float, samplesPerPixel: int, maxDepth: int, vectorUp = vec3(0, 1, 0)): #type: ignore
        super().__init__()
        self.imageWidth, self.imageHeight = imageWidth, calculateImageHeight(imageWidth, aspectRatio)
        self.tInterval, self.samplesPerPixel, self.maxDepth = interval(tMin, tMax), samplesPerPixel, maxDepth
        self.pixelField = ti.Vector.field(3, float, shape = (self.imageWidth, self.imageHeight))

        self.setCamera(cameraPos, lookAt, fov, vectorUp)

    def setCamera(self, cameraPos, lookAt, fov, vectorUp = vec3(0, 1, 0)):
        '''
        Set the camera to have these values
        '''
        self.cameraPos, self.lookAt, self.fov = cameraPos, lookAt, fov
        self.focalLength = magnitude(self.cameraPos - self.lookAt)
        
        self.k = calculateCameraK(self.cameraPos, self.lookAt)
        self.i = calculateCameraI(vectorUp, self.k)
        self.j = calculateCameraJ(self.i, self.k)

        self.viewportHeight = calculateViewportHeight(self.fov, self.focalLength)
        self.viewportWidth = calculateViewportWidth(self.viewportHeight, self.imageWidth, self.imageHeight)
        
        viewportWidthVector, viewportHeightVector = self.viewportWidth * self.i, self.viewportHeight * self.j
        self.pixelDX, self.pixelDY = calculatePixelDelta(viewportWidthVector, self.imageWidth), calculatePixelDelta(viewportHeightVector, self.imageHeight) 
        self.initPixelPos = calculateFirstPixelPos(self.cameraPos, viewportWidthVector, viewportHeightVector, self.pixelDX, self.pixelDY, self.focalLength, self.k)

    @ti.func 
    def getRayColor(self, ray): 
        '''
        Get ray color with support for recursion for bouncing light off of objects. Taichi doesn't support return in if statements so I have to use separate solution. 
        '''

        lightColor, throughput = vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0)
        for _ in range(self.maxDepth):
            didHit, rayHitRecord = self.hitObjects(ray, initDefaultHitRecord(self.tInterval))
    
            if didHit and rayHitRecord.didRayScatter:
                ray = rayHitRecord.rayScatter
                throughput *= rayHitRecord.rayColor
            elif didHit and not rayHitRecord.didRayScatter:
                break
            else:
                rayDirY = getY(tm.normalize(ray.direction))
                a = 0.5 * (rayDirY + 1)
                lightColor = (1 - a) * vec3(1, 1, 1) + a * vec3(0.5, 0.7, 1.0)
                break

        return lightColor * throughput
    
    @ti.func 
    def samplePixel(self):
        '''
        Returns a random vector with x and y ranging from [-0.5, 0.5] in order to get rays to sample random positions in the viewport for antialiasing
        '''
        return vec3(ti.random() - 0.5, ti.random() - 0.5, 0)

    @ti.func 
    def constructRay(self, i, j):
        '''
        Construct the ray from the camera to the viewport
        '''
        pixelOffset = self.samplePixel()
        rayDir = self.initPixelPos + (i + pixelOffset) * self.pixelDX + (j + pixelOffset) * self.pixelDY - self.cameraPos
        return ray3(self.cameraPos, rayDir)
    
    @ti.func 
    def linearToGamma(self, pixel):
        '''
        Do gamma correction on the pixel
        '''
        return tm.sqrt(pixel)

    @ti.func 
    def antialiasing(self, i, j):
        '''
        Implmement basic antialiasing for pixels
        '''
        pixelColor = vec3(0, 0, 0)
        for _ in ti.static(range(self.samplesPerPixel)):
            pixelColor += self.getRayColor(self.constructRay(i, j))
        return self.linearToGamma(pixelColor / self.samplesPerPixel)

    @ti.kernel
    def render(self): 
        '''
        Render the camera's scene to a matrix that can be displayed
        '''
        for i, j in self.pixelField:
            self.pixelField[i, j] = self.antialiasing(i, j)
        