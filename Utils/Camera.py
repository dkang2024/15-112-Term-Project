from Vectors import *
from Rays import *
from Objects import *
from World import *

import warnings
warnings.filterwarnings("ignore") #Taichi throws warnings because classes are used in ti.kernel. We want to ignore these warnings (the classes are specifically designed to allow taichi to work)

@ti.kernel
def calculateImageHeight(imageWidth: int, aspectRatio: float) -> int:
    '''
    Calculates image height given the image's width and aspect ratio
    '''
    return ti.ceil(imageWidth / aspectRatio, int) #Ceiling ensures imageHeight >= 1

@ti.kernel
def calculateViewportHeight(viewportWidth: float, imageWidth: int, imageHeight: int) -> float: 
    '''
    Calcuates viewport height given the viewport width, image width, and image height
    '''
    return viewportWidth * (imageHeight / imageWidth)

@ti.kernel
def calculatePixelDelta(viewportVector: vec3, imageLen: float) -> vec3: #type: ignore 
    '''
    Calculates the displacement vector in between pixels on the image in the viewport. Note that this only calculates in one dimension because Taichi kernels can only return one value. 
    '''
    return viewportVector / imageLen

@ti.kernel
def calculateFirstPixelPos(cameraPos: vec3, viewportWidthVector: vec3, viewportHeightVector: vec3, pixelDX: vec3, pixelDY: vec3, focalLength: float) -> vec3: #type: ignore
    '''
    Calculates the position of the first pixel on the viewport in terms of the camera's coordinate system
    '''
    viewportUpperLeftPos = cameraPos - vec3(0, 0, focalLength) - (viewportWidthVector + viewportHeightVector) / 2
    return viewportUpperLeftPos + (pixelDX + pixelDY) / 2

newWorld = World()
newWorld.addHittable(sphere3(vec3(0, 0, -2), 0.5))

@ti.func 
def getRayColor(ray):
    colorReturn, t = vec3(1, 0, 0), newWorld.hitObjects(ray.origin, ray.direction)
    if t < 0.0:
        rayDir = tm.normalize(ray.direction)
        a = 0.5 * (rayDir + 1)
        colorReturn = (1 - a) * vec3(1, 1, 1) + a * vec3(0.5, 0.7, 1.0)
    return colorReturn

@ti.data_oriented 
class Camera: 
    '''
    Class for a camera with render capabilities
    '''

    def __init__(self, cameraPos: vec3, imageWidth: int, viewportWidth: float, focalLength: float, aspectRatio: float): #type: ignore
        self.imageWidth, self.imageHeight = imageWidth, calculateImageHeight(imageWidth, aspectRatio)
        self.viewportWidth, self.viewportHeight = viewportWidth, calculateViewportHeight(viewportWidth, self.imageWidth, self.imageHeight)
        self.cameraPos = cameraPos
        
        viewportWidthVector, viewportHeightVector = vec3(self.viewportWidth, 0, 0), vec3(0, -self.viewportHeight, 0) #Note that we take the negative because the camera coordinates have up as positive, but the viewport coordinates have down as positive 
        self.pixelDX, self.pixelDY = calculatePixelDelta(viewportWidthVector, self.imageWidth), calculatePixelDelta(viewportHeightVector, self.imageHeight) 
        self.initPixelPos = calculateFirstPixelPos(self.cameraPos, viewportWidthVector, viewportHeightVector, self.pixelDX, self.pixelDY, focalLength)

        self.pixelField = ti.Vector.field(3, float, shape = (self.imageWidth, self.imageHeight))

    @ti.kernel
    def render(self): 
        '''
        Render the camera's scene to a matrix that can be displayed
        '''
        for i, j in self.pixelField:
            pixelPos = self.initPixelPos + i * self.pixelDX + j * self.pixelDY 
            rayDir = pixelPos - self.cameraPos 
            cameraRay = ray3(self.cameraPos, rayDir)
            self.pixelField[i, j] = getRayColor(cameraRay)
        