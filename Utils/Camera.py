from Vectors import *
from Rays import *
from Objects import *
from World import *
from Interval import *
from Hittable import * 

import warnings
warnings.filterwarnings("ignore") #Taichi throws warnings because classes are used in ti.kernel. We want to ignore these warnings (the classes are specifically designed to allow taichi to work)

def cameraKeyMovement(camera, window):
    '''
    Allow the camera to be moved using keys
    '''
    if window.is_pressed(ti.ui.LEFT, 'a'):
        camera.setMovementX(-1) 
    elif window.is_pressed(ti.ui.RIGHT, 'd'):
        camera.setMovementX(1) 
    else:
        camera.setMovementX(0)

    if window.is_pressed(ti.ui.UP, 'w'):
        camera.setMovementZ(-1)
    elif window.is_pressed(ti.ui.DOWN, 's'):
        camera.setMovementZ(1)
    else: 
        camera.setMovementZ(0)

@ti.kernel
def calculateImageHeight(imageWidth: int, aspectRatio: float) -> int:
    '''
    Calculates image height given the image's width and aspect ratio
    '''
    return ti.ceil(imageWidth / aspectRatio, int) #Ceiling ensures imageHeight >= 1

@ti.kernel 
def defaultCamVec() -> vec3: #type: ignore
    '''
    Allow default vectors to be created in Python's scope
    '''
    return vec3(0, 0, 0)

@ti.data_oriented
class cameraMovement:
    '''
    Store the data for camera position and movement in a Taichi field because everything else is immutatable for Taichi
    '''
    def __init__(self):
        self.positionField, self.movementField = ti.Vector.field(3, float, shape = (2,)), ti.field(int, shape = (2,))

    @ti.func
    def cameraPos(self):
        return self.positionField[0]

    @ti.func 
    def lookAt(self):
        return self.positionField[1]
    
    @ti.func 
    def dirX(self):
        return self.movementField[0]
    
    @ti.func 
    def dirZ(self):
        return self.movementField[1]
    
    @ti.func 
    def moveCamera(self, cameraSpeed, unitVectors):
        '''
        Move the camera position / what it's looking at 
        '''
        deltaX, deltaZ = self.dirX() * cameraSpeed * unitVectors.i(), self.dirZ() * cameraSpeed * unitVectors.k()
        for i in self.positionField: 
            self.positionField[i] += deltaX 
            self.positionField[i] += deltaZ
    
@ti.data_oriented
class cameraUnitVectors: 
    '''
    Store and manipulate the data for the camera's unit vectors in a Taichi field because everything else is immutatable for Taichi
    '''
    def __init__(self):
        self.unitVectorField = ti.Vector.field(3, float, shape = (3,))

    @ti.func 
    def i(self):
        return self.unitVectorField[0]
    
    @ti.func 
    def j(self):
        return self.unitVectorField[1]
    
    @ti.func 
    def k(self):
        return self.unitVectorField[2]

    @ti.func
    def calculateI(self, vectorUp): #type: ignore 
        '''
        Calculate the camera's unit vector in the +x direction
        '''
        crossProduct = tm.cross(vectorUp, self.k())
        if nearZero(crossProduct):
            crossProduct = tm.cross(vec3(0, 0, 1), self.k())
        self.unitVectorField[0] = tm.normalize(crossProduct)
    
    @ti.func
    def calculateJ(self): #type: ignore
        '''
        Calculate the camera's unit vector in the +y direction
        '''
        self.unitVectorField[1] = tm.cross(self.k(), self.i())
    
    @ti.func
    def calculateK(self, movement: ti.template()): #type: ignore
        '''
        Calculate the camera's unit vector in the +z direction
        '''
        self.unitVectorField[2] = tm.normalize(movement.cameraPos() - movement.lookAt())

@ti.data_oriented
class cameraRenderValues:
    '''
    Store and calculate the camera's important render values
    '''  
    def __init__(self):
        self.pixelField = ti.Vector.field(3, float, shape = (3,))
    
    @ti.func 
    def pixelDX(self):
        return self.pixelField[0]
    
    @ti.func 
    def pixelDY(self):
        return self.pixelField[1]
    
    @ti.func 
    def initPixelPos(self):
        return self.pixelField[2]

    @ti.func
    def calculatePixelDelta(self, viewportWidthVector, imageWidth, viewportHeightVector, imageHeight): 
        '''
        Calculates the displacement vector in between pixels on the image in the viewport and sets them in the Taichi field.
        '''
        self.pixelField[0] = viewportWidthVector / imageWidth 
        self.pixelField[1] = viewportHeightVector / imageHeight

    @ti.func
    def calculateFirstPixelPos(self, cameraPos: vec3, viewportWidthVector: vec3, viewportHeightVector: vec3, focalLength: float, k: vec3): #type: ignore
        '''
        Calculates the position of the first pixel on the viewport in terms of the world's coordinate system
        '''
        viewportBottomLeftPos = cameraPos - focalLength * k - (viewportWidthVector + viewportHeightVector) / 2
        self.pixelField[2] = viewportBottomLeftPos + (self.pixelDX() + self.pixelDY()) / 2

@ti.data_oriented 
class Camera(World): 
    '''
    Class for a camera with render capabilities. Add on the world list to the camera for ease of use (Taichi kernels don't accept classes as arguments)
    '''
    def __init__(self, cameraPos: vec3, imageWidth: int, fov: float, lookAt: vec3, aspectRatio: float, tMin: float, tMax: float, samplesPerPixel: int, maxDepth: int, vectorUp = vec3(0, 1, 0), cameraSpeed = 0.2): #type: ignore
        super().__init__()
        self.cameraSpeed, self.fov, self.vectorUp = cameraSpeed, fov, vectorUp
        self.createCameraMovement(cameraPos, lookAt)
        self.cameraUnitVectors = cameraUnitVectors()
        self.cameraRenderValues = cameraRenderValues()

        self.imageWidth, self.imageHeight = imageWidth, calculateImageHeight(imageWidth, aspectRatio)
        self.tInterval, self.samplesPerPixel, self.maxDepth = interval(tMin, tMax), samplesPerPixel, maxDepth
        self.pixelField = ti.Vector.field(3, float, shape = (self.imageWidth, self.imageHeight))

        self.setCamera()

    def createCameraMovement(self, cameraPos: vec3, lookAt: vec3): #type: ignore
        '''
        Create the camera movement dataclass
        '''
        self.cameraMovement = cameraMovement()
        self.cameraMovement.positionField[0] = cameraPos 
        self.cameraMovement.positionField[1] = lookAt 
    
    @ti.kernel 
    def setMovementX(self, value: int):
        self.cameraMovement.movementField[0] = value 
    
    @ti.kernel 
    def setMovementZ(self, value: int):
        self.cameraMovement.movementField[1] = value

    @ti.kernel 
    def calculateUnitVectors(self): #type: ignore 
        '''
        Calculate the camera's unit vectors
        '''
        self.cameraUnitVectors.calculateK(self.cameraMovement)
        self.cameraUnitVectors.calculateI(self.vectorUp)
        self.cameraUnitVectors.calculateJ()

    @ti.func
    def calculateRenderValues(self, viewportWidthVector: vec3, viewportHeightVector: vec3, focalLength: float): #type: ignore
        '''
        Calculate specific values for the camera's render values
        '''
        self.cameraRenderValues.calculatePixelDelta(viewportWidthVector, self.imageWidth, viewportHeightVector, self.imageHeight)
        self.cameraRenderValues.calculateFirstPixelPos(self.cameraMovement.cameraPos(), viewportWidthVector, viewportHeightVector, focalLength, self.cameraUnitVectors.k())

    @ti.func 
    def calculateFocalLength(self):
        '''
        Calculate the camera's focal length. 
        '''
        cameraPosVector = self.cameraMovement.cameraPos() - self.cameraMovement.lookAt()
        return tm.dot(cameraPosVector, cameraPosVector) ** 0.5
    
    @ti.func
    def calculateViewportWidth(self, viewportHeight: float) -> float: 
        '''
        Calcuate the camera's viewport width
        '''
        return viewportHeight * (self.imageWidth / self.imageHeight)
    
    @ti.func
    def calculateViewportHeight(self, focalLength: float) -> float: 
        '''
        Calculate the camera's viewport height
        '''
        tanTheta = ti.tan(tm.radians(self.fov) / 2)
        return ti.abs(2 * tanTheta * focalLength) 

    @ti.kernel 
    def calculateRender(self):
        '''
        Calculate the render values necessary for the camera, including the intermediate ones necessary for the calculation.
        '''
        focalLength = self.calculateFocalLength()
        viewportHeight = self.calculateViewportHeight(focalLength)
        viewportWidth = self.calculateViewportWidth(viewportHeight)
        viewportWidthVector, viewportHeightVector = viewportWidth * self.cameraUnitVectors.i(), viewportHeight * self.cameraUnitVectors.j()
        self.calculateRenderValues(viewportWidthVector, viewportHeightVector, focalLength)

    @ti.kernel 
    def moveCamera(self):
        self.cameraMovement.moveCamera(self.cameraSpeed, self.cameraUnitVectors)

    def setCamera(self): #type: ignore
        '''
        Reset the camera's specific values that depend upon its position and what it's looking at
        '''
        self.moveCamera()
        self.calculateUnitVectors()
        self.calculateRender()

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
        rayDir = self.cameraRenderValues.initPixelPos() + (i + pixelOffset) * self.cameraRenderValues.pixelDX() + (j + pixelOffset) * self.cameraRenderValues.pixelDY() - self.cameraMovement.cameraPos()
        return ray3(self.cameraMovement.cameraPos(), rayDir)
    
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
        