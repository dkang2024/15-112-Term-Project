from Vectors import *
from Rays import *
from Objects import *
from World import *
from Interval import *
from Hittable import * 

import warnings
warnings.filterwarnings("ignore") #Taichi throws warnings because list methods are used (and Taichi doesn't handle these but Python does). We want to ignore these warnings (the classes are specifically designed to allow taichi to work)

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

    if window.is_pressed(ti.ui.SPACE) and not window.is_pressed(ti.ui.SHIFT):
        camera.setMovementY(1)
    elif window.is_pressed(ti.ui.SPACE) and window.is_pressed(ti.ui.SHIFT):
        camera.setMovementY(-1)
    else:
        camera.setMovementY(0)

    if window.is_pressed(ti.ui.UP, 'w'):
        camera.setMovementZ(-1)
    elif window.is_pressed(ti.ui.DOWN, 's'):
        camera.setMovementZ(1)
    else: 
        camera.setMovementZ(0)

def cameraMouseMovement(camera, window):
    '''
    Allow the camera to change what it's looking at using the mouse
    '''
    mouseX, mouseY = window.get_cursor_pos()    
    camera.mousePositions.setMouseX(mouseX)
    camera.mousePositions.setMouseY(mouseY)

@ti.kernel
def calculateImageHeight(imageWidth: int, aspectRatio: float) -> int:
    '''
    Calculates image height given the image's width and aspect ratio
    '''
    return ti.ceil(imageWidth / aspectRatio, int) #Ceiling ensures imageHeight >= 1

@ti.data_oriented 
class cameraMousePositions: 
    '''
    Store the data for the position of the mouse for the camera
    '''

    def __init__(self):
        self.mousePositionField = ti.field(float, shape = (2,))

    @ti.func 
    def mouseX(self):
        return self.mousePositionField[0]
    
    @ti.func
    def mouseY(self):
        return self.mousePositionField[1]
    
    @ti.kernel 
    def setMouseX(self, mouseX: float):
        self.mousePositionField[0] = mouseX 
    
    @ti.kernel 
    def setMouseY(self, mouseY: float):
        self.mousePositionField[1] = mouseY

@ti.data_oriented
class cameraMovement:
    '''
    Store the data for camera position and movement in a Taichi field because everything else is immutatable for Taichi
    '''
    def __init__(self):
        self.positionField, self.lookAtField, self.movementField = ti.Vector.field(3, float, shape = (2,)), ti.Vector.field(3, float, shape = ()), ti.field(int, shape = (3,))

    @ti.func
    def cameraPos(self):
        return self.positionField[0]

    @ti.func 
    def lookAtPreRotation(self):
        return self.positionField[1]
    
    @ti.func 
    def lookAtPostRotation(self):
        return self.lookAtField[None]
    
    @ti.func 
    def dirX(self):
        return self.movementField[0]
    
    @ti.func 
    def dirY(self):
        return self.movementField[1]
    
    @ti.func 
    def dirZ(self):
        return self.movementField[2]
    
    @ti.func 
    def moveCamera(self, cameraSpeed, unitVectors):
        '''
        Move the camera position / what it's looking at 
        '''
        deltaX, deltaY, deltaZ = self.dirX() * cameraSpeed * unitVectors.i(), self.dirY() * cameraSpeed * unitVectors.j(), self.dirZ() * cameraSpeed * unitVectors.k()
        for i in self.positionField: 
            self.positionField[i] += deltaX + deltaY + deltaZ

    @ti.func 
    def mouseToAngle(self, mousePos):
        '''
        Convert mouse position on the screen to an angle in the camera's i or j direction
        '''
        return 178 * (mousePos - 0.5)

    @ti.func 
    def calculateExtraDisplacement(self, distancePoint, angle, unitVector):
        return distancePoint * ti.tan(tm.radians(angle)) * unitVector

    @ti.func 
    def calculateLookAt(self, unitVectors, mousePositions):
        alpha, beta = self.mouseToAngle(mousePositions.mouseX()), self.mouseToAngle(mousePositions.mouseY())
        distancePoint = tm.distance(self.cameraPos(), self.lookAtPreRotation())

        self.lookAtField[None] = self.positionField[1] + self.calculateExtraDisplacement(distancePoint, alpha, unitVectors.i()) + self.calculateExtraDisplacement(distancePoint, beta, unitVectors.j())
    
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
    def calculateK(self, movement, lookAt): #type: ignore
        '''
        Calculate the camera's unit vector in the +z direction
        '''
        self.unitVectorField[2] = tm.normalize(movement.cameraPos() - lookAt)

@ti.data_oriented 
class cameraIntermediateValues:
    '''
    Store and calculate the camera's intermediate values that are used to calculate more important values
    '''
    def __init__(self):
        self.focalLengthField, self.viewportVectors = ti.field(float, shape = ()), ti.Vector.field(3, float, shape = (2,))

    @ti.func
    def focalLength(self):
        return self.focalLengthField[None]

    @ti.func
    def viewportWidthVector(self):
        return self.viewportVectors[0]
    
    @ti.func 
    def viewportHeightVector(self):
        return self.viewportVectors[1]
    
    @ti.func 
    def calculateFocalLength(self, movement):
        '''
        Calculate the camera's focal length. 
        '''
        cameraPosVector = movement.cameraPos() - movement.lookAtPostRotation()
        self.focalLengthField[None] = tm.dot(cameraPosVector, cameraPosVector) ** 0.5
    
    @ti.func 
    def calculateViewportVectors(self, viewportWidth, viewportHeight, unitVectors):
        self.viewportVectors[0], self.viewportVectors[1] = viewportWidth * unitVectors.i(), viewportHeight * unitVectors.j()

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
    def calculatePixelDelta(self, intermediateValues, imageWidth, imageHeight): 
        '''
        Calculates the displacement vector in between pixels on the image in the viewport and sets them in the Taichi field.
        '''
        self.pixelField[0] = intermediateValues.viewportWidthVector() / imageWidth 
        self.pixelField[1] = intermediateValues.viewportHeightVector() / imageHeight

    @ti.func
    def calculateFirstPixelPos(self, movement, intermediateValues, unitVectors): #type: ignore
        '''
        Calculates the position of the first pixel on the viewport in terms of the world's coordinate system
        '''
        viewportBottomLeftPos = movement.cameraPos() - intermediateValues.focalLength() * unitVectors.k() - (intermediateValues.viewportWidthVector() + intermediateValues.viewportHeightVector()) / 2
        self.pixelField[2] = viewportBottomLeftPos + (self.pixelDX() + self.pixelDY()) / 2

@ti.data_oriented 
class Camera(World): 
    '''
    Class for a camera with render capabilities. Add on the world list to the camera for ease of use (Taichi kernels don't accept classes as arguments)
    '''
    def __init__(self, cameraPos: vec3, imageWidth: int, fov: float, lookAt: vec3, aspectRatio: float, tMin: float, tMax: float, samplesPerPixel: int, maxDepth: int, vectorUp = vec3(0, 1, 0), cameraSpeed = 0.1): #type: ignore
        super().__init__()
        self.cameraSpeed, self.fov, self.vectorUp = cameraSpeed, fov, vectorUp
        self.createCameraMovement(cameraPos, lookAt)
        self.createCameraMousePositions()
        self.unitVectors = cameraUnitVectors()
        self.intermediateValues = cameraIntermediateValues()
        self.renderValues = cameraRenderValues()

        self.imageWidth, self.imageHeight = imageWidth, calculateImageHeight(imageWidth, aspectRatio)
        self.tInterval, self.samplesPerPixel, self.maxDepth = interval(tMin, tMax), samplesPerPixel, maxDepth
        self.pixelField = ti.Vector.field(3, float, shape = (self.imageWidth, self.imageHeight))

        self.setCamera()

    def createCameraMousePositions(self):
        '''
        Initialize to default camera mouse positions
        '''
        self.mousePositions = cameraMousePositions()
        self.mousePositions.mousePositionField.fill(0.5)

    def createCameraMovement(self, cameraPos: vec3, lookAt: vec3): #type: ignore
        '''
        Create the camera movement dataclass
        '''
        self.movement = cameraMovement()
        self.movement.positionField[0] = cameraPos 
        self.movement.positionField[1] = lookAt 
        self.movement.lookAtField[None] = lookAt
    
    @ti.kernel 
    def setMovementX(self, value: int):
        self.movement.movementField[0] = value 
    
    @ti.kernel 
    def setMovementY(self, value: int):
        self.movement.movementField[1] = value

    @ti.kernel 
    def setMovementZ(self, value: int):
        self.movement.movementField[2] = value

    @ti.kernel 
    def calculateUnitVectors(self, isPostRotation: bool):
        '''
        Calculate the camera's unit vectors pre and post rotation
        '''
        if isPostRotation: 
            self.unitVectors.calculateK(self.movement, self.movement.lookAtPostRotation())
        else: 
            self.unitVectors.calculateK(self.movement, self.movement.lookAtPreRotation())
        self.unitVectors.calculateI(self.vectorUp)
        self.unitVectors.calculateJ()

    @ti.func
    def calculateRenderValues(self): #type: ignore
        '''
        Calculate specific values for the camera's render values
        '''
        self.renderValues.calculatePixelDelta(self.intermediateValues, self.imageWidth, self.imageHeight)
        self.renderValues.calculateFirstPixelPos(self.movement, self.intermediateValues, self.unitVectors)
    
    @ti.func
    def calculateViewportWidth(self, viewportHeight: float) -> float: 
        '''
        Calcuate the camera's viewport width
        '''
        return viewportHeight * (self.imageWidth / self.imageHeight)
    
    @ti.func
    def calculateViewportHeight(self) -> float: 
        '''
        Calculate the camera's viewport height
        '''
        tanTheta = ti.tan(tm.radians(self.fov) / 2)
        return ti.abs(2 * tanTheta * self.intermediateValues.focalLength()) 

    @ti.kernel 
    def calculateRender(self):
        '''
        Calculate the render values necessary for the camera, including the intermediate ones necessary for the calculation.
        '''
        self.intermediateValues.calculateFocalLength(self.movement)
        viewportHeight = self.calculateViewportHeight()
        viewportWidth = self.calculateViewportWidth(viewportHeight)
        self.intermediateValues.calculateViewportVectors(viewportWidth, viewportHeight, self.unitVectors)
        self.calculateRenderValues()

    @ti.kernel 
    def moveCamera(self):
        self.movement.moveCamera(self.cameraSpeed, self.unitVectors)

    def setCamera(self): #type: ignore
        '''
        Reset the camera's specific values that depend upon its position and what it's looking at
        '''
        self.moveCamera()
        self.calculateUnitVectors(False)
        self.calculateLookAt()
        self.calculateUnitVectors(True)
        self.calculateRender()

    @ti.kernel 
    def calculateLookAt(self):
        self.movement.calculateLookAt(self.unitVectors, self.mousePositions)

    @ti.func 
    def getRayColor(self, ray): 
        '''
        Get ray color with support for recursion for bouncing light off of objects. Taichi doesn't support return in if statements so I have to use separate solution. 
        '''

        lightColor, throughput = vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0)
        for _ in range(self.maxDepth):
            rayHitRecord = self.hitObjects(ray, initDefaultHitRecord(self.tInterval))
    
            if rayHitRecord.hitAnything and rayHitRecord.didRayScatter:
                ray = rayHitRecord.rayScatter
                throughput *= rayHitRecord.rayColor
            elif rayHitRecord.hitAnything and not rayHitRecord.didRayScatter:
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
        rayDir = self.renderValues.initPixelPos() + (i + pixelOffset) * self.renderValues.pixelDX() + (j + pixelOffset) * self.renderValues.pixelDY() - self.movement.cameraPos()
        return ray3(self.movement.cameraPos(), rayDir)
    
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
        