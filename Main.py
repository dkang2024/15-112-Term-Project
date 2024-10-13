from Utils import *

def cameraKeyMovement(camera, window):
    if window.is_pressed(ti.ui.LEFT, 'a'):
        camera.dirX = -1 
    elif window.is_pressed(ti.ui.RIGHT, 'd'):
        camera.dirX = 1 
    else:
        camera.dirX = 0

    if window.is_pressed(ti.ui.UP, 'w'):
        camera.dirZ = 1
    elif window.is_pressed(ti.ui.DOWN, 's'):
        camera.dirZ = -1
    else: 
        camera.dirZ = 0

def renderScene(cameraPos: vec3, imageWidth: int, fov: float, focalLength: float, aspectRatio: float, samplesPerPixel: float, maxDepth: int, tMin = 0.001, tMax = 1e10): #type: ignore
    camera = Camera(cameraPos, imageWidth, fov, focalLength, aspectRatio, tMin, tMax, samplesPerPixel, maxDepth)

    materialGround = lambertianMaterial(vec3(0.8, 0.8, 0.0))
    materialCenter = lambertianMaterial(vec3(0.1, 0.2, 0.5))
    materialLeft = dielectricMaterial(1.0 / 1.3)
    materialRight = reflectiveMaterial(vec3(0.8, 0.6, 0.2), 1.0)

    camera.addHittable(sphere3(vec3(0, 0, -1), 0.5, materialCenter))
    camera.addHittable(sphere3(vec3(0, -100.5, -1), 100, materialGround))
    camera.addHittable(sphere3(vec3(-1, 0, -1), 0.5, materialLeft))
    camera.addHittable(sphere3(vec3(1, 0, -1), 0.5, materialRight))

    window = ti.ui.Window('Render Test', res = (camera.imageWidth, camera.imageHeight))
    canvas = window.get_canvas()

    while window.running: 
        camera.render()
        canvas.set_image(camera.pixelField)
        window.show()
        
renderScene(vec3(-2, 2, 1), 2000, 90, vec3(0, 0, -1), 16 / 9, 2, 20)