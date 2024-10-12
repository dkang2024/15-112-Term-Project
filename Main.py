from Utils import *

def renderScene(cameraPos: vec3, imageWidth: int, viewportHeight: float, focalLength: float, aspectRatio: float, samplesPerPixel: float, maxDepth: int, tMin = 0.001, tMax = 1e10): #type: ignore
    camera = Camera(cameraPos, imageWidth, viewportHeight, focalLength, aspectRatio, tMin, tMax, samplesPerPixel, maxDepth)

    materialGround = lambertianMaterial(vec3(0.8, 0.8, 0.0))
    materialCenter = lambertianMaterial(vec3(0.1, 0.2, 0.5))
    materialLeft = reflectiveMaterial(vec3(0.8, 0.8, 0.8), 0.3)
    materialRight = reflectiveMaterial(vec3(0.8, 0.6, 0.2), 1.0)

    camera.addHittable(sphere3(vec3(0, 0, -1), 0.5, materialCenter))
    camera.addHittable(sphere3(vec3(0, -100.5, -1), 100, materialGround))
    camera.addHittable(sphere3(vec3(-1, 0, -1), 0.5, materialLeft))
    camera.addHittable(sphere3(vec3(1, 0, -1), 0.5, materialRight))

    gui = ti.ui.Window('Render Test', res = (camera.imageWidth, camera.imageHeight))
    canvas = gui.get_canvas()

    while gui.running: 
        camera.render()
        canvas.set_image(camera.pixelField)
        gui.show()
        
renderScene(vec3(0, 0, 0), 2000, 2, 1, 16 / 9, 2, 4)