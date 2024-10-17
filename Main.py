from Utils import *

def renderScene(cameraPos: vec3, imageWidth: int, fov: float, focalLength: float, aspectRatio: float, samplesPerPixel: float, maxDepth: int, tMin = 0.001, tMax = 1e10): #type: ignore
    camera = Camera(cameraPos, imageWidth, fov, focalLength, aspectRatio, tMin, tMax, samplesPerPixel, maxDepth)

    materialGround = lambertianMaterial(vec3(0.8, 0.8, 0.0))
    materialCenter = lambertianMaterial(vec3(0.1, 0.2, 0.5))
    materialLeft = dielectricMaterial(1.0 / 1.3)
    materialRight = reflectiveMaterial(vec3(0.8, 0.6, 0.2), 0.5)
    materialFront = reflectiveMaterial(vec3(0.8, 0.8, 0.8), 0.2)

    camera.addHittable(sphere3(vec3(0, 0, -1), 0.5, materialCenter))
    camera.addHittable(sphere3(vec3(0, -100.5, -1), 100, materialGround))
    camera.addHittable(sphere3(vec3(-1, 0, -1), 0.5, materialLeft))
    camera.addHittable(sphere3(vec3(1, 0, -1), 0.5, materialRight))
    camera.addHittable(sphere3(vec3(0, 0, 0), 0.5, materialFront))

    window = ti.ui.Window('Render Test', res = (camera.imageWidth, camera.imageHeight), pos = (100, 100))
    canvas = window.get_canvas()

    while window.running: 
        cameraKeyMovement(camera, window)
        cameraMouseMovement(camera, window)
        camera.setCamera()
        camera.render()
        canvas.set_image(camera.pixelField)
        window.show()
        
renderScene(vec3(0, 0, 1), 2000, 90, vec3(0, 0, -1), 16 / 9, 2, 25)