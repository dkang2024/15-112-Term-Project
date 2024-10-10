from Utils import *

def renderScene(cameraPos: vec3, imageWidth: int, viewportWidth: float, focalLength: float, aspectRatio: float, tMin = 0.001, tMax = 1e10): #type: ignore
    camera = Camera(cameraPos, imageWidth, viewportWidth, focalLength, aspectRatio, tMin, tMax)
    camera.addHittable(sphere3(vec3(0, 0, -2), 0.5))
    camera.addHittable(sphere3(vec3(0, -100.5, -1), 100))

    gui = ti.ui.Window('Render Test', res = (camera.imageWidth, camera.imageHeight))
    canvas = gui.get_canvas()

    while gui.running: 
        camera.render()
        canvas.set_image(camera.pixelField)
        gui.show()
        
renderScene(vec3(0, 0, 0), 800, 2, 1, 16 / 9)