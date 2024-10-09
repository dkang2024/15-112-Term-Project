from Utils import *

def renderScene(cameraPos: vec3, imageWidth: int, viewportWidth: float, focalLength: float, aspectRatio: float): #type: ignore
    camera = Camera(cameraPos, imageWidth, viewportWidth, focalLength, aspectRatio)
    gui = ti.ui.Window('Render Test', res = (camera.imageWidth, camera.imageHeight))
    canvas = gui.get_canvas()

    while gui.running: 
        camera.render()
        canvas.set_image(camera.pixelField)
        gui.show()

renderScene(vec3(0, 0, 0), 800, 2, 1, 16 / 9)