import taichi as ti 
import warnings 
from typing import Union
warnings.filterwarnings("ignore") #Taichi throws warnings because list methods are used (and Taichi doesn't handle these but Python does). We want to ignore these warnings (the classes are specifically designed to allow taichi to work)

ti.init(ti.cpu)

@ti.dataclass 
class Mat1:
    value: int 

@ti.dataclass 
class Mat2:
    value: float 

@ti.dataclass 
class sphere:
    material1: Mat1 
    material2: Mat2

newSphere = sphere(Mat2(1))