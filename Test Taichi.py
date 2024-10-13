import taichi as ti 
import taichi.math as tm 

ti.init(ti.cpu)
vec3 = tm.vec3

def vec3Field(): #type: ignore
    return ti.Vector.field(3, dtype = float, shape = ())

@ti.kernel
def vec3GetVector(field: ti.template()) -> vec3: #type: ignore
    return field[None] 

def vec3CreateField(vector: vec3): #type: ignore
    field = vec3Field()
    field.fill(vector)
    return field 

def vec3SetField(field, vector):
    field[None] = vector
    return field 

def floatField(): #type: ignore
    return ti.field(float, shape = ())

@ti.kernel
def floatGet(field: ti.template()) -> float: #type: ignore
    return field[None]

def floatCreateField(number: float): #type: ignore
    field = floatField()
    field.fill(number)
    return field

def floatSetField(field, number): 
    field[None] = number
    return field