import taichi as ti 

ti.init(ti.cpu)

@ti.dataclass 
class sphere: 
    radius: float 

    def add(self, addValue):
        self.property = addValue

@ti.data_oriented 
class newThing: 
    def __init__(self, value):
        self.value = value 

testThing = newThing(5.0)
testSphere = sphere(2.0)
testSphere.add(testThing)
print(testSphere.property)