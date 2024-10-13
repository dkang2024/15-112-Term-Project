import taichi as ti 

ti.init(ti.cpu)

@ti.dataclass 
class TestTaichi:
    radius: float 

    def incrementRadius(self):
        self.radius += 1

    def getRadius(self):
        return self.radius

test = TestTaichi(5)
test.incrementRadius()
print(test.radius, test.getRadius)