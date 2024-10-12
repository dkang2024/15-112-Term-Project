import taichi as ti 
ti.init(ti.cpu)

@ti.data_oriented
class TiArray:
    def __init__(self, n):
        self.x = ti.field(dtype=ti.i32, shape=n)

    @ti.kernel
    def inc(self):
        for i in self.x:
            self.x[i] += 1
    
    @ti.kernel 
    def printX(self):
        for i in self.x:
            print(self.x[i])

a = TiArray(32)
a.inc()
print(a.x)
a.printX()