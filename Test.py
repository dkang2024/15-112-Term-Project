import taichi as ti 
import numpy as np 


print(bin(np.uint32(415)))

ti.init(ti.cpu)

def countLeadingZeros(num):
    return bin(num)[2:].find('1')

@ti.kernel 
def test():
    number = ti.uint32(412)
    print(bin(number))
