import os 
os.environ["JAX_PLATFORMS"] = 'cpu'

import jax.numpy as jnp 
import jax 

@jax.jit
def addVectors(vector1, vector2):
    return vector1 + vector2 

@jax.jit 
def vectorDotProduct(vector1, vector2):
    return jnp.dot(vector1, vector2)

@jax.jit
def vectorCrossProduct(vector1, vector2):
    return jnp.cross(vector1, vector2)

@jax.jit 
def distanceBetweenVectors(vector1, vector2):
    return jnp.sqrt(jnp.sum(jnp.square(vector2) - jnp.square(vector1)))

class vector3D():
    def __init__(self, x, y, z, dataType = jnp.float32):
        self.vector = jnp.array([x, y, z], dtype=dataType)

    def __add__(self, vector2):
        return addVectors(self.vector, vector2.vector)
    
    def __sub__(self, vector2):
        return addVectors(self.vector, -vector2.vector)
    
    def __mul__(self, vector2):
        return vectorDotProduct(self.vector, vector2.vector)
    
    def __matmul__(self, vector2):
        return vectorCrossProduct(self.vector, vector2.vector)
    
    def __neg__(self):
        newVec = -self.vector
        return vector3D(*newVec)
    
    def add(self, vector2):
        self.vector = addVectors(self.vector, vector2.vector)

    def subtract(self, vector2):
        self.vector = addVectors(self.vector, -vector2.vector)
    
    def dotProduct(self, vector2):
        self.vector = vectorDotProduct(self.vector, vector2.vector)
    
    def crossProduct(self, vector2):
        self.vector = vectorCrossProduct(self.vector, vector2.vector)

    def magnitude(self):
        return jnp.sqrt(jnp.dot(self.vector, self.vector))
    
    def mutatingNormalize(self):
        self.vector = self.vector / self.magnitude()

    def nonmutatingNormalize(self):
        return vector3D(*(self.vector / self.magnitude()))

def testFunctions():
    vector1 = vector3D(0, 5, 2)
    vector2 = vector3D(5, 2, 3)
    vector1 = -vector1
    print(vector1.vector)
    print(distanceBetweenVectors(vector1.vector, vector2.vector))

if __name__ == '__main__':
    testFunctions()
