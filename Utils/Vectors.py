import os 
os.environ["JAX_PLATFORMS"] = 'cpu'

import jax.numpy as jnp 
import jax 

@jax.jit
def addVectors(vector1: jax.Array, vector2: jax.Array) -> jax.Array: 
    return vector1 + vector2 

@jax.jit 
def vectorDotProduct(vector1: jax.Array, vector2: jax.Array) -> jax.Array:
    return jnp.dot(vector1, vector2)

@jax.jit
def vectorCrossProduct(vector1: jax.Array, vector2: jax.Array) -> jax.Array:
    return jnp.cross(vector1, vector2)

@jax.jit 
def distanceBetweenVectors(vector1: jax.Array, vector2: jax.Array) -> float:
    return jnp.sqrt(jnp.sum(jnp.square(vector2) - jnp.square(vector1)))

class vector3D():
    __slots__ = ('vector')

    def __init__(self, x: float, y: float, z: float, dataType = jnp.float32):
        self.vector = jnp.array([x, y, z], dtype=dataType)

    def __add__(self, vector2):
        return vector3D(*addVectors(self.vector, vector2.vector))
    
    def __sub__(self, vector2):
        return vector3D(*addVectors(self.vector, -vector2.vector))
    
    def __mul__(self, vector2):
        return vector3D(*vectorDotProduct(self.vector, vector2.vector))
    
    def __matmul__(self, vector2):
        return vector3D(*vectorCrossProduct(self.vector, vector2.vector))
    
    def __neg__(self):
        return vector3D(*-self.vector)
    
    def __repr__(self):
        return f'x: {self.vector[0]}, y: {self.vector[1]}, z: {self.vector[2]}'
    
    def __eq__(self, vector2):
        return jnp.array_equal(self.vector, vector2.vector)

    def add(self, vector2):
        self.vector = addVectors(self.vector, vector2.vector)

    def subtract(self, vector2):
        self.vector = addVectors(self.vector, -vector2.vector)

    def scalarMultiply(self, scalar, mutating = False):
        if not mutating:
            return vector3D(*(self.vector * scalar))
        self.vector = self.vector * scalar
    
    def dotProduct(self, vector2):
        self.vector = vectorDotProduct(self.vector, vector2.vector)
    
    def crossProduct(self, vector2):
        self.vector = vectorCrossProduct(self.vector, vector2.vector)

    def magnitude(self):
        return jnp.sqrt(jnp.dot(self.vector, self.vector))
    
    def normalize(self, mutating = False):
        if not mutating: 
            return vector3D(*(self.vector / self.magnitude()))
        self.vector = self.vector / self.magnitude()