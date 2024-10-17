from Vectors import *

@ti.func
def leftShift(x): #type: ignore
    '''
    Performs the bit operations in order to shift the bottom 10 bits of a 32 bit integer to allow 2 bits in between each for creating a Morton code. NOTE THAT I SHOULD GET NO CREDIT FOR THIS: https://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies. The website gave me the algorithm and I just coded it up in python with hexadecimal rather than having the full binary representation. I would have done the shift using the naive method of shifting each bit otherwise.
    '''
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x <<  8)) & 0x0300F00F
    x = (x | (x <<  4)) & 0x030C30C3
    x = (x | (x <<  2)) & 0x09249249
    return x

@ti.func 
def scaleToInt(value):
    '''
    Scale the value to a full 10 bit integer if it's in the range [0, 1] by multiplying it by 1024 (the maximum number of bits that an unsigned 10 bit integer can hold [because the range for unsigned 10 bit integers is going to be [0, 1023] inclusive bounds] [although note that I'm using signed 32 bit integers just to hold these 10 bits because it's more convenient]) 
    '''
    return int(ti.min(value * 1024, 1023)) 

@ti.func 
def mortonEncode(boundingBoxCentroid: vec3): #type: ignore 
    '''
    Morton encode a 3D vector with floating point numbers ranging from 0 to 1 to represent relative position of the bounding box's centroid. 
    '''
    x, y, z = scaleToInt(boundingBoxCentroid.x), scaleToInt(boundingBoxCentroid.y), scaleToInt(boundingBoxCentroid.z)
    return (z << 2) | (y << 1) | x