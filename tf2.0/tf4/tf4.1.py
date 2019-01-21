
import tensorflow as tf
import numpy as np
import random

def product(x, y): 
    h = 0.01
    z = ( x + random.random()*h )*( y + random.random()*h )
    return z

a = np.array([3]) 
b = np.array([4]) 
print(product(a, b))

def product2(x, y): 
    h = 0.01
    z = ( x + h*y )*( y + h*x )

    return z

a = np.array([-3]) 
b = np.array([-4]) 
print(product2(a, b))
