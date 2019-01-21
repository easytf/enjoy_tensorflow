import numpy as np
import random

def sigmoid(x):
	return 1/(1+np.exp(-x))
def derivative_sigmoid(x):
    print('sigmoid(x)', sigmoid(x))
    print('1-sigmoid(x)', 1 - sigmoid(x))
    return np.multiply(sigmoid(x),(1-sigmoid(x)))

#initialization
X=np.matrix("2,4,-2")
print(X)

W=np.random.normal(size=(3,2))
print('W:',W)
#label
ycap=[0]
#number of training of examples
num_examples=1
#step size
h=.01
#forward-propogation
y=np.dot(X,W)
print('y:',y)
y_o=sigmoid(y)
print('y_o:',y_o)
print('ycap:', ycap)
print('range(num_examples):',range(num_examples))
print('y_o[ range(num_examples) ,ycap ]->y_o[',range(num_examples), ycap,']:',y_o[ range(num_examples) ,ycap ])
#loss calculation
loss=-np.sum( np.log( y_o[ range(num_examples) ,ycap ] ) )
print(loss)     #outputs 7.87 (for you it would be different due to random initialization of weights.)

#backprop starts
temp1=np.copy(y_o)
#implementation of derivative of cost function with respect to y_o
temp1[ range(num_examples) ,ycap ] = 1/-( temp1[range(num_examples),ycap] )
temp=np.zeros_like(y_o)

temp[range(num_examples),ycap]=1
#derivative of cost with respect to y_o
print('temp:',temp)
print('temp1:',temp1)
dcost=np.multiply(temp,temp1)
print('dcost:', dcost)
#derivative of y_o with respect to y
dy_o = derivative_sigmoid(y)
print('derivative_sigmoid:',dy_o)
#element-wise multiplication
dgrad=np.multiply(dcost,dy_o)
print('dgrad:',dgrad)
print('X.T:',X.T)
dw=np.dot(X.T, dgrad)
print('dw:',dw)
#weight-update
W-=h*dw
#forward prop again with updated weight to find new loss
y=np.dot(X,W)
yo=sigmoid(y)
loss=-np.sum(np.log(yo[range(num_examples),ycap]))
print(loss)     #outpus 7.63 (again for you it would be different!)
