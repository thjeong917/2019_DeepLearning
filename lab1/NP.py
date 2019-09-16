import numpy as np
import random
import math
dataset = []

def sigmoid(g):
    if g<0:
        return 1-(1/(1 + math.exp(g)))
    else:
        return 1/(1 + math.exp(-g))

for i in range(0, 128):
    u = np.random.uniform(-1,1)
    v = np.random.uniform(-1,1)
    if u*u<v:
        z=0
    else:
        z=1
        
    dataset.append(((u,v),z))
    print(u,v,z)

J=0

W1=np.ones((2,2))
b1=np.ones((2,1))
W2=np.ones((1,2))
b2=np.ones((1,1))

# W1=np.array([[np.random.uniform(0,1),np.random.uniform(0,1)],[np.random.uniform(0,1),np.random.uniform(0,1)]])
# b1=np.array([[np.random.uniform(0,1)],[np.random.uniform(0,1)]])
# W2=np.array([[np.random.uniform(0,1),np.random.uniform(0,1)]])
# b2=np.array([[np.random.uniform(0,1)]])

dW1=np.zeros((2,2))
db1=np.zeros((2,1))
dW2=np.zeros((1,2))
db2=np.zeros((1,1))

alpha=0.01
tCase=128.0
rTotal=0

for j in range(0,10):
    for i in range(0,5000):
        J=0
        for cur in range(0,128):
            x=dataset[cur][0][0]
            y=dataset[cur][0][1]
            cls=dataset[cur][1]

            X1=np.array([[x],[y]])
            Z1=W1@X1 + b1

            x1=sigmoid(Z1[0][0])
            y1=sigmoid(Z1[1][0])

            X2=np.array([[x1],[y1]])
            Z2=W2@X2 + b2

            a=sigmoid(Z2[0])

            if a<=0:
                J+=1
            elif a==1:
                J=J
            else:
                J += -(cls*math.log(a) + (1-cls)*math.log(1-a))/tCase

            dz=a-cls
            dW2+=(dz/tCase)*(X2.transpose())
            db2+=dz/tCase

            W2-=(alpha*dW2)
            b2-=(alpha*db2)

            dW1 += X1@(dz*W2)/tCase
            db1 += dz*(W2.transpose())/tCase

            W1-=alpha*dW1
            b1-=alpha*db1



    total=0
    for cur in range(0,128):
        x=dataset[cur][0][0]
        y=dataset[cur][0][1]
        cls=dataset[cur][1]

        X1=np.array(([x],[y]))
        Z1=W1@X1 + b1

        x1=sigmoid(Z1[0][0])
        y1=sigmoid(Z1[1][0])

        X2=np.array(([x1],[y1]))
        Z2=W2@X2 + b2

        a=sigmoid(Z2[0])

        if abs(1 - Z2[0]) > abs(0 - Z2[0]):
            t = 0
        else:
            t = 1
        if t == cls:
            total += 1

    print(W1)
    print(W2)
    accuracy=float(total)/float(128) * 100
    print("accuracy :", accuracy)
    rTotal+=accuracy
    
print("Mean accuracy :", rTotal/10)