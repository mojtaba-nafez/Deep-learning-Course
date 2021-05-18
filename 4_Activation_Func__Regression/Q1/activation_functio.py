import matplotlib.pyplot as plt
import numpy as np

def RELU(x):
    x1=[]
    x2=[]
    for i in x:
        if i<0:
            x1.append(0)
            x2.append(0)
        else:
            x1.append(i)
            x2.append(1)
    return x1, x2

def Leaky_RELU(x):
    x1=[]
    x2=[]
    for i in x:
        if i<0:
            x1.append(0.1*i)
            x2.append(-0.1)
        else:
            x1.append(i)
            x2.append(1)
    return x1, x2

def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

def ELU(x):
    x1=[]
    x2=[]
    for i in x:
        if i<0:
            x1.append(0.1*(np.exp(i)-1))
            x2.append(0.1*np.exp(i))
        else:
            x1.append(i)
            x2.append(1)
    return x1, x2

def Swish(x):
    x1 = x*sigmoid(x)
    x2 = sigmoid(x)+x*sigmoid(x)*(1-sigmoid(x))
    return x1, x2
'''
def Mish(x):
    x1 = x*np.tanh((np.log(1+ np.exp(x))/np.log(2.718281)))
    x2 = np.tanh((np.log(1+ np.exp(x))/np.log(2.718281))) + x*np.square(1/np.cosh(((np.log(1+ np.exp(x))/np.log(2.718281)))))*np.exp(x)/(np.log(1+ np.exp(x))/np.log(2.718281))
    return x1, x2
'''
def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
def softplus(x):
    return np.log(1 + np.exp(x))
def mish(x):
    return x * tanh(softplus(x))
def mish_derivative(x):
    omega = np.exp(3*x) + 4*np.exp(2*x) + (6+4*x)*np.exp(x) + 4*(1 + x)
    delta = 1 + pow((np.exp(x) + 1), 2)
    derivative = np.exp(x) * omega / pow(delta, 2)
    return derivative

x=np.arange(-6,6,0.01)
sigmoid(x)
fig, ax = plt.subplots(figsize=(9, 5))
plt.plot(x,RELU(x)[0], color="#FFFF00", linewidth=8, label="Relu")
plt.plot(x,RELU(x)[1], color="#FFFF00", linewidth=8, label="Relu derivative")
plt.plot(x,Leaky_RELU(x)[0], color="#FF0000", linewidth=5, label="Leaky Relu")
plt.plot(x,Leaky_RELU(x)[1], color="#FF0000", linewidth=5, label="Leaky Relu derivative")
plt.plot(x,ELU(x)[0], color="#000000", linewidth=1, label="Elu")
plt.plot(x,ELU(x)[1], color="#000000", linewidth=1, label="Elu derivative")

plt.plot(x,Swish(x)[0], color="#00FF00", linewidth=2, label="Swish")
plt.plot(x,Swish(x)[1], color="#00FF00", linewidth=2, label="Swish derivative")

#plt.plot(x,Mish(x)[0], color="#C0C0C0", linewidth=1, label="Mish")
#plt.plot(x,Mish(x)[1], color="#C0C0C0", linewidth=1, label="Mish derivative")
plt.plot(x,mish(x), color="#0000A0", linewidth=1, label="Mish")
plt.plot(x,mish_derivative(x), color="#0000A0", linewidth=1, label="Mish derivative")


plt.legend(loc="upper right", frameon=False)
plt.show()