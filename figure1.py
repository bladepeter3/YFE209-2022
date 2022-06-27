import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import quad
import math as m
from scipy.interpolate import interp1d
import warnings


"""
def f(x,y,z):
    return z*x**(-2) 
def g(x,y,z,n):
    return -y**(n)*x**(2)


    
x_0=0.1;y_0=1.;z_0=0.; t_step=0.1
n=1/(1.2-1)
x=np.array([(float)(x_0)])
y=np.array([(float)(y_0)])
z=np.array([(float)(z_0)])
N=(int)((10./t_step)-1.)
for i in range(1,N+1):
    k_0 = t_step * f(x[i-1], y[i-1], z[i-1])
    l_0 = t_step * g(x[i-1], y[i-1], z[i-1], n)
    k_1 = t_step * f(x[i-1]+1/2*t_step, y[i-1]+1/2*k_0, z[i-1]+1/2*l_0)
    l_1 = t_step * g(x[i-1]+1/2*t_step, y[i-1]+1/2*k_0, z[i-1]+1/2*l_0, n)
    k_2 = t_step * f(x[i-1]+1/2*t_step, y[i-1]+1/2*k_1, z[i-1]+1/2*l_1)
    l_2 = t_step * g(x[i-1]+1/2*t_step, y[i-1]+1/2*k_1, z[i-1]+1/2*l_1, n)
    k_3 = t_step * f(x[i-1]+t_step, y[i-1]+k_2, z[i-1]+l_2)
    l_3 = t_step * g(x[i-1]+t_step, y[i-1]+k_2, z[i-1]+l_2, n)
    x=np.append(x,x[i-1] + t_step)
    y=np.append(y,y[i-1] + (1.0 / 6.0)*(k_0 + 2 * k_1 + 2 * k_2 + k_3))
    z=np.append(z,z[i-1] + (1.0 / 6.0)*(l_0 + 2 * l_1 + 2 * l_2 + l_3))
plt.figure(figsize = (12, 8))

plt.plot(x, y, label='γ=1.2')
    
x=np.array([(float)(x_0)])
y=np.array([(float)(y_0)])
z=np.array([(float)(z_0)])
n=1/(1.3-1)
for i in range(1,N+1):
    k_0 = t_step * f(x[i-1], y[i-1], z[i-1])
    l_0 = t_step * g(x[i-1], y[i-1], z[i-1], n)
    k_1 = t_step * f(x[i-1]+1/2*t_step, y[i-1]+1/2*k_0, z[i-1]+1/2*l_0)
    l_1 = t_step * g(x[i-1]+1/2*t_step, y[i-1]+1/2*k_0, z[i-1]+1/2*l_0, n)
    k_2 = t_step * f(x[i-1]+1/2*t_step, y[i-1]+1/2*k_1, z[i-1]+1/2*l_1)
    l_2 = t_step * g(x[i-1]+1/2*t_step, y[i-1]+1/2*k_1, z[i-1]+1/2*l_1, n)
    k_3 = t_step * f(x[i-1]+t_step, y[i-1]+k_2, z[i-1]+l_2)
    l_3 = t_step * g(x[i-1]+t_step, y[i-1]+k_2, z[i-1]+l_2, n)
    x=np.append(x,x[i-1] + t_step)
    y=np.append(y,y[i-1] + (1.0 / 6.0)*(k_0 + 2 * k_1 + 2 * k_2 + k_3))
    z=np.append(z,z[i-1] + (1.0 / 6.0)*(l_0 + 2 * l_1 + 2 * l_2 + l_3))

plt.plot(x, y, label='γ=1.3')

x=np.array([(float)(x_0)])
y=np.array([(float)(y_0)])
z=np.array([(float)(z_0)])
n=1/(1.4-1)
for i in range(1,N+1):
    k_0 = t_step * f(x[i-1], y[i-1], z[i-1])
    l_0 = t_step * g(x[i-1], y[i-1], z[i-1], n)
    k_1 = t_step * f(x[i-1]+1/2*t_step, y[i-1]+1/2*k_0, z[i-1]+1/2*l_0)
    l_1 = t_step * g(x[i-1]+1/2*t_step, y[i-1]+1/2*k_0, z[i-1]+1/2*l_0, n)
    k_2 = t_step * f(x[i-1]+1/2*t_step, y[i-1]+1/2*k_1, z[i-1]+1/2*l_1)
    l_2 = t_step * g(x[i-1]+1/2*t_step, y[i-1]+1/2*k_1, z[i-1]+1/2*l_1, n)
    k_3 = t_step * f(x[i-1]+t_step, y[i-1]+k_2, z[i-1]+l_2)
    l_3 = t_step * g(x[i-1]+t_step, y[i-1]+k_2, z[i-1]+l_2, n)
    x=np.append(x,x[i-1] + t_step)
    y=np.append(y,y[i-1] + (1.0 / 6.0)*(k_0 + 2 * k_1 + 2 * k_2 + k_3))
    z=np.append(z,z[i-1] + (1.0 / 6.0)*(l_0 + 2 * l_1 + 2 * l_2 + l_3))

plt.plot(x, y, label='γ=1.4')

x=np.array([(float)(x_0)])
y=np.array([(float)(y_0)])
z=np.array([(float)(z_0)])
n=1/(1.5-1)
for i in range(1,N+1):
    k_0 = t_step * f(x[i-1], y[i-1], z[i-1])
    l_0 = t_step * g(x[i-1], y[i-1], z[i-1], n)
    k_1 = t_step * f(x[i-1]+1/2*t_step, y[i-1]+1/2*k_0, z[i-1]+1/2*l_0)
    l_1 = t_step * g(x[i-1]+1/2*t_step, y[i-1]+1/2*k_0, z[i-1]+1/2*l_0, n)
    k_2 = t_step * f(x[i-1]+1/2*t_step, y[i-1]+1/2*k_1, z[i-1]+1/2*l_1)
    l_2 = t_step * g(x[i-1]+1/2*t_step, y[i-1]+1/2*k_1, z[i-1]+1/2*l_1, n)
    k_3 = t_step * f(x[i-1]+t_step, y[i-1]+k_2, z[i-1]+l_2)
    l_3 = t_step * g(x[i-1]+t_step, y[i-1]+k_2, z[i-1]+l_2, n)
    x=np.append(x,x[i-1] + t_step)
    y=np.append(y,y[i-1] + (1.0 / 6.0)*(k_0 + 2 * k_1 + 2 * k_2 + k_3))
    z=np.append(z,z[i-1] + (1.0 / 6.0)*(l_0 + 2 * l_1 + 2 * l_2 + l_3))

plt.plot(x, y, label='γ=1.5')

x=np.array([(float)(x_0)])
y=np.array([(float)(y_0)])
z=np.array([(float)(z_0)])
n=1/(1.6-1)
for i in range(1,N+1):
    k_0 = t_step * f(x[i-1], y[i-1], z[i-1])
    l_0 = t_step * g(x[i-1], y[i-1], z[i-1], n)
    k_1 = t_step * f(x[i-1]+1/2*t_step, y[i-1]+1/2*k_0, z[i-1]+1/2*l_0)
    l_1 = t_step * g(x[i-1]+1/2*t_step, y[i-1]+1/2*k_0, z[i-1]+1/2*l_0, n)
    k_2 = t_step * f(x[i-1]+1/2*t_step, y[i-1]+1/2*k_1, z[i-1]+1/2*l_1)
    l_2 = t_step * g(x[i-1]+1/2*t_step, y[i-1]+1/2*k_1, z[i-1]+1/2*l_1, n)
    k_3 = t_step * f(x[i-1]+t_step, y[i-1]+k_2, z[i-1]+l_2)
    l_3 = t_step * g(x[i-1]+t_step, y[i-1]+k_2, z[i-1]+l_2, n)
    x=np.append(x,x[i-1] + t_step)
    y=np.append(y,y[i-1] + (1.0 / 6.0)*(k_0 + 2 * k_1 + 2 * k_2 + k_3))
    z=np.append(z,z[i-1] + (1.0 / 6.0)*(l_0 + 2 * l_1 + 2 * l_2 + l_3))

plt.plot(x, y, label='γ=1.6')
 
x=np.array([(float)(x_0)])
y=np.array([(float)(y_0)])
z=np.array([(float)(z_0)])
n=1/(1.7-1)
for i in range(1,N+1):
    k_0 = t_step * f(x[i-1], y[i-1], z[i-1])
    l_0 = t_step * g(x[i-1], y[i-1], z[i-1], n)
    k_1 = t_step * f(x[i-1]+1/2*t_step, y[i-1]+1/2*k_0, z[i-1]+1/2*l_0)
    l_1 = t_step * g(x[i-1]+1/2*t_step, y[i-1]+1/2*k_0, z[i-1]+1/2*l_0, n)
    k_2 = t_step * f(x[i-1]+1/2*t_step, y[i-1]+1/2*k_1, z[i-1]+1/2*l_1)
    l_2 = t_step * g(x[i-1]+1/2*t_step, y[i-1]+1/2*k_1, z[i-1]+1/2*l_1, n)
    k_3 = t_step * f(x[i-1]+t_step, y[i-1]+k_2, z[i-1]+l_2)
    l_3 = t_step * g(x[i-1]+t_step, y[i-1]+k_2, z[i-1]+l_2, n)
    x=np.append(x,x[i-1] + t_step)
    y=np.append(y,y[i-1] + (1.0 / 6.0)*(k_0 + 2 * k_1 + 2 * k_2 + k_3))
    z=np.append(z,z[i-1] + (1.0 / 6.0)*(l_0 + 2 * l_1 + 2 * l_2 + l_3))

plt.plot(x, y, label='γ=1.7')
    
x=np.array([(float)(x_0)])
y=np.array([(float)(y_0)])
z=np.array([(float)(z_0)])
n=1/(1.8-1)
for i in range(1,N+1):
    k_0 = t_step * f(x[i-1], y[i-1], z[i-1])
    l_0 = t_step * g(x[i-1], y[i-1], z[i-1], n)
    k_1 = t_step * f(x[i-1]+1/2*t_step, y[i-1]+1/2*k_0, z[i-1]+1/2*l_0)
    l_1 = t_step * g(x[i-1]+1/2*t_step, y[i-1]+1/2*k_0, z[i-1]+1/2*l_0, n)
    k_2 = t_step * f(x[i-1]+1/2*t_step, y[i-1]+1/2*k_1, z[i-1]+1/2*l_1)
    l_2 = t_step * g(x[i-1]+1/2*t_step, y[i-1]+1/2*k_1, z[i-1]+1/2*l_1, n)
    k_3 = t_step * f(x[i-1]+t_step, y[i-1]+k_2, z[i-1]+l_2)
    l_3 = t_step * g(x[i-1]+t_step, y[i-1]+k_2, z[i-1]+l_2, n)
    x=np.append(x,x[i-1] + t_step)
    y=np.append(y,y[i-1] + (1.0 / 6.0)*(k_0 + 2 * k_1 + 2 * k_2 + k_3))
    z=np.append(z,z[i-1] + (1.0 / 6.0)*(l_0 + 2 * l_1 + 2 * l_2 + l_3))

plt.plot(x, y, label='γ=1.8')

x=np.array([(float)(x_0)])
y=np.array([(float)(y_0)])
z=np.array([(float)(z_0)])
n=1/(1.9-1)
for i in range(1,N+1):
    k_0 = t_step * f(x[i-1], y[i-1], z[i-1])
    l_0 = t_step * g(x[i-1], y[i-1], z[i-1], n)
    k_1 = t_step * f(x[i-1]+1/2*t_step, y[i-1]+1/2*k_0, z[i-1]+1/2*l_0)
    l_1 = t_step * g(x[i-1]+1/2*t_step, y[i-1]+1/2*k_0, z[i-1]+1/2*l_0, n)
    k_2 = t_step * f(x[i-1]+1/2*t_step, y[i-1]+1/2*k_1, z[i-1]+1/2*l_1)
    l_2 = t_step * g(x[i-1]+1/2*t_step, y[i-1]+1/2*k_1, z[i-1]+1/2*l_1, n)
    k_3 = t_step * f(x[i-1]+t_step, y[i-1]+k_2, z[i-1]+l_2)
    l_3 = t_step * g(x[i-1]+t_step, y[i-1]+k_2, z[i-1]+l_2, n)
    x=np.append(x,x[i-1] + t_step)
    y=np.append(y,y[i-1] + (1.0 / 6.0)*(k_0 + 2 * k_1 + 2 * k_2 + k_3))
    z=np.append(z,z[i-1] + (1.0 / 6.0)*(l_0 + 2 * l_1 + 2 * l_2 + l_3))

plt.plot(x, y, label='γ=1.9')
    
plt.title('Lane-Emden Solutions')
plt.xlabel('ξ')
plt.ylabel('θ(ξ)')
plt.grid()
plt.legend(loc='upper right')
#plt.ylim([0,1])
plt.show()
"""


def dSdx(S,x,g):
    y1,y2=S
    return [-y2*x**(-2),x**(2)*y1**(1/(g-1))]

y1_0=1.
y2_0=0.
S_0=(y1_0,y2_0)
gamma=[1.2,1.4,1.7]
x=np.linspace(10**(-5),10,3000)
x_R=np.zeros(3)
counter=0
sol=[]
for g in gamma:    
    sol_temp=odeint(dSdx,S_0,x,args=(g,))
    if np.isnan(sol_temp[:,0]).any():
        stopper = np.where(np.isnan(sol_temp[:,0]))[0][0]
        for i in range(stopper,-1):
            sol_temp[i,0]=0
    sol.append(sol_temp[:, 0])
    if counter==0:
        x_R[counter]=x[-1]     
    else:
        x_R[counter]=x[stopper-1]
    counter+=1
plt.plot(x, sol[0], color='r', label='γ=1.2')
plt.plot(x, sol[1], color='b',label='γ=1.4')
plt.plot(x, sol[2], color='g',label='γ=1.7')
plt.title('Lane-Emden Solutions')
plt.xlabel('ξ')
plt.ylabel('θ(ξ)')
plt.grid()
plt.legend(loc='upper right')
plt.show()

warnings.filterwarnings('ignore')

k_min=m.pi/x_R   
k1=np.linspace(k_min[0],10,3000)
k2=np.linspace(k_min[1],10,3000)
k3=np.linspace(k_min[2],10,3000)

f1=interp1d(x,sol[0])
f2=interp1d(x,sol[1])
f3=interp1d(x,sol[2])

h1=[];
h2=[];
h3=[];

for i in range(3000):
    integrand1=lambda x: (f1(x))**(1/(gamma[0]-1))*m.sin(k1[i]*x)*x
    integrand2=lambda x: (f2(x))**(1/(gamma[1]-1))*m.sin(k2[i]*x)*x
    integrand3=lambda x: (f3(x))**(1/(gamma[2]-1))*m.sin(k3[i]*x)*x

    integral1=quad(integrand1,10**(-8),x_R[0])
    integral2=quad(integrand2,10**(-8),x_R[1])
    integral3=quad(integrand3,10**(-8),x_R[2])

    h1.append((integral1[0]/k1[i])**2)
    h2.append((integral2[0]/k2[i])**2)
    h3.append((integral3[0]/k3[i])**2)

f_1=h1/h1[0]
f_2=h2/h2[0]
f_3=h3/h3[0]

#*************************
n1=1/(gamma[0]-1)
n2=1/(gamma[1]-1)
n3=1/(gamma[2]-1)
cg1=np.sqrt(gamma[0]*n1)#|
cg2=np.sqrt(gamma[1]*n2)#|
cg3=np.sqrt(gamma[2]*n3)#|
#*************************

plt.plot(k1/cg1, f_1, color='r', label='γ=1.2')
plt.plot(k2/cg2, f_2, color='b',label='γ=1.4')
plt.plot(k3/cg3, f_3, color='g',label='γ=1.7')
plt.legend(loc='upper right')
plt.grid()
plt.xlim([0,1.5])
plt.ylabel('$\~f (|k|)$')
plt.xlabel('$k/\sqrt{4 \pi G/K}$')
plt.show()





