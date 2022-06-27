import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import quad
import math as m
from scipy.interpolate import interp1d
import warnings


def dSdx(S,x,g):
    y1,y2=S
    return [-y2*x**(-2),x**(2)*y1**(1/(g-1))]

y1_0=1.
y2_0=0.
S_0=(y1_0,y2_0)
gamma=np.linspace(1.25,1.7,90)
x=np.linspace(10**(-5),10,3000)
x_R=np.zeros(90)
counter=0
sol=[]
for g in gamma:
    sol_temp=odeint(dSdx,S_0,x,args=(g,))
    if np.isnan(sol_temp[:,0]).any():
        stopper = np.where(np.isnan(sol_temp[:,0]))[0][0]
        for i in range(stopper,-1):
            sol_temp[i,0]=0
    sol.append(sol_temp[:, 0])
    if np.isnan(sol_temp[:,0]).any():
        x_R[counter]=x[stopper-1]
    else:
        x_R[counter]=x[-1]
    counter+=1

warnings.filterwarnings('ignore')

k_min=m.pi/x_R

S=[]
for i in range(90):
    k=np.linspace(k_min[i],10,3000)
    f=interp1d(x,sol[i])
    h=[]
    for j in range(3000):
        integrand=lambda x: (f(x))**(1/(gamma[i]-1))*m.sin(k[j]*x)*x
        integral=quad(integrand,10**(-8),x_R[i])
        h.append((integral[0]/k[j])**2)
    f_=h/h[0]
    n=1/(gamma[i]-1)
    cg=(gamma[i]*n)**(-3/2)
    g=interp1d(k,f_)
    integrand=lambda k: g(k)*m.log10(g(k))*k**(2)
    integral=quad(integrand,k[0],k[-1])
    S.append(integral[0]*cg)
S_final = [i *(-4)*m.pi*2.2 for i in S]

M=[]
for i in range(90):
    f=interp1d(x,sol[i])
    h=[]
    for j in range(3000):
        integrand=lambda x: (f(x))**(1/(gamma[i]-1))*x**(2)
        integral=quad(integrand,10**(-8),x_R[i])
        h.append(integral[0])
        n=1/(gamma[i]-1)
    cg=(gamma[i]*n)**(3/2)
    M.append(integral[0]*cg)
M_final = [i *(4)*m.pi/204.8 for i in M]

plt.plot(gamma,S_final, label='$Sp_{0}^{-1}$')
plt.plot(gamma,M_final, label='$M$')
plt.legend(loc='lower right')
plt.xlabel('$\gamma$')
plt.grid()
plt.show()