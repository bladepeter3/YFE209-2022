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

z=[0.95,1.,1.05]

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

S=[]
S_temp=[]
S_final=[]
for o in z:
    k_min=m.pi/(x_R*o)
    S=[]
    S_temp=[]
    for i in range(90):
        k=np.linspace(k_min[i],10,3000)
        f=interp1d(x,sol[i])
        h=[]
        for j in range(3000):
            integrand=lambda x: (f(x))**(1/(gamma[i]-1))*m.sin(k[j]*x)*x
            integral=quad(integrand,10**(-8),x_R[i])
            h.append((integral[0]/k[j])**2)
        f_=h/h[0]
        g=interp1d(k,f_)
        integrand=lambda k: g(k)*m.log10(g(k))*k**(2)
        integral=quad(integrand,k[0],k[-1])
        S_temp.append(integral[0])
        print('i='+str(i)+'\n')
    S = [i *(-4)*m.pi*2.2+0.2 for i in S_temp]
    S_final.append(S)


plt.plot(gamma,S_final[0],label='π/(0.95R)')
plt.plot(gamma,S_final[1],label='π/(1.00R)')
plt.plot(gamma,S_final[2],label='π/(1.05R)')
plt.grid()
plt.axvline(x=4/3,ls='--')
plt.axvline(x=5/3,ls='--')
plt.legend(loc='upper right')
plt.ylabel('$S\alpha^{3}$')
plt.xlabel('$\gamma$')
plt.xlim([1.3,1.7])
plt.show()