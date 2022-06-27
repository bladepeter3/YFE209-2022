import math as m
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d 

def diaforiki(xi,sinthikes):
    y=sinthikes[0]
    dy=sinthikes[1]
    return [dy,-2/xi*dy-y**(1/(g-1))]
def S(k):
    hak = lambda xi : np.sin(k*xi)*xi*th(xi)**(1/(g-1))
    hakmin = lambda xi : np.sin(ak_min*xi)*xi*th(xi)**(1/(g-1))
    
    Integral_min=quad(hakmin,10**(-5),xi_R)
    Integral_min=Integral_min[0]**2/ak_min**2
    
    Integral=quad(hak,10**(-5),xi_R)
    Integral=Integral[0]**2/k**2
   
    fak=Integral/Integral_min
    return fak*np.log(fak)*k**2
    
x1=np.linspace(10**(-5),20,3000)
sinthikes=[1,0]

Ss=[[],[],[]]
gamma=np.arange(1.25,1.75,0.005)
for g in gamma:
    print('i='+str(g)+'\n')
    w=solve_ivp(fun=diaforiki,t_span=[x1[0],x1[-1]],y0=sinthikes,t_eval=x1, method='RK45')
    w=w.y[0]

    stopper=len(w)
    for i in range(len(w)):
        if m.isnan(w[i]):
            stopper=i
            break

    x2=x1[0:stopper]
    w=w[0:stopper]

    th=interp1d(x2,w)

    xi_R=x2[-1]
    ak_min=np.pi/xi_R

    mass=lambda xi: th(xi)**(1/(g-1))*xi**2

    The_3ak=[ak_min/0.95,ak_min,ak_min/1.05]
    for i in range(len(The_3ak)):
        ak_min=The_3ak[i]

        Int_S=quad(S,ak_min,10)
        Int_S=Int_S[0]
        Ss[i].append(-Int_S*4*m.pi)
        
plt.plot(gamma,Ss[0],label='π/(0.95R)')
plt.plot(gamma,Ss[1],label='π/(1.00R)')
plt.plot(gamma,Ss[2],label='π/(1.05R)')
plt.grid()
plt.axvline(x=4/3,ls='--')
plt.axvline(x=5/3,ls='--')
plt.legend(loc='upper right')
plt.ylabel('$S\alpha^{3}$')
plt.xlabel('$\gamma$')
plt.xlim([1.3,1.7])
plt.show()

