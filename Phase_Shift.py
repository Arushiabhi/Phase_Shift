import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numpy import pi, abs, mean, sin, cos, exp, sqrt, cosh, log
import numba
import matplotlib.pyplot as plt

#M_n = 938.272046  # MeV / c ^ 2
M_alpha = 3728.7705  # MeV / c ^ 2
hbarc = 197.3269718  # MeV fm
mu = (M_alpha * M_alpha) / (M_alpha + M_alpha)
den = (hbarc ** 2) / ( mu)
den2 = (hbarc ** 2) / (2*mu)

# for l=0 state
#E = np.array([0.85,0.95,1,1.5,2,2.5,3,3.84,6.47,6.96,7.47,7.88,8.87,9.88,10.88,11.88,18,21.12,24.11,25.55,26.99,28.5]).astype(np.float)  # // np  scattering    data
#pS = np.array( [175,173,171,159,148,137.5,128.4,114.1,79.5,75.9,71.4,68,59.4,51.6,45.6,41,7.14,-4.96,-15.33,-19.64,-22.81,-27.03] ).astype(np.float)

# for l=2 state
#E = np.array([3.84,6.47,6.96,7.47,7.88,8.87,9.88,10.88,11.88,18,21.12,24.11,25.55,26.99,28.5]).astype(np.float)  # // np  scattering    data
#pS = np.array( [7.5,80.8,92.7,102.1,107.5,113.8,115.2,116.3,114.9,101.38,96.33,91.86,89.37,88.98,86.79] ).astype(np.float)

# for l=4 state
E = np.array([18,21.12,24.11,25.50,26.99,28.50]).astype(np.float)  # // np  scattering    data
pS = np.array([15.86,36.27,72.61,88.64,105.29,115.26]).astype(np.float)

abs_pS_inv = abs(pS ** - 1)
k = sqrt(E / den)
k_1 = k ** -1
den_1 = -den2 ** -1

h = 0.01
h_4 = h / 4
h_8 = h / 8
h_2 = h / 2
h_34 = h * 0.75
h_3_16 = h * 3 / 16
h_9_16 = h * 9 / 16
h_3_7 = h * 3 / 7
h_12_7 = h * 12 / 7
h_90 = h / 90
h_8_7 = h * 8 / 7
h_2_7 = h * 2 / 7
rad2deg = 180. / pi

def morse():
    
    # for l=0 state potential parameters
       # a1=56.0127817474538 # V1
       # a2=0.18366034332069 # alpha1
       # a3=4.46287055232512 # r1
       # a5=0.37358698372827 # alpha2
       # a6=11.9650862880996 # r2
       # a7=8.07611206030489E-006 # V2
       # a4=6.1725160927311 # X point
       
    # for l=2 state potential parameters
       # a1	=	7.05745947839409
       # a2	=	0.80187764689511
       # a3	=	1.56796490708802
       # a5	=	2.32297554096664
       # a6	=	4.60094769226312
       # a7	=	2.02966177198505E-009
       # a4	=	4.47228070582174
       
    # for l=4 state potential parameters
       a1 = 150.189669942879
       a2 =	0.353558931883
       a3 =	0.648438854906
       a5 =	0.820924841109
       a6 =	7.800143893319
       a7 =	0
       a4 =	2.639926908941
       
       x1 = np.arange(0.01, a4, 0.0025)
       y1 = np.arange(a4, 40, 0.0025)
         
       f2 = exp(-2*a5*(a4 - a6)) - (2*exp(-a5*(a4 - a6)))
   
       f1 = exp(-2*a2*(a4 - a3)) - (2*exp(-a2*(a4 - a3)))
   
       g2 = exp(-2 *a5 *(a4 - a6)) - (exp(-a5 *(a4 - a6)))
   
       g1 = exp(-2 *a2 *(a4 - a3)) - (exp(-a2 *(a4 - a3)))
   
       D2 = (a2 * g1 * (a1-a7))/(f1*a5*g2 - (f2*a2*g1))
       D1 = ((a5 * g2 * (a7-a1))/(f1*a5*g2 - (f2*a2*g1)))
   
       V_1 = a1 + D1 *(exp(-2*a2*(x1-a3)) - 2*exp(-a2*(x1-a3)))
       V_2 = a7 - D2*(exp(-2*a5*(y1-a6)) - 2*exp(-a5*(y1-a6)))
   
       v_new = np.append(V_1,V_2)
       
       z= np.append(x1,y1)

       plt.figure()
       plt.plot(z, v_new)
       plt.legend(['Best Potential', 'Morse Potential'])
       plt.xlabel('r (fm)')
       plt.ylabel('V(r) (MeV)')
       plt.axis([0, 15, -11, 10])
       plt.grid()
       plt.title('Potential Plot')
   
       return v_new
   
@numba.jit(nopython=True)
def f(x_in, y_in, v_new):
    kx = k * x_in
    
  # for l=0 state non-linear equation   
    # res = den_1 * k_1 * v_new * (cos(y_in) * sin(kx) + sin(y_in) * cos(kx)) ** 2
    
  # for l=2 state non-linear equation
    #res =  k_1 * v_new * den_1 * (((np.sin(kx))*(3*(kx)**(-2)-1)-3*(((kx)**(-1))*np.cos(kx)))*(np.cos(y_in))-((np.cos(kx))*((-3*(kx)**(-2))+1)-3*(((kx)**(-1))*np.sin(kx)))*(np.sin(y_in)))**2 
   
  # for l=4 state non-linear equation
    res = k_1 * v_new * den_1 *(np.sin(y_in+kx) + 10*np.cos(y_in+kx)*((kx)**(-1))-45*np.sin(y_in+kx)*((kx)**(-2))-105*np.cos(y_in+kx)*((kx)**(-3)) + 105*np.sin(y_in+kx)*(kx**-4))**2  
      
    return res

#@numba.jit(nopython=True)
def rk_method():
    idx = 0
    x = 0.01
    y = np.zeros_like(k)
    v_new = morse()
    for _ in range(3998):
        k1 = f(x, y, v_new[idx])
        k2 = f(x + h_4, y + h_4 * k1, v_new[idx + 1])
        k3 = f(x + h_4, y + h_8 * (k1 + k2), v_new[idx + 1])
        k4 = f(x + h_2, y - h_2 * k2 + h * k3, v_new[idx + 2])
        k5 = f(x + h_34, y + h_3_16 * k1 + h_9_16 * k4, v_new[idx + 3])
        k6 = f(x + h, y - h_3_7 * k1 + h_2_7 * k2 + h_12_7 * (k3 - k4) + h_8_7 * k5, v_new[idx + 4])
        y = y + h_90 * (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5 + 7 * k6)
        idx += 4
        x += h
    return y

delta_best = rk_method() * rad2deg


   


plt.figure()
plt.plot(E, pS, 'ro')
plt.grid()
plt.plot(E, delta_best, 'g-')
plt.xlabel('Energy MeV')
plt.ylabel('Phase Shift (degrees)')
plt.legend(['Experimental', 'Simulated'])
plt.title('Phase Shift Plot')
print('idx   Energy   1S0-Phase-Exp.      1S0-Phase-Sim.')
for idx, _ in enumerate(E):
    print(f'{idx + 1:2}   {int(E[idx]):2}       {pS[idx]:.3f}           {delta_best[idx]:.3f}')
@numba.jit(nopython=True)
def metric_absPercentError(delta_best):
        return mean(abs(delta_best - pS) * abs_pS_inv) * 100

print(f'metric\nAbsolute Percent Error:\t{metric_absPercentError(delta_best)}\n')