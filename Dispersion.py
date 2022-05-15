# Spectral method for propagation t,z
# with air detail
# In summer 2021

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

start_time = datetime.now()


landa = 800*1e-9            # Wave legenth
k0 = 2*np.pi/landa			# Centeral wave number
beta_2 = 8*1e-27            # GVD
wt = 50*1e-15               # Pulse width

c = 3*1e8				    # Speed of light
n = 1						# Refractive index
n0 = 1						# Linear index coefficient

N= 1024					    # Number of grid points
Nz = 1024					# Number of steps
Nt = 2048

z = np.linspace(0,2,Nz)             # Discretizing propagation direction
T = 2048 * 1e-15
t0 = np.linspace(-T/2, T/2, Nt)     # Discretizing time domain
dt = t0[2]-t0[1]                    # Time step size
dz = max(z)/Nz				        # Step size

################### Definition initial field #####################

E = np.zeros(Nt,dtype=np.complex_)

St = np.zeros((Nt,Nz),dtype=np.complex_)

E0 = np.sqrt(1)                      # Initial intensity of pulse
E = E0 * np.exp(-(t0/wt)**2 )        # Pulse structure

St[:,0] = E

nu = (1/T) * np.append(np.array(range(0, (int(Nt/2))), dtype=np.complex_), np.array(range(-int(Nt/2), 0), dtype=np.complex_))
Aj = np.exp((0-1j)*dz*beta_2*((2*np.pi*nu)**2)/2)
################### Propagation Pulse #####################


for j in range(1,Nz):
    nn = j
    
    E = np.fft.fft(E)
    E = E *Aj
    E = np.fft.ifft(E)
    
    St[:,nn] = E
    if (nn%200) == 0:
        print(nn)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
################### Plot #####################

Itz = abs(St)**2

plt.contourf(z*1e2,t0*1e15,Itz,levels=50, cmap='hot')
cbar=plt.colorbar(shrink=0.75)         # Colorbar
plt.grid()
plt.xlabel('$z (cm)$')                 # Axes labels, title, plot and axes range
plt.ylabel('$time (fs)$')
plt.gca().set_aspect(0.025)
#plt.savefig('Ixz.png')
plt.show()                              # Displays figure on screen

plt.plot(z*1e2,np.max(Itz,axis=0))
plt.grid()
plt.xlabel('$z (cm)$')                 # axes labels, title, plot and axes range
plt.ylabel('$Intensity (w/m^{-2})$')
plt.gca().set_aspect('auto')
plt.show()

