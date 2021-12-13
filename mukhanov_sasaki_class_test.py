'''
Mukhanov-Sasaki class test using Deformed Starobinsky potential
'''
import numpy as np
from numpy import sqrt, exp
import time
import matplotlib.pyplot as plt
import numba

alpha = sqrt(2./3)
xi = -0.480964
beta = 1.114905
V0 = 1.27*10**-9

dN = 0.001

@numba.njit
def V(x):
    return V0*(1 + xi - exp(-alpha*x) - xi*exp(-beta*x**2))**2

@numba.njit
def dV(x):
    return 2*V0*(1 + xi - exp(-alpha*x) - xi*exp(-beta*x**2))*(alpha*exp(-alpha*x) \
        + 2*beta*xi*x*exp(-beta*x**2))

@numba.njit
def ddV(x):
    return 2*V0*(alpha*exp(-alpha*x) + 2*beta*xi*x*exp(-beta*x**2))**2 + 2*V0 \
        *(1 + xi - exp(-alpha*x) - xi*exp(-beta*x**2))*(-(alpha**2)*exp(-alpha*x) \
            + 2*beta*xi*exp(-beta*x**2) - 4*xi*(beta**2)*(x**2)*exp(-beta*x**2))

@numba.njit
def back_evolve(phi_in, efolds):
    N_test = efolds
    n = int(N_test/dN)
    phi = np.zeros(n); phi[0] = phi_in
    phi_M = np.zeros(n); phi_M[0] = -dV(phi_in)/V(phi_in)

    for i in range(n-1):
        K1 = dN*phi_M[i]
        L1 = dN*( -3*phi_M[i] + 0.5*phi_M[i]**3 - (3 - 0.5*phi_M[i]**2)*dV(phi[i])/V(phi[i]) )
        K2 = dN*(phi_M[i] + L1)
        L2 = dN*( -3*(phi_M[i] + L1) + 0.5*(phi_M[i] + L1)**3 - (3 - 0.5*(phi_M[i] + L1)**2)*dV(phi[i] + K1)/V(phi[i] + K1) )
        
        phi[i+1] = phi[i] + 0.5*(K1 + K2)
        phi_M[i+1] = phi_M[i] + 0.5*(L1 + L2)

    return phi, phi_M

if __name__ == "__main__":
    N_end = 70
    N = np.linspace(0, N_end, int(N_end/dN))
    phi_bg, dphi_bg = back_evolve(5.82, N_end)
    potential = V(phi_bg)
    a0 = 0.217198
    a = a0*exp(N)

    eps1 = 0.5*dphi_bg**2
    eps2 = np.gradient(eps1, dN)/eps1 
    eta = eps1 - 0.5*eps2 
    d_eps = np.gradient(eps1, dN)
    d_eta = np.gradient(eta, dN)

    #import the class
    from mukhanov_sasaki_class import mukhanov_sasaki

    k = 0.05 #mode for which the evolution is computed
    MS = mukhanov_sasaki(N, a, phi_bg, dphi_bg, potential)
    Nii, Nff = MS.efold_bounds(k)
    n = Nff - Nii + 1
    N = np.linspace(dN*Nii, dN*Nff, n)
    uk, duk, vk, dvk = MS.mode_evolve(k, eps1, eta, d_eps, d_eta)

    '''
    The method mode_evolve() returns the Mukhanov variable. The curvature perturbation
    can be computed using zeta_k = u_k / z, where z = a*dphi.
    '''

    Pzeta_k = (k**3 /(2*np.pi**2))*(uk**2 + vk**2)/( a[Nii:Nff+1]*dphi_bg[Nii:Nff+1] )**2

    plt.plot(N, Pzeta_k)
    plt.yscale('log')
    plt.show()
   