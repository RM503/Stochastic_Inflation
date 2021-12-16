'''
This code solves the coarse-grained evolution of the inflaton and conjugate momentum
by solving the inflaton-Langevin equation using the exact expressions for noise.

V(phi) = V0(1 + xi - exp(-alpha*phi) - xi*exp(-beta*phi^2))^2

The inflaton-Langevin equations are simulated I-times and the loop is optimized 
using the parallel-range (prange) function of numba.
'''
import numpy as np
from numpy import sqrt, exp
import time
import matplotlib.pyplot as plt
import numba
import sys

location = "/home/rafid/Documents/codes/Stochastic_Inflation/"
sys.path.append(location)

from mukhanov_sasaki_class import mukhanov_sasaki

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

@numba.njit(parallel=True)
def field_correlations(I, phi_, dphi_, eps_, eta_, d_eps_, d_eta_, efolds_):
    '''
    Pass d_eps and d_eta since numpy.gradient is not supported in numba
    '''
    N_end = efolds_ 
    N = np.linspace(0, N_end, int(N_end/dN))
    phi_bg = phi_; dphi_bg = dphi_
    potential = V(phi_bg)
    a0 = 0.217198
    a = a0*exp(N)

    eps1 = eps_
    eta = eta_ 
    d_eps = d_eps_ 
    d_eta = d_eta_ 
    H = sqrt(potential/(3-eps1))

    sigma = 10**-2 #set coarse-graining scale by k_sigma = sigma a*H
    NN = 10**4 #observable scale set an N=10, corresponding to point 10000 in efold grid
    N_run = np.linspace(0, 60, int(60/dN))
    delphi2 = np.zeros(len(N_run))
    delpi2 = np.zeros(len(N_run))
    NoiseAmp_phiphi = np.zeros(len(N_run))
    NoiseAmp_pipi = np.zeros(len(N_run))

    MS = mukhanov_sasaki(N_run, a, phi_bg, dphi_bg, potential)

    for j in range(len(N_run)):
        
        k = sigma*a[NN+j]*H[NN+j]
        Nii, Nff = MS.efold_bounds(k)
        Ni = dN*Nii 
        Nf = dN*Nff 
        n = Nff - Nii + 1
        N = np.linspace(Ni, Nf, n)

        uk, duk, vk, dvk = MS.mode_evolve(k, eps1, eta, d_eps, d_eta)
        
        '''
        Inflaton perturbations are related to the Mukhanov variable through the transformation
        Re_delphi = uk/a and Im_delphi = vk/a (similarly for conjugate momentum) 
        '''

        Re_delphi = uk/a[Nii:Nff+1]
        Im_delphi = vk/a[Nii:Nff+1]

        Re_delpi = duk/a[Nii:Nff+1]
        Im_delpi = dvk/a[Nii:Nff+1]

        Pdelphidelphi = (k**3 /(2*np.pi**2))*( Re_delphi[-1]**2 + Im_delphi[-1]**2 )
        Pdelpidelpi = (k**3 /(2*np.pi**2))*( Re_delpi[-1]**2 + Im_delpi[-1]**2 )

        '''
        Define the noise amplitudes, which constitute the deterministic parts of the noise terms
        in the SDEs. The noise amplitudes are evaluated beforehand for all the Fourier modes of
        interest, determined by k_sigma = sigma a(N)*H(N) at each N.
        '''

        NoiseAmp_phiphi[j] = (1-eps1[NN+j])*Pdelphidelphi
        NoiseAmp_pipi[j] = (1-eps1[NN+j])*Pdelpidelpi

    phi_bg_run, dphi_bg_run = back_evolve(5.51, 60)
    
    for i in numba.prange(I):

        F = np.random.randn(len(N_run))/sqrt(dN)
        S = np.random.choice(np.array([-1, 1]), len(N_run))

        phi_cg = np.zeros(N_run)
        dphi_cg = np.zeros(N_run)
        phi_cg[0] = 5.51
        dphi_cg[0] = -dV(phi_cg[0])/V(phi_cg[0])

        for j in range(len(N_run)):

            k1 = dN*dphi_cg[j] + (dN*F[j] - S[j]*sqrt(dN))*sqrt(NoiseAmp_phiphi[j])
            l1 = dN*( -3*dphi_cg[j] + 0.5*dphi_cg[j]**3 - (3 - 0.5*dphi_cg[j]**2)*dV(phi_cg[j])/V(phi_cg[j]) ) \
                    + (dN*F[j] - S[j]*sqrt(dN))*sqrt(NoiseAmp_pipi[j])
            k2 = dN*(dphi_cg[j] + l1) + (dN*F[j] + S[j]*sqrt(dN))*sqrt(NoiseAmp_phiphi[j+1])
            l2 = dN*( -3*(dphi_cg[j] + l1) + 0.5*(dphi_cg[j] + l1)**3 - (3 - 0.5*(dphi_cg[j] + l1)**2) \
                     *dV(phi_cg[j] + k1)/V(phi_cg[j] + k1) ) +  (dN*F[j] + S[j]*sqrt(dN))*sqrt(NoiseAmp_pipi[j+1])
            
            phi_cg[j+1] = phi_cg[j] + 0.5*(k1 + k2)
            dphi_cg[j+1] = dphi_cg[j] + 0.5*(l1 + l2)

        delphi = phi_cg - phi_bg_run
        delpi = dphi_cg - dphi_bg_run 
        delphi2 += delphi**2
        delpi2 += delpi**2 

    delphi2 = delphi2/I 
    delpi2 = delpi2/I