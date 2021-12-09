'''
This code solves the coarse-grained evolution of the inflaton and conjugate momentum
by solving the inflaton-Langevin equation using slow-roll approximated noise.

V(phi) = V0(1 + xi - exp(-alpha*phi) - xi*exp(-beta*phi^2))^2

The inflaton-Langevin equations are simulated I-times and the loop is optimized 
using the parallel-range (prange) function of numba.
'''
import numpy as np
from numpy import sqrt, exp
import time
import matplotlib.pyplot as plt
import numba
#from multiprocessing import Pool

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
def field_correlations(I, phi_, dphi_, efolds_):
    N_end = efolds_
    N = np.linspace(0, N_end, int(N_end/dN))
    phi_bg = phi_; dphi_bg = dphi_ #background evolution relabelled
    
    eps1 = 0.5*dphi_bg**2
    H = sqrt(V(phi_bg)/(3 - eps1))
    delphi2 = np.zeros(len(N))
    
    
    for i in numba.prange(I):
        
        F = np.random.randn(len(N))/sqrt(dN)
        S = np.random.choice(np.array([-1, 1]), len(N))
        
        #define the coarse-grained fields with cg
        '''
        Stochastic RK2 implementation that readily reduces to classical RK2
        in the absence of a diffusion term. The modified stochastic terms depend
        on S which alternate between +-1 with probability 1/2
        '''
        
        phi_cg = np.zeros(len(N))
        dphi_cg = np.zeros(len(N))
        phi_cg[0] = phi_bg[0]
        dphi_cg[0] = dphi_bg[0]
        
        
        for j in range(len(N)-1):
            
            k1 = dN*dphi_cg[j] + (dN*F[j] - S[j]*sqrt(dN))*H[j]/(2*np.pi)
            l1 = dN*( -3*dphi_cg[j] + 0.5*dphi_cg[j]**3 - (3 - 0.5*dphi_cg[j]**2)*dV(phi_cg[j])/V(phi_cg[j]) )
            k2 = dN*(dphi_cg[j] + l1) + (dN*F[j] + S[j]*sqrt(dN))*H[j+1]/(2*np.pi)
            l2 = dN*( -3*(dphi_cg[j] + l1) + 0.5*(dphi_cg[j] + l1)**3 - (3 - 0.5*(dphi_cg[j] + l1)**2) \
                     *dV(phi_cg[j] + k1)/V(phi_cg[j] + k1) )
            
            phi_cg[j+1] = phi_cg[j] + 0.5*(k1 + k2)
            dphi_cg[j+1] = dphi_cg[j] + 0.5*(l1 + l2)

        delphi = phi_cg - phi_bg
        delphi2 += delphi**2
        
    delphi2 = delphi2/I
    
    return delphi2
    
if __name__ == "__main__":
    
    ti = time.time()

    print("Starting operation!")
    
    N_end = 70
    N = np.linspace(0, N_end, int(N_end/dN))
    phi_bg, dphi_bg = back_evolve(5.82, N_end)
    nsim = 10**5
    delphi2 = field_correlations(nsim, phi_bg, dphi_bg, N_end)
    
    eps1 = 0.5*dphi_bg**2
    eps2 = np.gradient(eps1, dN)/eps1
    H = sqrt(V(phi_bg)/(3-eps1))
    d_delphi2 = np.gradient(delphi2, dN)
    P_zeta_stochastic = ( 0.5/(eps1*(1-eps1)) )*(d_delphi2 - eps2*delphi2) #Stochastic power spectrum
    P_zeta = H**2 /(8*np.pi**2 *eps1) #Power spectrum (standard calculation in slow-roll)

    #np.savetxt('Deformed_Starobinsky_SR_delphi2_1million.txt', np.transpose([N, delphi2]))
    #np.savetxt('Deformed_Starobinsky_SR_Pzeta_1million.txt', np.transpose([N, P_zeta_stochastic, P_zeta]))
    
    tf = time.time()
    print("Code execution time for " + str(nsim) + "simulations: " + str(tf-ti) + " seconds.")

    plt.scatter(N, P_zeta_stochastic, s=2)
    plt.plot(N, P_zeta, c='r')
    plt.yscale('log')
    plt.show()

