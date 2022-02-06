import numpy as np
from numpy import sqrt
import time
import matplotlib.pyplot as plt
import numba 

dN = 0.01

@numba.njit
def back_evolve(Phi_in):
  
    N_test = 64.0
    n = int(N_test/dN)
    Phi = np.zeros(n); DPhi = np.zeros(n)
    Phi[0] = Phi_in; DPhi[0] = -2/Phi_in
    
    for i in range(n-1):
        
        Phi[i+1] = Phi[i] + dN*DPhi[i]
        
        DPhi[i+1] = DPhi[i] - dN*3*DPhi[i] + dN*0.5*DPhi[i]**3 - dN*( 3 - 0.5*DPhi[i]**2 \
                                                                        )*(2/Phi[i])
            
    return Phi, DPhi

@numba.njit(parallel=True)
def field_correlations(nsim, phi_, dphi_):
    N_end = 64
    N = np.linspace(0, N_end, int(N_end/dN))
    phi_bg = phi_; dphi_bg = dphi_
    
    m = sqrt(4.4*10**-11)
    delphi2 = np.zeros(len(N))
    
    for i in numba.prange(nsim):
        
        phi_cg = np.zeros(len(N)); phi_cg[0] = phi_bg[0]
        dphi_cg = np.zeros(len(N)); dphi_cg[0] = dphi_bg[0]
        
        dW = np.random.randn(len(N))
        xi = np.zeros(len(N))
        xi[0] = dW[0]
        
        for j in range(len(N)-1):
            
            xi[j+1] = xi[j] - 2*xi[j]*dN + 2*sqrt(2*dN)*dW[j]
            phi_cg[j+1] = phi_cg[j] + dphi_cg[j]*dN + ( m*phi_cg[j]/sqrt(48*np.pi**2) ) \
                *xi[j]*dN
            dphi_cg[j+1] = dphi_cg[j] - 3*dN*dphi_cg[j] + dN*0.5*dphi_cg[j]**3 \
                -dN*(3 - 0.5*dphi_cg[j]**2)*(2/phi_cg[j])
                
        delphi = phi_cg - phi_bg
        delphi2 += delphi**2
        
    delphi2 = delphi2/nsim
    
    return delphi2

if __name__ == "__main__":
    
    ti = time.time()
    
    print("Starting operation!")
    
    N_end = 64
    N = np.linspace(0, N_end, int(N_end/dN))
    phi_bg, dphi_bg = back_evolve(16.0)
    nsim = 10**5
    delphi2 = field_correlations(nsim, phi_bg, dphi_bg)
    
    eps1 = 0.5*dphi_bg**2
    P_zeta_stochastic = np.gradient(delphi2/(2*eps1), dN)
    
    tf = time.time()
    print("Code execution time for " + str(nsim) + " simulations: " + str(tf-ti) + " seconds.")
    
    plt.plot(N, P_zeta_stochastic)
    plt.show()