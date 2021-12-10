import numpy as np
#import numba 

class mukhanov_sasaki:
    '''
    Python class for numerically solving the Mukhanov-Sasaki equation beyond slow-roll
    approximations. 

    Input parameters:
    k - comoving wavenumber of perturbation mode (scalar)
    N - efolds (array)
    a - scale factor (array)
    phi - background evolution of inflaton (array)
    dphi - background evolution of inflaton conjugate momentum (array)
    V - inflaton potential (array)
    '''
    def __init__(self, k, N, a, phi, dphi, V):
        self.k = k
        self.N = N 
        self.a = a
        self.phi = phi 
        self.dphi = dphi 
        self.V = V

    def efold_bounds(self):
        '''
        This method computes the efold times for which the modes are evolved. Defining
        the comoving horizon to be aH, the sub and superhorizon limits are given by
        100aH and 0.01aH. The function returns Nii and Nff which are efolds for which
        the sub and superhorizon conditions are satisfied for a given value of k. The
        function returns the real and imaginary components of the Mukhanov variable
        (uk and vk) and their efold derivatives (duk and dvk).
        '''
        H = np.sqrt( self.V/(3-0.5*self.dphi**2) )
        horizon = self.a*H 
        initial = 100*horizon 
        final = 0.01*horizon 
        test_in = np.absolute(initial-self.k)
        test_fin = np.absolute(final-self.k)

        Nii = np.where( test_in==np.min(test_in) )[0][0]
        Nff = np.where( test_fin==np.min(test_fin) )[0][0]

        return Nii, Nff 

    def mode_evolve(self):
        '''
        This method computes the evolution of inflaton modes from Nii to Nff by solving
        the Mukhanov-Sasaki equation using fourth order Runge-Kutta (RK4) method.  
        '''
        dN = self.N[1] - self.N[0]
        eps1 = 0.5*self.dphi**2
        eps2 = np.gradient(eps1, dN)/eps1 
        eta = eps1 - 0.5*eps2 
        d_eps = np.gradient(eps1, dN)
        d_eta = np.gradient(eta, dN)

        H = np.sqrt( self.V/(3-0.5*self.dphi**2) )

        Ni, Nf = self.efold_bounds()
        n = Nf - Ni + 1

        uk = np.zeros(n); duk = np.zeros(n)
        vk = np.zeros(n); dvk = np.zeros(n)

        #Bunch-Davies vacuum conditions
        uk[0] = 1/np.sqrt(2*self.k); duk[0] = 0
        vk[0] = 0; dvk[0] = -np.sqrt(self.k)/(0.01*np.sqrt(2)*self.k)

        for i in range(n-1):

            f1 = -(1-eps1[Ni+i])*duk[i] - (self.k**2 /(self.a[Ni+i]*H[Ni+i])**2)*uk[i] - (1+eps1[Ni+i]-eta[Ni+i]) \
                *(eta[Ni+i]-2)*uk[i] + (d_eps[Ni+i]-d_eta[Ni+i])*uk[i]
            F1 = duk[i]
            f2 = -(1-eps1[Ni+i])*(duk[i]+0.5*f1*dN) - (self.k**2 /(self.a[Ni+i]*H[Ni+i])**2)*(uk[i]+0.5*F1*dN) \
                - (1+eps1[Ni+i]-eta[Ni+i])*(eta[Ni+i]-2)*(uk[i]+0.5*F1*dN) + (d_eps[Ni+i]-d_eta[Ni+i])*(uk[i]+0.5*F1*dN)
            F2 = duk[i] + 0.5*f1*dN 
            f3 = -(1-eps1[Ni+i])*(duk[i]+0.5*f2*dN) - (self.k**2 /(self.a[Ni+i]*H[Ni+i])**2)*(uk[i]+0.5*F2*dN) \
                - (1+eps1[Ni+i]-eta[Ni+i])*(eta[Ni+i]-2)*(uk[i]+0.5*F2*dN) + (d_eps[Ni+i]-d_eta[Ni+i])*(uk[i]+0.5*F2*dN)
            F3 = duk[i] + 0.5*f2*dN 
            f4 = -(1-eps1[Ni+i])*(duk[i]+f3*dN) - (self.k**2 /(self.a[Ni+i]*H[Ni+i])**2)*(uk[i]+F3*dN) \
                - (1+eps1[Ni+i]-eta[Ni+i])*(eta[Ni+i]-2)*(uk[i]+F3*dN) + (d_eps[Ni+i]-d_eta[Ni+i])*(uk[i]+F3*dN)
            F4 = duk[i] + f3*dN

            uk[i+1] = uk[i] + dN*(F1 + 2*F2 + 2*F3 + F4)/6
            duk[i+1] = duk[i] + dN*(f1 + 2*f2 + 2*f3 + f4)/6 

            g1 = -(1-eps1[Ni+i])*dvk[i] - (self.k**2 /(self.a[Ni+i]*H[Ni+i])**2)*vk[i] - (1+eps1[Ni+i]-eta[Ni+i]) \
                *(eta[Ni+i]-2)*vk[i] + (d_eps[Ni+i]-d_eta[Ni+i])*vk[i]
            G1 = dvk[i]
            g2 = -(1-eps1[Ni+i])*(dvk[i]+0.5*g1*dN) - (self.k**2 /(self.a[Ni+i]*H[Ni+i])**2)*(vk[i]+0.5*G1*dN) \
                - (1+eps1[Ni+i]-eta[Ni+i])*(eta[Ni+i]-2)*(vk[i]+0.5*G1*dN) + (d_eps[Ni+i]-d_eta[Ni+i])*(vk[i]+0.5*G1*dN)
            G2 = dvk[i] + 0.5*g1*dN 
            g3 = -(1-eps1[Ni+i])*(dvk[i]+0.5*g2*dN) - (self.k**2 /(self.a[Ni+i]*H[Ni+i])**2)*(vk[i]+0.5*G2*dN) \
                - (1+eps1[Ni+i]-eta[Ni+i])*(eta[Ni+i]-2)*(vk[i]+0.5*G2*dN) + (d_eps[Ni+i]-d_eta[Ni+i])*(vk[i]+0.5*G2*dN)
            G3 = dvk[i] + 0.5*g2*dN 
            g4 = -(1-eps1[Ni+i])*(dvk[i]+g3*dN) - (self.k**2 /(self.a[Ni+i]*H[Ni+i])**2)*(vk[i]+G3*dN) \
                - (1+eps1[Ni+i]-eta[Ni+i])*(eta[Ni+i]-2)*(vk[i]+G3*dN) + (d_eps[Ni+i]-d_eta[Ni+i])*(vk[i]+G3*dN)
            G4 = dvk[i] + g3*dN

            vk[i+1] = vk[i] + dN*(G1 + 2*G2 + 2*G3 + G4)/6
            dvk[i+1] = dvk[i] + dN*(g1 + 2*g2 + 2*g3 + g4)/6 

        return uk, duk, vk, dvk 