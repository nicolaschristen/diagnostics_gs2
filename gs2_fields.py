import numpy as np
import numpy.fft as fft
import scipy.special as sp
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rcParams

from gs2_fft import gs2fft

class fieldobj:

    def __init__(self, myout, mygrids, mytime):
        
        ( self.phi2_avg, self.phi_gs2, self.dens_gs2, self.upar_gs2, self.tpar_gs2, self.tperp_gs2,
                self.phi, self.phi2, self.gphi_gs2, self.Ex_gs2, self.Ey_gs2, 
                self.zonal_E ) = self.get_attr(myout, mygrids, mytime)

    def get_attr(self, myout, mygrids, mytime):

        phi2_avg = myout['phi2']

        print()
        print('arranging fields in Fourier space ...',end='')

        # combine real and imaginary parts into single complex variables
        mydim = (mytime.ntime, mygrids.ny, mygrids.nx)
        phi_gs2 = form_complex('phi_igomega_by_mode', myout, mydim)
        
        mydim = (mytime.ntime, myout['nspec'], mygrids.ny, mygrids.nx)
        dens_gs2 = form_complex('ntot_igomega_by_mode', myout, mydim)
        upar_gs2 = form_complex('upar_igomega_by_mode', myout, mydim)
        tpar_gs2 = form_complex('tpar_igomega_by_mode', myout, mydim)
        tperp_gs2 = form_complex('tperp_igomega_by_mode', myout, mydim)

        # reorder kx grid to be monotonic
        phi = np.concatenate((phi_gs2[:,:,mygrids.nxmid:],phi_gs2[:,:,:mygrids.nxmid]),axis=2)
        phi2 = np.abs(phi)**2
        # get gyro-averaged phi (at vperp = vth)
        gphi_gs2 = np.copy(phi_gs2)
        for i in range(mygrids.nx):
            for j in range(mygrids.ny):
                gphi_gs2[:,j,i] = phi_gs2[:,j,i]*sp.j0(mygrids.kperp[j,i])

        # x and y components of gyro-averaged, non-zonal electric field
        mydim = (mytime.ntime, mygrids.ny, mygrids.nx)
        Ex_gs2 = np.zeros(mydim, dtype=complex)
        Ey_gs2 = np.zeros(mydim, dtype=complex)
        
        for i in range(mytime.ntime):
            for j in range(mygrids.nx):
                Ex_gs2[i,:,j] = -1j*mygrids.kx_gs2[j]*gphi_gs2[i,:,j]
                Ey_gs2[i,:,j] = -1j*mygrids.ky*gphi_gs2[i,:,j]
        # separate out the zonal component
        zonal_E = np.copy(Ex_gs2[:,0,:])
        Ex_gs2[:,0,:] = 0.0
        
        print('complete')

        return ( phi2_avg, phi_gs2, dens_gs2, upar_gs2, tpar_gs2, tperp_gs2,
                phi, phi2, gphi_gs2, Ex_gs2, Ey_gs2, zonal_E )

class field_ffted_obj:

    def __init__(self, myout, mygrids, mytime, myfields):
        
        ( self.Ex_power_spectrum, self.Ey_power_spectrum, self.phi_xyt, self.Ex_xyt, self.Ey_xyt,
                self.Ex, self.Ey, self.Ex2, self.Ey2 ) = self.get_attr(myout, mygrids, mytime, myfields)

    def get_attr(self, myout, mygrids, mytime, myfields):

        print()
        print('calculating Ex and Ey power spectrum...',end='')
        
        if (mygrids.nx > 2) and (mygrids.ny > 2):
            Ex_power_spectrum = np.arange(mytime.ntime_steady,dtype=float)
            Ex_power_spectrum = np.fft.fftshift( np.sum(
                np.abs(np.fft.fft(myfields.Ex_gs2[mytime.it_min:mytime.it_max,:,:],axis=0))**2,axis=(1,2) ) )
            Ey_power_spectrum = np.arange(mytime.ntime_steady,dtype=float)
            Ey_power_spectrum = np.fft.fftshift( np.sum(
                np.abs(np.fft.fft(myfields.Ey_gs2[mytime.it_min:mytime.it_max,:,:],axis=0))**2,axis=(1,2) ) )
        else:
            Ex_power_spectrum = 0
            Ex_power_spectrum = 0
            Ey_power_spectrum = 0
            Ey_power_spectrum = 0

        print('complete')

        phi_xyt=[]
        Ex_xyt=[]
        Ey_xyt=[]
        Ex=[]
        Ey=[]
        Ex2=[]
        Ey2=[]
        if (mygrids.nx > 1) and (mygrids.ny > 1):
            
            print()
            print('calculating Phi in real space...',end='')
            
            #Phi in real-space
            mydim = (mytime.ntime, 2*mygrids.ny-1, mygrids.nx)
            phi_xyt = np.zeros(mydim, dtype=float)
            for i in range (mytime.ntime):
                phi_xyt[i,:,:] = np.fft.fftshift(gs2fft(myfields.phi_gs2[i,:,:]*(2*mygrids.ny-1)*mygrids.nx, mygrids)[0])
            
            print('complete')

            print()
            print('calculating Ex and Ey in real space...',end='')
            
            # non-zonal Ex and Ey in real-space
            Ex_xyt = np.zeros(mydim, dtype=float)
            Ey_xyt = np.zeros(mydim, dtype=float)
            for i in range (mytime.ntime):
                Ex_xyt[i,:,:] = np.fft.fftshift(gs2fft(myfields.Ex_gs2[i,:,:]*(2*mygrids.ny-1)*mygrids.nx, mygrids)[0])
                Ey_xyt[i,:,:] = np.fft.fftshift(gs2fft(myfields.Ey_gs2[i,:,:]*(2*mygrids.ny-1)*mygrids.nx, mygrids)[0])

            # now that we've obtained Ex and Ey in real-space,
            # shift kx indices so kx is monotonic
            mydim = (mytime.ntime, mygrids.ny, mygrids.nx)
            Ex = np.zeros(mydim, dtype=complex)
            Ey = np.zeros(mydim, dtype=complex)
            for i in range(mytime.ntime):
                for j in range(mygrids.nx):
                    Ex[i,:,j] = 1j*mygrids.kx[j]*phi[i,:,j]
                    Ey[i,:,j] = 1j*mygrids.ky*phi[i,:,j]
            # zero out zonal component
            Ex[:,0,:] = 0.0

            Ex2 = np.abs(Ex)**2
            Ey2 = np.abs(Ey)**2

            print('complete')

        return ( Ex_power_spectrum, Ey_power_spectrum, phi_xyt, Ex_xyt, Ey_xyt,
                Ex, Ey, Ex2, Ey2 )
            
def form_complex(varname, myout, outdim):

    if (myout[varname + '_present']):
        arr = myout[varname][..., 0] + 1j*myout[varname][..., 1]
    else:
        arr = np.zeros(outdim,dtype=complex)

    return arr


def plot_power_spectrum(myfields, mytime):

    xlab = '$\omega$'
    fig = plt.figure(figsize=(12,8))
    plt.plot(mytime.frequency, myfields.Ex_power_spectrum,label='$E_x$')
    plt.plot(mytime.frequency, myfields.Ey_power_spectrum,label='$E_y$')
    plt.xlabel(xlab)
    plt.yscale('log')
    plt.legend()
    
    return fig
