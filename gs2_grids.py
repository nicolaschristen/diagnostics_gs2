import numpy as np
import copy as cp
from math import pi

class gridobj:

    def __init__(self, myout, myin):

        print()
        print('constructing grids ... ', end='')

        # self.jtwist = myin['kt_grids_box_parameters']['jtwist']

        self.kx_gs2 = np.copy(myout['kx'])
        self.theta0_gs2 = np.copy(myout['theta0'])
        self.ky = np.copy(myout['ky'])

        if myout['kx_shift_present']:
            self.kx_shift = myout['kx_shift']
        else:
            self.kx_shift = None

        # number of kx and ky grid points
        self.nx = self.kx_gs2.size
        self.ny = self.ky.size

        # this is the index of the first negative value of kx
        # note gs2 orders kx as (0, dkx, ..., kx_max, -kx_max, -kx_max+dkx, ..., -dkx)
        self.nxmid = self.nx//2+1

        # get the monotonically increasing kx grid
        self.kx = np.concatenate((self.kx_gs2[self.nxmid:],self.kx_gs2[:self.nxmid]))
        self.theta0 = np.concatenate((self.theta0_gs2[:,self.nxmid:],self.theta0_gs2[:,:self.nxmid]),axis=1)

        self.kperp = np.arange(self.ny*self.nx,dtype=float).reshape(self.ny,self.nx)
        for i in range(self.nx):
            for j in range(self.ny):
                self.kperp[j,i] = np.sqrt(self.kx[i]**2 + self.ky[j]**2)
     
        if (self.nx > 1):
            # get real space grid in x
            self.xgrid_fft = 2*pi*np.fft.fftfreq(self.nx,self.kx_gs2[1])
            self.xgrid = np.concatenate((self.xgrid_fft[self.nxmid:],self.xgrid_fft[:self.nxmid]))
            self.xgrid_fine = np.linspace(self.xgrid[0],self.xgrid[-1],10*self.nx)
        else:
            self.xgrid_fft = np.arange(1,dtype=float)
            self.xgrid = np.arange(1,dtype=float)
            self.xgrid_fine = np.arange(1,dtype=float)
     
        if (self.ny > 1):
            # get real space grid in y
            ygrid_fft = 2*pi*np.fft.fftfreq(2*self.ny-1,self.ky[1])
            self.ygrid = np.concatenate((ygrid_fft[self.ny:],ygrid_fft[:self.ny]))
        else:
            ygrid_fft = np.arange(1,dtype=float)
            self.ygrid = np.arange(1,dtype=float)
     
        self.theta = cp.copy(myout['theta'])
        if self.theta is not None:
            self.ntheta = self.theta.size
        else:
            self.ntheta = 0
     
        self.vpa = cp.copy(myout['vpa'])
        if self.vpa is not None:
            self.nvpa = self.vpa.size
        else:
            self.nvpa = 0
     
        print('complete')  
