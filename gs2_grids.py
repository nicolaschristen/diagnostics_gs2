import numpy as np
import copy as cp
from math import pi
import gs2_storage as gstorage


def reorder(idx_first, var, dim):

    varshape = var.shape
    myslice = [slice(None)]*var.ndim
    myslice[dim] = list(range(varshape[dim]))
    myslice[dim] = myslice[dim][idx_first:] + myslice[dim][:idx_first]

    output = np.copy(var[myslice])

    return output


def init_and_save_mygrids(myout, fname):

    kx_gs2 = np.copy(myout['kx'])
    ky = np.copy(myout['ky'])

    # number of kx and ky grid points
    nx = kx_gs2.size
    ny = ky.size

    # this is the index of the first negative value of kx
    # note gs2 orders kx as (0, dkx, ..., kx_max, -kx_max, -kx_max+dkx, ..., -dkx)
    nxmid = nx//2+1

    # get the monotonically increasing kx grid
    kx = reorder(nxmid, kx_gs2, dim=0)

    kperp = np.arange(ny*nx,dtype=float).reshape(ny,nx)
    for i in range(nx):
        for j in range(ny):
            kperp[j,i] = np.sqrt(kx[i]**2 + ky[j]**2) # NDC: this is wrong, missing cross terms
 
    if (nx > 1):
        # get real space grid in x
        xgrid_fft = 2*pi*np.fft.fftfreq(nx,kx_gs2[1])
        xgrid = reorder(nxmid, xgrid_fft, dim=0)
    else:
        xgrid_fft = np.arange(1,dtype=float)
        xgrid = np.arange(1,dtype=float)
 
    if (ny > 1):
        # get real space grid in y
        ygrid_fft = 2*pi*np.fft.fftfreq(2*ny-1,ky[1])
        ygrid = reorder(ny, ygrid_fft, dim=0)
    else:
        ygrid_fft = np.arange(1,dtype=float)
        ygrid = np.arange(1,dtype=float)
 
    theta = cp.copy(myout['theta'])
    if theta is not None:
        ntheta = theta.size
    else:
        ntheta = 0
 
    vpa = cp.copy(myout['vpa'])
    if vpa is not None:
        nvpa = vpa.size
    else:
        nvpa = 0

    del i, j
    mygrids = locals()
    del mygrids['fname'], mygrids['myout']
    gstorage.save_to_file(fname, mygrids)
    return mygrids
