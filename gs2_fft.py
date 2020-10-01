import numpy as np

def fft_gs2(fld, nx, ny, ky, kx_shift = None, x = None):
    
    # Fill the field for full kyrange using
    # the fact that the field is hermitian.
    # NB. the order of the wavenumbers should be the same as in GS2,
    # ie for example in x:
    # [0, dkx, ..., kxmax, -kmax, -kxmax+dkx, ..., -dkx]
    
    ny_full = 2*ny-1
    dky = ky[1]
    
    ky_full = np.zeros(ny_full)
    for iky in range(ny):
        ky_full[iky] = ky[iky]
    for iky in range(ny, ny_full):
        ky_full[iky] = -ky[ny_full-iky]
    
    fld_full = np.zeros((ny_full, nx))
    fld_full = fld_full.astype(complex)
    
    # First copy over ky>=0
    for iky in range(ny):
        fld_full[iky,:] = fld[iky,:]
    
    # Phase factor to consider for cases with flow shear
    if kx_shift is None:
        phase_fac = np.ones((ny_full, nx))
    else:
        # Expand kx_shift to have it for ky_full,
        # with kx_shift(-ky) = -kx_shift(ky),
        # gs2 ordered ky.
        kx_shift_full = np.zeros(ny_full)
        for iky in range(ny):
            kx_shift_full[iky] = kx_shift[iky]
        for iky in range(ny, ny_full):
            kx_shift_full[iky] = -kx_shift[ny_full-iky]
    
        expo = kx_shift_full.reshape(ny_full,1) * x
        phase_fac = np.exp(1j*expo)
    
    # Then for ky<0, use fld[-ky,kx] = conj(fld[ky,-kx])
    for iky in range(ny, ny_full):
        fld_full[iky,0] = np.conj(fld[ny_full-iky,0])
        for ikx in range(1,nx):
            fld_full[iky,ikx] = np.conj(fld[ny_full-iky, nx-ikx])
    		
    # Compute 2D inverse FFT
    
    # First FFT in x
    fld_full_fftx = np.fft.ifft(fld_full, axis=1)
    # Then multiply by (ky,x)-dependent phase factor and FFT in y
    fld_full_fftxy = np.real(np.fft.ifft(fld_full_fftx*phase_fac, axis=0))

    # Then re-arrange field to have growing x axis
    nxmid = nx//2 + 1
    fld_full_fftxy = np.concatenate((fld_full_fftxy[..., nxmid:],fld_full_fftxy[..., :nxmid]),axis=1)

    # np.fft.ifft returns 1/n * sum(phik*exp(...)) -> undo this normalisation
    fld_full_fftxy = fld_full_fftxy * nx * ny_full
    
    return ky_full, fld_full, fld_full_fftxy


def gs2fft(fld, mygrids): #we assume kx has positive and negative wavenumbers, ky only positive => add conjugates

    import numpy as np

    ny_full = 2*mygrids.ny-1
    #create the fftfreq-like array
    ky_full = np.arange(2*mygrids.ny-1,dtype=float)
    ky_full[:mygrids.ny] = mygrids.ky
    for i in range(-mygrids.ny+1,0):
        ky_full[i+2*mygrids.ny-1] = mygrids.ky[1]*i
                
    ###enforce hermitian symmetry
    fld_full=np.zeros(mygrids.nx*ny_full,dtype=complex).reshape(ny_full,mygrids.nx)
    # fill ky >= 0 entries
    fld_full[:mygrids.ny,:]=fld[:mygrids.ny,:]
    # generate ky < 0 entries using Hermiticity
    # i.e. fld(-k) = conjg(fld(k))
    for i in range(-mygrids.ny+1,0):
        # treat special case of kx=0
        fld_full[i+ny_full,0] = np.conj(fld[-i,0])
        # next treat general case of kx /= 0
        for j in range(1,mygrids.nx):
            fld_full[i+ny_full,j] = np.conj(fld[-i,mygrids.nx-j])
                        
    #calculate ifft2
    fft2d = np.real(np.fft.ifft2(fld_full))

    return fft2d, ky_full, fld_full
