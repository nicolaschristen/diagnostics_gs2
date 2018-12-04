##### Function to define inverse fft properly, with hermitian symmetries
##### the input will be f(kx,ky), kx and ky; the output will be the FT and
##### the axes x and y

def gs2fft(fld, mygrids): #we assume kx has positive and negative wavenumbers, ky only positive => add conjugates

	import numpy as np

	ny_full = 2*mygrids['ny']-1
	#create the fftfreq-like array
	ky_full = np.arange(2*mygrids['ny']-1,dtype=float)
	ky_full[:mygrids['ny']] = mygrids['ky']
	for i in range(-mygrids['ny']+1,0):
		ky_full[i+2*mygrids['ny']-1] = mygrids['ky'][1]*i
		
	###enforce hermitian symmetry
	fld_full=np.zeros(mygrids['nx']*ny_full,dtype=complex).reshape(ny_full,mygrids['nx'])
	# fill ky >= 0 entries
	fld_full[:mygrids['ny'],:]=fld[:mygrids['ny'],:]
	# generate ky < 0 entries using Hermiticity
	# i.e. fld(-k) = conjg(fld(k))
	for i in range(-mygrids['ny']+1,0):
		# treat special case of kx=0
		fld_full[i+ny_full,0] = np.conj(fld[-i,0])
		# next treat general case of kx /= 0
		for j in range(1,mygrids['nx']):
			fld_full[i+ny_full,j] = np.conj(fld[-i,mygrids['nx']-j])
			
	#calculate ifft2
	fft2d = np.real(np.fft.ifft2(fld_full))

	return fft2d, ky_full, fld_full
