import numpy as np
import os
from matplotlib import pyplot as plt
import pickle
import scipy.optimize as opt
import sys
import scipy.interpolate as scinterp

# Add path to directory where pygs2 files are stored
maindir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.insert(0, maindir)

import gs2_plotting as gplot

def main():

    delt_scan = np.round(np.linspace(0.1,1.0,10),1)
    Ndelt = delt_scan.size
    jtwist_scan = [12,6,4,3,2,1] # [24,12,6,4,3,2,1]
    dkx_scan = np.array([0.125,0.25,0.375,0.5,0.75,1.5]) # [0.1,0.2,0.4,0.6,0.8,1.2,2.4]
    Ndkx = dkx_scan.size

    gamma_nrmed_scan = np.zeros((Ndelt, Ndkx))
    gamma_hires = 1.0

    for fname in os.listdir('.'):

        if fname.endswith('.floquet.dat'):

            my_vars = read_from_dat(fname)

            t = my_vars['t']
            delt = my_vars['delt']
            nwrite = my_vars['nwrite']
            jtwist = my_vars['jtwist']
            g_exb = my_vars['g_exb']
            Nf = my_vars['Nf']
            iky_list = my_vars['iky_list']
            sum_phi2bloon = my_vars['sum_phi2bloon']

            Tf = Nf*delt
            nt = t.size

            # Determine where this file will go in gamma array
            idelt = int(round(delt/0.1))-1
            idkx = jtwist_scan.index(jtwist)

            # Start comparing simulations at time-step it_start = N_start*Tfloquet/dt
            # ie after N_start Floquet oscillations
            # Normalise sum_phi2 by sum_phi2[it_start] for each run
            if g_exb != 0.0:
                N_start = 10 # TODO
                it_start = int(round((N_start*Tf/delt)/nwrite))
            else:
                fac = 0.5 # TODO
                it_start = round(fac*nt) # adapt this

            iiky = 0 # Look at smallest nonzero ky
            iky = iky_list[iiky]

            # Crop out first few Floquet periods
            sum_phi2_tmp = np.zeros(len(sum_phi2bloon[iiky])-it_start)
            for it in range(sum_phi2_tmp.size):
                sum_phi2_tmp[it] = sum_phi2bloon[iiky][it_start+it]
            # Nomalise sum_phi by its initial value
            if it_start > 0:
                sum_phi2_tmp = sum_phi2_tmp/sum_phi2_tmp[0]
            
            # Crop time array
            t_tmp = np.zeros(len(t)-it_start)
            for it in range(t_tmp.size):
                t_tmp[it] = t[it_start+it]

            # Compute growthrate
            [slope,dummy] = leastsq_lin(t_tmp,np.log(sum_phi2_tmp))
            gamma_nrmed_scan[idelt, idkx] = slope/2. # divide by 2 because fitted square

            # Check if this is the converged gamma to use for normalisations
            if idelt==0 and idkx==0:
                gamma_hires = slope/2.

    # Normalise growthrates to converged value
    gamma_nrmed_scan = gamma_nrmed_scan/gamma_hires

    # interpolate to nearest neighbour on fine, regular mesh ...
    grid_1d_dkx = np.zeros(Ndkx*Ndelt)
    grid_1d_delt = np.zeros(Ndkx*Ndelt)
    gamma_1d = np.zeros(Ndkx*Ndelt)
    idx_1d = 0
    for idkx in range(Ndkx):
        for idelt in range(Ndelt):
            grid_1d_dkx[idx_1d] = dkx_scan[idkx]
            grid_1d_delt[idx_1d] = delt_scan[idelt]
            gamma_1d[idx_1d] = gamma_nrmed_scan[idelt, idkx]
            idx_1d = idx_1d + 1
    Ndkx_fine = 1000
    dkx_scan_fine = np.linspace(dkx_scan.min(),dkx_scan.max(),Ndkx_fine)
    gamma_fine = scinterp.griddata((grid_1d_dkx, grid_1d_delt), gamma_1d, \
            (dkx_scan_fine[None,:], delt_scan[:,None]), method='nearest')

    # Set up potting defaults
    gplot.set_plot_defaults()

    myfig = plt.figure(figsize=(12,8))
    cmap = 'RdBu_r'
    my_xlabel = '$(\\Delta k_x)\\rho_i$'
    my_ylabel = '$(\\Delta t)v_{th,i}/a$'
    my_title = '$\\langle \\gamma \\rangle_t$'
    use_logcolor = False
    z = gamma_fine # gamma_nrmed_scan
    z_min, z_max = 1.0, z.max()
    gplot.plot_2d(z,dkx_scan_fine,delt_scan,z_min,z_max,my_xlabel,my_ylabel,my_title,cmap,use_logcolor)
    plt.xticks(np.round(np.array([0.25,0.5,0.75,1.5]),2))
    plt.yticks(delt_scan)

    figname = 'algo_mix_vs_delt_dkx.pdf' # TODO
    plt.savefig(figname)
    

def read_from_dat(datfile_name):

    with open(datfile_name, 'rb') as infile: # 'rb' stands for read bytes
        my_vars = pickle.load(infile)

    return my_vars

def leastsq_lin(x, y):
    
    # y_fit = a*x + b
    # minimising sum((y - f_fit)^2)
    N_x = x.size

    a = 1./(N_x*np.sum(np.power(x,2)) - np.sum(x)**2) * (N_x*np.sum(np.multiply(x,y)) - np.sum(x)*np.sum(y))
    
    b = 1./(N_x*np.sum(np.power(x,2)) - np.sum(x)**2) * (-1.*np.sum(x)*np.sum(np.multiply(x,y)) + np.sum(np.power(x,2))*np.sum(y))

    return [a, b]

def get_growthrate(t,tofit,it_start):
   
    popt, pcov = opt.curve_fit(lin_func, t[it_start:], np.log(tofit[it_start:]))
    return popt[0]

def lin_func(x,a,b):
    return a*x+b

# Execute main
if __name__ == '__main__':
    main()
