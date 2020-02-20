import sys
import os

# Add path to directory where scan-files are stored
sys.path.insert(1, '/marconi/home/userexternal/nchriste/codes/scan_gs2/paramfiles')

import f90nml as fnml
import numpy as np
import copy as cp
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
from math import pi

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=20)
rcParams.update({'figure.autolayout': True})
rcParams.update({'legend.fontsize': 12, 'legend.handlelength': 4})
rcParams.update({'legend.frameon': False})

ONE = 'one'
TWO = 'two'
NDIM_MAX = TWO

# Dict that will contain every parameter to scan
scan = {ONE:[],TWO:[]}






# vvv User parameters vvv

# Import all parameters from paramfiles/myfile.py
base_name = 'rpsi_0.6'  
pf = __import__('scan_linrange_coll_elec_ijp_950_rpsi_06')  

# Number of dimensions in the scan
# e.g. vs R/LTi -> ndim = ONE
ndim = ONE

# Define first dimension of the scan
firstdim_label = '$\\nu_{ee}\ [v_{th}/a]$' # for plotting
firstdim_var = 'vnewk' # name of variable to append to figure names
firstdim = pf.vnewk # variable name in paramfiles/myfile.py

# Apply limits to axis when plotting ?
use_my_xlim = False
my_xlim = (0.0, 2.0)  

use_my_ylim = False
my_ylim_max = (0.0, 0.50)
my_ylim_avg = (-0.1, 0.15)

# ^^^ User parameters ^^^







def main():


    # Add all parameters to the current scan

    nparams = len(pf.name)
    for iparam in range(nparams):
        add_param_to_scan(scan, pf.name[iparam], pf.dim[iparam], pf.namelist[iparam], pf.scandim[iparam], pf.func[iparam])


    # Read data from .dat files for every file in scan

    tt0_collec = []
    ky_collec = []
    gamma_collec = []
    omega_collec = []
    qe_vs_qi_collec = []
    scandim=ONE
    valtree = [0*i for i in range(nparams)]
    read_data(base_name, valtree, scandim, tt0_collec, ky_collec, gamma_collec, omega_collec, qe_vs_qi_collec)


    # Save data for whole scan in file

    vardict = {}
    vardict['tt0_collec'] = tt0_collec
    vardict['ky_collec'] = ky_collec
    vardict['gamma_collec'] = gamma_collec
    vardict['omega_collec'] = omega_collec
    vardict['qe_vs_qi_collec'] = qe_vs_qi_collec
    write_to_file(vardict)





    # Plotting

    naky = gamma_collec[0].shape[0]
    ntt0 = gamma_collec[0].shape[1]
    valdim = firstdim.size

    # Check whether to plot vs ky, theta0, or both.
    # Squeeze out unsuseful dimensions.
    if ntt0 > 1 and naky > 1:
        plot1d = False
        plot2d = True
    else:
        if ntt0 > 1:
            xlabel = '$\\theta_0$'
            xvar = 'theta0'
            xvar_collec = tt0_collec
        else:
            xlabel = '$k_y\\rho_i$'
            xvar = 'ky'
            xvar_collec = ky_collec
        for ival in range(valdim):
            gamma_collec[ival] = np.squeeze(gamma_collec[ival])
            omega_collec[ival] = np.squeeze(omega_collec[ival])
        plot1d = True
        plot2d = False


    # Plots vs xvar

    if plot1d:

        color_collec = plt.cm.gnuplot_r(np.linspace(0.05,0.9,firstdim.size))


        # Plot gamma vs xvar

        pdfname = 'postproc/linrange_gamma_scan_'+firstdim_var+'_vs_'+xvar+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel(xlabel)
        plt.ylabel('$\\gamma\ [v_{th}/a]$')
        valdim = firstdim.size
        for ival in range(valdim):
            plt.plot(xvar_collec[ival], gamma_collec[ival], linewidth=2.0, color=color_collec[ival])
            my_legend.append(firstdim_label+'$ = '+str(firstdim[ival])+'$')
        if use_my_ylim:
            plt.ylim(my_ylim_avg)
        if use_my_xlim:
            plt.xlim(my_xlim)
        legend = plt.legend(my_legend, frameon = True, fancybox = False, fontsize=10)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        frame.set_linewidth(0.5)
        frame.set_alpha(1)
        plt.savefig(pdfname)

        
        # Plot real frequency vs xvar

        pdfname = 'postproc/linrange_omega_scan_'+firstdim_var+'_vs_'+xvar+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel(xlabel)
        plt.ylabel('$\\omega\ [v_{th}/a]$')
        valdim = firstdim.size
        for ival in range(valdim):
            plt.plot(xvar_collec[ival], omega_collec[ival], linewidth=2.0, color=color_collec[ival])
            my_legend.append(firstdim_label+'$ = '+str(firstdim[ival])+'$')
        if use_my_ylim:
            plt.ylim(my_ylim_avg)
        if use_my_xlim:
            plt.xlim(my_xlim)
        legend = plt.legend(my_legend, frameon = True, fancybox = False)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        frame.set_linewidth(0.5)
        frame.set_alpha(1)
        plt.savefig(pdfname)


        # Plot Qe/Qi vs first scan dimension

        pdfname = 'postproc/linrange_qe_vs_qi_scan_'+firstdim_var+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel(firstdim_label)
        plt.ylabel('$\\langle Q_e/Q_i \\rangle_t$')
        plt.plot(firstdim , qe_vs_qi_collec, linewidth=2.0, color='k')
        plt.savefig(pdfname)





class gs2_param:
    def __init__(self, var='', dim=np.array([]), in_list='', func=None):
        self.name = var
        self.dim = dim
        self.namelist = in_list
        self.func = func

def add_param_to_scan(scan, name, dim, namelist, scandim, func):
    newparam = gs2_param(name, dim, namelist, func)
    if scandim == ONE:
        scan[ONE].append(newparam)
    elif scandim == TWO:
        scan[TWO].append(newparam)
    else:
        sys.exit('ERROR: this code only supports up to '+NDIM_MAX+' dimensions for a scan.')

def increment_dim(scandim):
    if scandim == ONE:
        scandim = TWO
    else:
        scandim = END
    return scandim

def read_data(fname, valtree, scandim, tt0_collec, ky_collec, gamma_collec, omega_collec, qe_vs_qi_collec):

    # Iterate over every set of values taken by parameters in this dimension of the scan.
    for ival in range(scan[scandim][0].dim):
    
        # Name-base and patch to be modified for this ival.
        my_fname = fname

        # For every parameter in this dimension of the scan, modify the files.
        for iparam in range(len(scan[scandim])):
            # Append parameter to namelist for in-file patching
            var = scan[scandim][iparam].name
            val = scan[scandim][iparam].func(ival,valtree)
            # Append parameter to the filenames
            my_fname = my_fname + '_' + var + '_' + str(val)
            # Update history tree
            if scandim == ONE:
                iparam_all = iparam
            if scandim == TWO:
                iparam_all = len(scan[ONE]) + iparam
            valtree[iparam_all] = val
        
        # If we are at the bottom of the tree, then read from the file.
        if scandim == ndim:
            [tt0,ky,gamma,omega,Qratio_avg] = read_from_file(my_fname)
            tt0_collec.append(tt0)
            ky_collec.append(ky)
            gamma_collec.append(gamma)
            omega_collec.append(omega)
            qe_vs_qi_collec.append(Qratio_avg)
        # Or move on to the next dimension of the scan by calling function recursively
        else:
            next_scandim = increment_dim(scandim)
            read_data(my_fname, valtree, next_scandim, tt0_collec, ky_collec, gamma_collec, omega_collec, qe_vs_qi_collec)

def read_from_file(fname):

    datfile_name = 'postproc/' + fname + '.linrange.dat' ### USER ###

    with open(datfile_name, 'rb') as infile: # 'rb' stands for read bytes
        my_vars = pickle.load(infile)

    return my_vars

def write_to_file(vardict):

    datfile_name = 'postproc/linscan.dat'
    with open(datfile_name, 'wb') as outfile: # 'wb' stands for write bytes
        pickle.dump(vardict,outfile)

# Execute main
if __name__ == '__main__':
    main()
