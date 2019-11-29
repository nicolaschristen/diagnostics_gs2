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

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=20)
rcParams.update({'figure.autolayout': True})
rcParams.update({'legend.fontsize': 12, 'legend.handlelength': 4})
rcParams.update({'legend.frameon': False})

ONE = 'one'
TWO = 'two'
NDIM_MAX = TWO
ndim = ONE ### USER ###

# Dict that will contain every parameter to scan
scan = {ONE:[],TWO:[]}

# Import all parameters from paramfiles/myfile.py
pf = __import__('scan_ky_fixed_dkx_kxmax_ijp_950_rpsi_051_final_res') ### USER ###

base_name = 'rpsi_0.51' ### USER ###
firstdim_var = 'ky' ### USER ###
firstdim = pf.ky ### USER ###
seconddim_var = 'negrid' ### USER ###
#seconddim = (2*np.round((pf.nx-1)/3)+1).astype(int) ### USER ###
seconddim = pf.ky ### USER ###
plot_converged = False ### USER ###
firstdim_converged = np.array([0.2, 0.5, 1.0]) ### USER ###
gamma_max_converged = np.array([0.05, 0.109, 0.008]) # 950, rpsi=0.51, kx_max_scan
gamma_avg_converged = np.array([-0.002, -0.003, -0.001]) # 950, rpsi=0.51, kx_max_scan

# Original code was written for ky=seconddim (invert_dims=True)
invert_dims = False ### USER ###

use_my_ylim = False ### USER ###
my_ylim_max = (0.0, 0.25) ### USER ###
my_ylim_avg = (-0.015, 0.025) ### USER ###

use_my_xlim = True ### USER ###
my_xlim = (0.0, 1.1) ### USER ###

scan_with_single_ky = False ### USER ###


def main():

    # Add all parameters to the current scan
    nparams = len(pf.name)
    for iparam in range(nparams):
        add_param_to_scan(scan, pf.name[iparam], pf.dim[iparam], pf.namelist[iparam], pf.scandim[iparam], pf.func[iparam])

    # Read data from .dat files for every file in scan
    firstdim_collec = []
    gamma_avg_collec = []
    gamma_max_collec = []
    scandim=ONE
    ival_firstdim = -1
    valtree = [0*i for i in range(nparams)]
    read_data(base_name, valtree, scandim, ival_firstdim, firstdim_collec, gamma_avg_collec, gamma_max_collec)

    # Re-organise data if ky=firstdim
    if not invert_dims and ndim == TWO:

        firstdim_collec_new = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        gamma_avg_collec_new = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        gamma_max_collec_new = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        for ival_first in range(firstdim.size):
            for ival_scnd in range(seconddim.size):
                firstdim_collec_new[ival_scnd][ival_first] = firstdim_collec[ival_first][ival_scnd]
                gamma_avg_collec_new[ival_scnd][ival_first] = gamma_avg_collec[ival_first][ival_scnd]
                gamma_max_collec_new[ival_scnd][ival_first] = gamma_max_collec[ival_first][ival_scnd]
        firstdim_collec = firstdim_collec_new
        gamma_avg_collec = gamma_avg_collec_new
        gamma_max_collec = gamma_max_collec_new

    # Save data for whole scan in file
    vardict = {}
    vardict['firstdim_collec'] = firstdim_collec
    vardict['gamma_avg_collec'] = gamma_avg_collec
    vardict['gamma_max_collec'] = gamma_max_collec
    write_to_file(vardict)

    # Plotting
    if ndim == ONE:

        pdfname = 'postproc/flowshear_gamma_avg_scan_'+firstdim_var+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel('$\\rho k_y$') ### USER ###
        plt.ylabel('$\\langle\\gamma\\rangle_t [v_{th}/a]$')
        plt.plot(firstdim_collec, gamma_avg_collec, linewidth=2.0, color='k')
        if use_my_ylim:
            plt.ylim(my_ylim_avg)
        plt.savefig(pdfname)

        pdfname = 'postproc/flowshear_gamma_max_scan_'+firstdim_var+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel('$\\rho k_y$') ### USER ###
        plt.ylabel('$\\gamma_{max} [v_{th}/a]$')
        plt.plot(firstdim_collec, gamma_max_collec, linewidth=2.0, color='k')
        if use_my_ylim:
            plt.ylim(my_ylim_max)
        plt.savefig(pdfname)

    elif ndim == TWO:

        color_collec = plt.cm.gnuplot_r(np.linspace(0.05,0.9,seconddim.size))
        if scan_with_single_ky:
            color_collec = ['blue']

        pdfname = 'postproc/flowshear_gamma_avg_scan_'+firstdim_var+'_'+seconddim_var+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel('$\\rho k_y$') ### USER ###
        #plt.xlabel('jtwist') ### USER ###
        plt.ylabel('$\\langle\\gamma\\rangle_t\ [v_{th}/a]$')
        valdim = seconddim.size
        if scan_with_single_ky:
            valdim = firstdim.size
        for ival in range(valdim):
            if scan_with_single_ky:
                plt.plot(seconddim, gamma_avg_collec, linewidth=2.0, color=color_collec[ival])
            else:
                plt.plot(firstdim_collec[ival], gamma_avg_collec[ival], linewidth=2.0, color=color_collec[ival])
            #my_legend.append('$N_x = {0:d}$'.format(seconddim[ival])) ### USER ###
            #my_legend.append('$\\Delta k_x = {0:1.3f}$'.format(seconddim[ival])) ### USER ###
            #my_legend.append('$\\rho k_y = {0:1.3f}$'.format(seconddim[ival])) ### USER ###
            #my_legend.append('$N_\\theta = {0:d}$'.format(seconddim[ival])) ### USER ###
            my_legend.append('$N_\\varepsilon = {0:d}$'.format(seconddim[ival])) ### USER ###
            #my_legend.append('$v_{max} ='+str(seconddim[ival])+'$') ### USER ###
            #my_legend.append('$N_{\\lambda,trap} = '+str(seconddim[ival])+'$') ### USER, with ntheta=32 ###
        if use_my_ylim:
            plt.ylim(my_ylim_avg)
        if use_my_xlim:
            plt.xlim(my_xlim)
        if plot_converged:
            plt.plot(firstdim_converged, gamma_avg_converged, linestyle = 'None', color='k', marker = 's', fillstyle = 'full')
            my_legend.append('$N_x = 535$') ### USER ###
        legend = plt.legend(my_legend, frameon = True, fancybox = False)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        frame.set_linewidth(0.5)
        frame.set_alpha(1)
        plt.savefig(pdfname)

        pdfname = 'postproc/flowshear_gamma_max_scan_'+firstdim_var+'_'+seconddim_var+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel('$\\rho k_y$') ### USER ###
        #plt.xlabel('jtwist') ### USER ###
        plt.ylabel('$\\gamma_{max}\ [v_{th}/a]$')
        for ival in range(valdim):
            if scan_with_single_ky:
                plt.plot(seconddim, gamma_max_collec, linewidth=2.0, color=color_collec[ival])
            else:
                plt.plot(firstdim_collec[ival], gamma_max_collec[ival], linewidth=2.0, color=color_collec[ival])
            #my_legend.append('$N_x = {0:d}$'.format(seconddim[ival])) ### USER ###
            #my_legend.append('$\\Delta k_x = {0:1.3f}$'.format(seconddim[ival])) ### USER ###
            #my_legend.append('$\\rho k_y = {0:1.3f}$'.format(seconddim[ival])) ### USER ###
            #my_legend.append('$N_\\theta = {0:d}$'.format(seconddim[ival])) ### USER ###
            my_legend.append('$N_\\varepsilon = {0:d}$'.format(seconddim[ival])) ### USER ###
            #my_legend.append('$v_{max} ='+str(seconddim[ival])+'$') ### USER ###
            #my_legend.append('$N_{\\lambda,trap} = '+str(seconddim[ival])+'$') ### USER, with ntheta=32 ###
        if use_my_ylim:
            plt.ylim(my_ylim_max)
        if use_my_xlim:
            plt.xlim(my_xlim)
        if plot_converged:
            plt.plot(firstdim_converged, gamma_max_converged, linestyle = 'None', color='k', marker = 's', fillstyle = 'full')
            my_legend.append('$N_x = 535$') ### USER ###
        legend = plt.legend(my_legend, frameon = True, fancybox = False)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        frame.set_linewidth(0.5)
        frame.set_alpha(1)
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

def read_data(fname, valtree, scandim, ival_firstdim, firstdim_collec, gamma_avg_collec, gamma_max_collec):

    # Iterate over every set of values taken by parameters in this dimension of the scan.
    for ival in range(scan[scandim][0].dim):
    
        # Name-base and patch to be modified for this ival.
        my_fname = fname

        # For every new value of firstdim, append elements
        if scandim==ONE and ival_firstdim!=ival:
            ival_firstdim = ival
            firstdim_collec.append([])
            gamma_avg_collec.append([])
            gamma_max_collec.append([])

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
            my_vars = read_from_file(my_fname)
            firstdim_collec[ival_firstdim].append(my_vars[firstdim_var][0])
            gamma_avg_collec[ival_firstdim].append(my_vars['gamma_avg'][0])
            gamma_max_collec[ival_firstdim].append(my_vars['gamma_max'][0])
        # Or move on to the next dimension of the scan by calling function recursively
        else:
            next_scandim = increment_dim(scandim)
            read_data(my_fname, valtree, next_scandim, ival_firstdim, firstdim_collec, gamma_avg_collec, gamma_max_collec)

def read_from_file(fname):

    datfile_name = 'postproc/' + fname + '.flowshear_lingrowth.dat' ### USER ###

    with open(datfile_name, 'rb') as infile: # 'rb' stands for read bytes
        my_vars = pickle.load(infile)

    return my_vars

def write_to_file(vardict):

    datfile_name = 'postproc/flowshear_lingrowth_scan2D.dat'
    with open(datfile_name, 'wb') as outfile: # 'wb' stands for write bytes
        pickle.dump(vardict,outfile)

# Execute main
if __name__ == '__main__':
    main()
