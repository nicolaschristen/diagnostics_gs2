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
ndim = TWO

# Dict that will contain every parameter to scan
scan = {ONE:[],TWO:[]}

# Import all parameters from paramfiles/myfile.py
pf = __import__('ntheta_ky_scan_085_new') ### USER ###

def main():

    base_name = 'psin_0.85' ### USER ###
    my_ylim = (-0.05, 0.3) ### USER ###

    # Add all parameters to the current scan
    nparams = len(pf.name)
    for iparam in range(nparams):
        add_param_to_scan(pf.name[iparam], pf.val[iparam], pf.namelist[iparam], scan, pf.scandim[iparam])

    # Read data from .dat files for every file in scan
    var_firstdim = scan[ONE][0].name
    val_firstdim = scan[ONE][0].value
    ky_collec = []
    gamma_avg_collec = []
    gamma_max_collec = []
    scandim=ONE
    ival_firstdim = -1
    read_data(base_name, scandim, ival_firstdim, ky_collec, gamma_avg_collec, gamma_max_collec)

    # Save data for whole scan in file
    vardict = {}
    vardict['var_firstdim'] = var_firstdim
    vardict['val_firstdim'] = val_firstdim
    vardict['ky_collec'] = ky_collec
    vardict['gamma_avg_collec'] = gamma_avg_collec
    vardict['gamma_max_collec'] = gamma_max_collec
    write_to_file(vardict)

    # Plotting
    color_collec = plt.cm.gnuplot_r(np.linspace(0,1,val_firstdim.size)) # for newalgo

    pdfname = 'postproc/flowshear_scan2D_gamma_avg.pdf'
    plt.figure()
    my_legend = []
    plt.grid(True)
    plt.xlabel('$\\rho k_y$')
    plt.ylabel('$\\langle\\gamma\\rangle_t [v_{th}/a]$')
    for ival in range(val_firstdim.size):
        plt.plot(ky_collec[ival], gamma_avg_collec[ival], linewidth=2.0, color=color_collec[ival])
        my_legend.append('$N_\\theta = {:d}$'.format(val_firstdim[ival]))
    plt.ylim(my_ylim)
    plt.legend(my_legend)
    plt.savefig(pdfname)

    pdfname = 'postproc/flowshear_scan2D_gamma_max.pdf'
    plt.figure()
    my_legend = []
    plt.grid(True)
    plt.xlabel('$\\rho k_y$')
    plt.ylabel('$\\gamma_{max} [v_{th}/a]$')
    for ival in range(val_firstdim.size):
        plt.plot(ky_collec[ival], gamma_max_collec[ival], linewidth=2.0, color=color_collec[ival])
        my_legend.append('$N_\\theta = {:d}$'.format(val_firstdim[ival]))
    plt.ylim(my_ylim)
    plt.legend(my_legend)
    plt.savefig(pdfname)

class gs2_param:
    def __init__(self, var='', val=np.array([]), in_list=''):
        self.name = var
        self.value = val
        self.namelist = in_list

def add_param_to_scan(name, values, namelist, scan, scandim):
    newparam = gs2_param(name, values, namelist)
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

def read_data(fname, scandim, ival_firstdim, ky_collec, gamma_avg_collec, gamma_max_collec):

    # Iterate over every set of values taken by parameters in this dimension of the scan.
    for ival in range(scan[scandim][0].value.size):
    
        # Name-base and variable container for this ival
        my_fname = fname
        #ky_collec = cp.deepcopy(ky_collec)
        #gamma_avg_collec = cp.deepcopy(gamma_avg_collec)
        #gamma_max_collec = cp.deepcopy(gamma_max_collec)

        if scandim==ONE and ival_firstdim!=ival:
            ival_firstdim = ival
            ky_collec.append([])
            gamma_avg_collec.append([])
            gamma_max_collec.append([])

        for iparam in range(len(scan[scandim])):
            var = scan[scandim][iparam].name
            val = scan[scandim][iparam].value[ival]
            # Append parameter to the filenames
            my_fname = my_fname + '_' + var + '_' + str(val)
        
        # If we are at the last dimension of the scan, then read from the file.
        if scandim == ndim:
            my_vars = read_from_file(my_fname)
            ky_collec[ival_firstdim].append(my_vars['ky'][0])
            gamma_avg_collec[ival_firstdim].append(my_vars['gamma_avg'][0])
            gamma_max_collec[ival_firstdim].append(my_vars['gamma_max'][0])

        # Or move on to the next dimension of the scan by calling function recursively
        else:
            next_scandim = increment_dim(scandim)
            read_data(my_fname, next_scandim, ival_firstdim, ky_collec, gamma_avg_collec, gamma_max_collec)

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
