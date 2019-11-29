import sys
import os

# Add path to directory where scan-files are stored
sys.path.insert(1, '/marconi/home/userexternal/nchriste/codes/scan_gs2/paramfiles')

import f90nml as fnml
import numpy as np
import copy as cp
import pickle
import matplotlib.pyplot as plt

ONE = 'one'
TWO = 'two'
NDIM_MAX = TWO
ndim = ONE

# Dict that will contain every parameter to scan
scan = {ONE:[],TWO:[]}

# Import all parameters from paramfiles/myfile.py
pf = __import__('ky_scan_psin_085_new') ### USER ###

def main():

    base_name = 'psin_0.85' ### USER ###

    # Add all parameters to the current scan
    nparams = len(pf.name)
    for iparam in range(nparams):
        add_param_to_scan(pf.name[iparam], pf.val[iparam], pf.namelist[iparam], scan, pf.scandim[iparam])

    # Read data from .dat files for every file in scan
    ky = []
    gamma_avg = []
    gamma_max = []
    scandim=ONE
    read_data(base_name, scandim, ky, gamma_avg, gamma_max)

    # Save data for whole scan in file
    vardict = {}
    vardict['ky'] = ky
    vardict['gamma_avg'] = gamma_avg
    vardict['gamma_max'] = gamma_max
    write_to_file(vardict)

    pdfname = 'postproc/flowshear_lingrowth.pdf'
    plt.figure()
    my_legend = []
    plt.grid(True)
    plt.xlabel('$\\rho k_y$')
    plt.ylabel('$[v_{th}/a]$')
    plt.plot(ky, gamma_avg, linewidth=3.0, color='k', linestyle=':')
    my_legend.append('$\\langle\\gamma\\rangle_t$')
    plt.plot(ky, gamma_max, linewidth=3.0, color='k', linestyle='--')
    my_legend.append('$\\gamma_{max}$')
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

def read_data(fname, scandim, ky, gamma_avg, gamma_max):

    # Iterate over every set of values taken by parameters in this dimension of the scan.
    for ival in range(scan[scandim][0].value.size):
    
        # Name-base and patch to be modified for this ival.
        my_fname = fname

        for iparam in range(len(scan[scandim])):
            var = scan[scandim][iparam].name
            val = scan[scandim][iparam].value[ival]
            # Append parameter to the filenames
            my_fname = my_fname + '_' + var + '_' + str(val)
        
        # If we are at the last dimension of the scan, then read from the file.
        if scandim == ndim:
            my_vars = read_from_file(my_fname)
            ky.append(my_vars['ky'][0])
            gamma_avg.append(my_vars['gamma_avg'][0])
            gamma_max.append(my_vars['gamma_max'][0])

        # Or move on to the next dimension of the scan by calling function recursively
        else:
            next_scandim = increment_dim(scandim)
            modify_files(my_fname, next_scandim)

def read_from_file(fname):

    datfile_name = 'postproc/' + fname + '.flowshear_lingrowth.dat' ### USER ###

    with open(datfile_name, 'rb') as infile: # 'rb' stands for read bytes
        my_vars = pickle.load(infile)

    return my_vars

def write_to_file(vardict):

    datfile_name = 'postproc/flowshear_lingrowth_kyscan.dat'
    with open(datfile_name, 'wb') as outfile: # 'wb' stands for write bytes
        pickle.dump(vardict,outfile)

# Execute main
if __name__ == '__main__':
    main()
