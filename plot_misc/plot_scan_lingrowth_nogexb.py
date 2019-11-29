import sys
import os

# Add path to directory where scan-files are stored
sys.path.insert(1, '/marconi/home/userexternal/nchriste/codes/scan_gs2/paramfiles')

import f90nml as fnml
import numpy as np
import copy as cp
import pickle
import matplotlib.pyplot as plt

#plt.rcParams.update({'font.size': 22})

ONE = 'one'
TWO = 'two'
NDIM_MAX = TWO
ndim = ONE

# Dict that will contain every parameter to scan
scan = {ONE:[],TWO:[]}

# Import all parameters from paramfiles/myfile.py
pf = __import__('ntheta_scan_nogexb_085_new') ### USER ###

def main():

    base_name = 'psin_0.85' ### USER ###

    # Add all parameters to the current scan
    nparams = len(pf.name)
    for iparam in range(nparams):
        add_param_to_scan(pf.name[iparam], pf.val[iparam], pf.namelist[iparam], scan, pf.scandim[iparam])

    # Read data from .dat files for every file in scan
    var_firstdim = scan[ONE][0].name
    val_firstdim = scan[ONE][0].value
    ky_collec = []
    gamma_collec = []
    scandim=ONE
    ival_firstdim = -1
    read_data(base_name, scandim, ival_firstdim, ky_collec, gamma_collec)

    # Plotting
    color_collec = plt.cm.gnuplot(np.linspace(0,1,val_firstdim.size)) # for newalgo

    pdfname = 'postproc/noflowshear_gamma.pdf'
    plt.figure()
    my_legend = []
    plt.grid(True)
    plt.xlabel('$\\rho k_y$')
    plt.ylabel('$\\gamma [v_{th}/a]$')
    for ival in range(val_firstdim.size):
        ky_collec[ival] = np.squeeze(ky_collec[ival][0])
        gamma_collec[ival] = np.squeeze(gamma_collec[ival][0])
        plt.plot(ky_collec[ival], gamma_collec[ival], linewidth=3.0, color=color_collec[ival])
        my_legend.append('$N_\\theta = {:d}$'.format(val_firstdim[ival]))
    plt.legend(my_legend)
    plt.savefig(pdfname)

    # Save data for whole scan in file
    vardict = {}
    vardict['var_firstdim'] = var_firstdim
    vardict['val_firstdim'] = val_firstdim
    vardict['ky_collec'] = ky_collec
    vardict['gamma_collec'] = gamma_collec
    write_to_file(vardict)

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

def read_data(fname, scandim, ival_firstdim, ky_collec, gamma_collec):

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
            gamma_collec.append([])

        for iparam in range(len(scan[scandim])):
            var = scan[scandim][iparam].name
            val = scan[scandim][iparam].value[ival]
            # Append parameter to the filenames
            my_fname = my_fname + '_' + var + '_' + str(val)
        
        # If we are at the last dimension of the scan, then read from the file.
        if scandim == ndim:
            ky, gamma = read_from_file(my_fname)
            ky_collec[ival_firstdim].append(ky)
            gamma_collec[ival_firstdim].append(gamma)

        # Or move on to the next dimension of the scan by calling function recursively
        else:
            next_scandim = increment_dim(scandim)
            read_data(my_fname, next_scandim, ival_firstdim, ky_collec, gamma_collec)

def read_from_file(fname):

    datfile_name = 'postproc/' + fname + '.lingrowth.dat' ### USER ###

    with open(datfile_name, 'rb') as infile: # 'rb' stands for read bytes
        dummy1, dummy2, ky, gamma, dummy3  = pickle.load(infile)

    return ky, gamma

def write_to_file(vardict):

    datfile_name = 'postproc/noflowshear_lingrowth_scan.dat'
    with open(datfile_name, 'wb') as outfile: # 'wb' stands for write bytes
        pickle.dump(vardict,outfile)

# Execute main
if __name__ == '__main__':
    main()