import sys
import os

# Add path to directory where scan-files are stored
taskdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'paramfiles')
sys.path.insert(0, taskdir)

import f90nml as fnml
import numpy as np
import copy as cp
import fileinput as fin
from shutil import copyfile
import subprocess
import run_parameters as runpar
import importlib

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

# From command-line arguments, get info about this analysis run (filenames, tasks to complete ...)
run = runpar.runobj()

# Import all parameters from paramfiles/myfile.py
pf = __import__(run.paramfile)

def main():

    base_name = 'psin_0.85' ### USER ###
    firstdim_var = 'ky' ### USER ###
    firstdim = pf.ky ### USER ###
    seconddim_var = 'jtwist' ### USER ###
    seconddim = pf.jtwist ### USER ###

    my_ylim = (-0.05, 0.3) ### USER ###

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
    read_data(base_name, valtree, scandim, ival_firstdim,ival_firstdim, firstdim_collec, gamma_avg_collec, gamma_max_collec)

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
        plt.plot(firstdim_collec[0], gamma_avg_collec[0], linewidth=3.0, color='k', linestyle=':')
        plt.savefig(pdfname)

        pdfname = 'postproc/flowshear_gamma_max_scan_'+firstdim_var+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel('$\\rho k_y$') ### USER ###
        plt.ylabel('$\\gamma_{max} [v_{th}/a]$')
        plt.plot(firstdim_collec[0], gamma_max_collec[0], linewidth=3.0, color='k', linestyle=':')
        plt.savefig(pdfname)

    elif ndim == TWO:

        color_collec = plt.cm.gnuplot_r(np.linspace(0,1,seconddim.size))

        pdfname = 'postproc/flowshear_gamma_avg_scan_'+firstdim_var+'_'+seconddim_var+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel('$\\rho k_y$') ### USER ###
        plt.ylabel('$\\langle\\gamma\\rangle_t [v_{th}/a]$')
        for ival in range(seconddim.size):
            plt.plot(firstdim_collec[ival], gamma_avg_collec[ival], linewidth=2.0, color=color_collec[ival])
            my_legend.append('$N_\\theta = {:d}$'.format(val_firstdim[ival])) ### USER ###
        plt.ylim(my_ylim)
        plt.legend(my_legend)
        plt.savefig(pdfname)

        pdfname = 'postproc/flowshear_gamma_max_scan_'+firstdim_var+'_'+seconddim_var+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel('$\\rho k_y$') ### USER ###
        plt.ylabel('$\\gamma_{max} [v_{th}/a]$')
        for ival in range(seconddim.size):
            plt.plot(firstdim_collec[ival], gamma_avg_collec[ival], linewidth=2.0, color=color_collec[ival])
            my_legend.append('$N_\\theta = {:d}$'.format(val_firstdim[ival])) ### USER ###
        plt.ylim(my_ylim)
        plt.legend(my_legend)
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
