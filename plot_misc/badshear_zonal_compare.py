import sys
import os

# Add path to directory where pygs2 files are stored
maindir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.insert(0, maindir)

import pickle
from matplotlib import pyplot as plt
import gs2_plotting as gplot
import numpy as np

# set plotting defaults
gplot.set_plot_defaults()

ispec = 0 # main ion

dirname = 'postproc/'

fname_new = dirname+'ollie_badshear_fexp_stitch'
fname_old = dirname+'ollie_badshear_old_stitch3'

# Read in flux quantities
with open(fname_new+'.fluxes.dat','rb') as datfile:
    fluxdict_new = pickle.load(datfile)
with open(fname_old+'.fluxes.dat','rb') as datfile:
    fluxdict_old = pickle.load(datfile)

phi2_by_ky_new = fluxdict_new['phi2_by_ky']
phi2_by_ky_old = fluxdict_old['phi2_by_ky']

# Read in time quantities
with open(fname_new+'.time.dat','rb') as datfile:
    mytime_new = pickle.load(datfile)
with open(fname_old+'.time.dat','rb') as datfile:
    mytime_old = pickle.load(datfile)

time_new = mytime_new.time
time_old = mytime_old.time

# Read in grids
with open('postproc/ollie_badshear_fexp_id_1.grids.dat','rb') as datfile:
    mygrids = pickle.load(datfile)

ky = mygrids.ky

##########################
## PLOT PHI2 FOR EVERY KY
##########################

## NEW
clr_new = plt.cm.YlOrBr(np.linspace(0.2,1,4))
clr_old = plt.cm.YlGnBu(np.linspace(0.2,1,4))
my_linewidth = 4.0
my_xlim = [mytime_new.time[0],max(mytime_new.time[-1],mytime_old.time[-1])]
my_ylim = [1.e-6, 1.e3]
my_xlabel = '$t [a/v_{th,i}]$'
my_ylabel = '$\\vert \\langle \\hat{\\varphi}\\rangle _{\\theta,k_x}\\vert ^2$'

fig = plt.figure(figsize=(10,10))

plt.semilogy(time_new,phi2_by_ky_new[:,0],label='$\\rho k_y = {:.2f}$'.format(ky[0]),linestyle='dashed',
        color=clr_new[3],linewidth=my_linewidth)
plt.semilogy(time_new,phi2_by_ky_new[:,1],label='$\\rho k_y = {:.2f}$'.format(ky[1]),
        color=clr_new[2],linewidth=my_linewidth)

plt.xlabel(my_xlabel,FontSize='38')
plt.ylabel(my_ylabel,FontSize='40')
plt.xlim(my_xlim)
plt.ylim(my_ylim)
plt.grid(True)

my_legend = plt.legend(frameon=True,fancybox=False,framealpha=1.0,loc='lower right',prop={'size': 35})
my_legend.get_frame().set_facecolor('w')
my_legend.get_frame().set_edgecolor('k')
my_legend.get_frame().set_linewidth(1.0)

filename = dirname + 'badshear_zonal_new' + '.pdf'
plt.savefig(filename)

## OLD
fig = plt.figure(figsize=(10,10))

plt.semilogy(time_old,phi2_by_ky_old[:,0],label='$\\rho k_y = {:.2f}$'.format(ky[0]),linestyle='dashed',
        color=clr_old[3],linewidth=my_linewidth)
plt.semilogy(time_old,phi2_by_ky_old[:,1],label='$\\rho k_y = {:.2f}$'.format(ky[1]),
        color=clr_old[2],linewidth=my_linewidth)

plt.xlabel(my_xlabel,FontSize='38')
plt.ylabel(my_ylabel,FontSize='40')
plt.xlim(my_xlim)
plt.ylim(my_ylim)
plt.grid(True)

my_legend = plt.legend(frameon=True,fancybox=False,framealpha=1.0,loc='lower right',prop={'size': 35})
my_legend.get_frame().set_facecolor('w')
my_legend.get_frame().set_edgecolor('k')
my_legend.get_frame().set_linewidth(1.0)

filename = dirname + 'badshear_zonal_old' + '.pdf'
plt.savefig(filename)
