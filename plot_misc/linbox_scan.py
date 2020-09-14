import sys
import os

# Add path to directory where scan-files are stored
sys.path.insert(1, '/marconi/home/userexternal/nchriste/codes/scan_gs2/paramfiles')
# Add path to directory where gs2_plotting is stored
sys.path.insert(1, '/marconi/home/userexternal/nchriste/codes/diagnostics_gs2')

import f90nml as fnml
import numpy as np
import copy as cp
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
from math import pi
import scipy.interpolate as scinterp
import scipy.optimize as opt
import gs2_plotting as gplot
from PyPDF2 import PdfFileMerger, PdfFileReader

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






# vvv USER PARAMETERS vvv

# Import all parameters from paramfiles/myfile.py
base_name = 'rpsi_0.8'  
pf = __import__('scan_tprimi_linbox_gexb_vs_ky_theta0_ijp_950_rpsi_08')  

# Number of dimensions in the scan
# e.g. vs (ky, R/LTi) -> ndim = TWO
ndim = TWO

# Define which dmid to use when plotting quant vs ky.
# dmid is the number of dkx between kx=0 and the
# smallest kx>0 that is a member of a particular
# twist and shift chain at t=0, ie it identifies that chain.
# NB: if None is selected, growthrates are averaged over all chains
dmid_for_plots_vs_ky = None

# Save data for a scan in (tprimi,gexb)
sav_for_tprimigexb_scan = False
tprimi_orig = 1.7392
fac_tprimi = 0.6

# Define first dimension of the scan
firstdim_label = '$\\rho_i k_y$' # for plotting
firstdim_var = 'ky' # name of variable to append to figure names
firstdim = pf.ky # variable name in paramfiles/myfile.py

# Define second dimension of the scan
if ndim == TWO:

    #seconddim_label = '$N_x$'
    #seconddim_label = '$N_{2\\pi}$'
    #seconddim_label = '$k_{x,max}$'
    #seconddim_label = '$\\Delta k_x$'
    #seconddim_label = '$\\rho k_y$'
    #seconddim_label = '$N_\\theta$'
    #seconddim_label = '$N_\\varepsilon$'
    #seconddim_label = '$v_{max}$'
    #seconddim_label = '$N_{\\lambda,untrap}$'
    seconddim_label = '$a/L_{T_i}$'
    #seconddim_label = '$\\gamma_E$'
    #seconddim_label = '$\\Delta t$ [$a/v_{th,i}$]'

    seconddim_var = 'tprimi'  

    #seconddim = (2*np.round((pf.nx-1)/3)+1).astype(int) # nakx
    #seconddim = np.round((pf.nx-1)/3) * pf.dkx  # kxmax
    #seconddim = np.round((2*np.round((pf.nx-1)/3))/(2*pi*pf.shat*firstdim[-1]/pf.dkx)+1,2) # Ntwopi
    seconddim = pf.tprimi

elif ndim == ONE:

    seconddim_label = 'dummy'
    seconddim_var = 'dummy'
    seconddim = np.array([0])

# Does the scan have only a single ky ?
scan_with_single_ky = False  

# Apply limits to axis when plotting ?
use_my_xlim = True
my_xlim = (0.0, None)  

use_my_ylim = True
#### rpsi = 0.5, Batch 1 gexb
#~ my_ylim_max = (0.0, 0.15)
#~ my_ylim_max_gamoverksq = (0.0, 0.75)
#~ my_ylim_avg = (-0.035, 0.020)
#~ my_ylim_avg_gamoverksq = (-0.1, 0.15)
#### rpsi = 0.5, Batch 2,3,4 gexb
#~ my_ylim_max = (0.0, 0.3)
#~ my_ylim_max_gamoverksq = (0.0, 0.75)
#~ my_ylim_avg = (-0.035, 0.020)
#~ my_ylim_avg_gamoverksq = (-0.1, 0.15)
#### rpsi = 0.5, Batch 6 gexb
#~ my_ylim_max = (0.0, 0.3)
#~ my_ylim_max_gamoverksq = (0.0, 0.75)
#~ my_ylim_avg = (-0.02, 0.1)
#~ my_ylim_avg_gamoverksq = (-0.1, 0.15)
#### rpsi = 0.5, Batch 2 tprimi
#~ my_ylim_max = (0.0, 0.3)
#~ my_ylim_max_gamoverksq = (0.0, 0.75)
#~ my_ylim_avg = (-0.02, 0.1)
#~ my_ylim_avg_gamoverksq = (-0.1, 0.15)
#### rpsi = 0.6
#~ my_ylim_max = (0.0, 0.2)
#~ my_ylim_max_gamoverksq = (0.0, 0.85)
#~ my_ylim_avg = (-0.035, 0.02)
#~ my_ylim_avg_gamoverksq = (-0.1, 0.1)
#### rpsi = 0.7
#~ my_ylim_max = (0.0, 0.40)
#~ my_ylim_max_gamoverksq = (0.0, 0.85)
#~ my_ylim_avg = (-0.02, 0.09)
#~ my_ylim_avg_gamoverksq = (-0.1, 0.1)
#### rpsi = 0.8
#~ my_ylim_max = (0.0, 0.50)
#~ my_ylim_max_gamoverksq = (0.0, 0.85)
#~ my_ylim_avg = (-0.03, 0.03)
#~ my_ylim_avg_gamoverksq = (-0.1, 0.1)
#### rpsi = 0.9
#~ my_ylim_max = (0.0, 0.30)
#~ my_ylim_avg = (-0.06, 0.00)

my_ylim_max = (0.0, 0.3)
my_ylim_max_gamoverksq = (0.0, 0.75)
my_ylim_avg = (-0.03, 0.02)
my_ylim_avg_gamoverksq = (-0.1, 0.15)

# For cases without flow shear (otherwise comment out):
my_ylim_avg = my_ylim_max

# Fix colorbar limits ?
fix_cbarlim = True
# rpsi = 0.5
#my_cbarmin = -0.20
#my_cbarmax = 0.20
# rpsi = 0.6
my_cbarmin = -0.20
my_cbarmax = 0.2
# rpsi = 0.7
#my_cbarmin = -0.30
#my_cbarmax = 0.3
# rpsi = 0.8
#my_cbarmin = -0.25
#my_cbarmax = 0.25
# rpsi = 0.9
#my_cbarmin = -0.30
#my_cbarmax = 0.3

# Original code was written for kyas second dim of scan (not first).
# To restore this, set invert_dims = True
invert_dims = False  

# ^^^ USER PARAMETERS ^^^






def main():

    # Logical testing if instantaneous growthrate was computed
    gamma_inst_pres = True

    if scan_with_single_ky:
        valdim = firstdim.size
    else:
        valdim = seconddim.size

    # Add all parameters to the current scan
    nparams = len(pf.name)
    for iparam in range(nparams):
        add_param_to_scan(scan, pf.name[iparam], pf.dim[iparam], pf.namelist[iparam], pf.scandim[iparam], pf.func[iparam])

    # Read data from .dat files for every file in scan
    g_exb_vs_v2_ky_tt0 = []
    dmid_list_vs_v2_ky_tt0 = []
    itheta0_list_vs_v2_ky_tt0 = []
    theta0_vs_v2_ky_tt0 = []
    theta0_star_for_inst_vs_v2_ky_tt0 = []
    firstdim_vs_v2_ky_tt0 = []
    gamma_avg_vs_v2_ky_tt0 = []
    gamma_avg_fromSum_vs_v2_ky_tt0 = []
    gamma_max_vs_v2_ky_tt0 = []
    gamma_max_fromSum_vs_v2_ky_tt0 = []
    gamma_inst_vs_v2_ky_tt0 = []
    gamma_inst_fromSum_vs_v2_ky_tt0 = []
    Qratio_avg_vs_v2_ky_tt0 = []
    scandim=ONE
    ival_firstdim = -1
    valtree = [0*i for i in range(nparams)]
    read_data(base_name, valtree, scandim, ival_firstdim, dmid_list_vs_v2_ky_tt0, itheta0_list_vs_v2_ky_tt0, theta0_vs_v2_ky_tt0, \
            firstdim_vs_v2_ky_tt0, gamma_avg_vs_v2_ky_tt0, gamma_avg_fromSum_vs_v2_ky_tt0, gamma_max_vs_v2_ky_tt0, \
             gamma_max_fromSum_vs_v2_ky_tt0, Qratio_avg_vs_v2_ky_tt0, g_exb_vs_v2_ky_tt0, \
            gamma_inst_vs_v2_ky_tt0, gamma_inst_fromSum_vs_v2_ky_tt0, theta0_star_for_inst_vs_v2_ky_tt0)
    if not gamma_inst_vs_v2_ky_tt0[0]:
        gamma_inst_pres = False




    # Re-organise data

    if not invert_dims:

        # Arrays for data with all theta0
        dmid_list_vs_v2_ky_tt0_new = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        itheta0_list_vs_v2_ky_tt0_new = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        theta0_vs_v2_ky_tt0_new = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        firstdim_vs_v2_ky_tt0_new = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        gamma_avg_vs_v2_ky_tt0_new = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        gamma_avg_fromSum_vs_v2_ky_tt0_new = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        gamma_max_vs_v2_ky_tt0_new = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        gamma_max_fromSum_vs_v2_ky_tt0_new = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        Qratio_avg_vs_v2_ky_tt0_new = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        g_exb_vs_v2_ky_tt0_new = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        if gamma_inst_pres:
            gamma_inst_vs_v2_ky_tt0_new = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
            gamma_inst_fromSum_vs_v2_ky_tt0_new = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
            theta0_star_for_inst_vs_v2_ky_tt0_new = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]

        # Arrays for data with only theta0 = 0
        firstdim_vs_v2_ky = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        gamma_avg_vs_v2_ky = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        gamma_avg_fromSum_vs_v2_ky = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        gamma_max_vs_v2_ky = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        gamma_max_fromSum_vs_v2_ky = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        Qratio_avg_vs_v2_ky = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]
        g_exb_vs_v2_ky = [[0*i*j for i in range(firstdim.size)] for j in range(seconddim.size)]

        for ival_first in range(firstdim.size):

            for ival_scnd in range(seconddim.size):

                # Rearrange keeping all theta0
                dmid_list_vs_v2_ky_tt0_new[ival_scnd][ival_first] = dmid_list_vs_v2_ky_tt0[ival_first][ival_scnd]
                itheta0_list_vs_v2_ky_tt0_new[ival_scnd][ival_first] = itheta0_list_vs_v2_ky_tt0[ival_first][ival_scnd]
                theta0_vs_v2_ky_tt0_new[ival_scnd][ival_first] = theta0_vs_v2_ky_tt0[ival_first][ival_scnd]
                firstdim_vs_v2_ky_tt0_new[ival_scnd][ival_first] = firstdim_vs_v2_ky_tt0[ival_first][ival_scnd]
                gamma_avg_vs_v2_ky_tt0_new[ival_scnd][ival_first] = gamma_avg_vs_v2_ky_tt0[ival_first][ival_scnd]
                gamma_avg_fromSum_vs_v2_ky_tt0_new[ival_scnd][ival_first] = gamma_avg_fromSum_vs_v2_ky_tt0[ival_first][ival_scnd]
                gamma_max_vs_v2_ky_tt0_new[ival_scnd][ival_first] = gamma_max_vs_v2_ky_tt0[ival_first][ival_scnd]
                gamma_max_fromSum_vs_v2_ky_tt0_new[ival_scnd][ival_first] = gamma_max_fromSum_vs_v2_ky_tt0[ival_first][ival_scnd]
                Qratio_avg_vs_v2_ky_tt0_new[ival_scnd][ival_first] = Qratio_avg_vs_v2_ky_tt0[ival_first][ival_scnd]
                g_exb_vs_v2_ky_tt0_new[ival_scnd][ival_first] = g_exb_vs_v2_ky_tt0[ival_first][ival_scnd]
                if gamma_inst_pres:
                    gamma_inst_vs_v2_ky_tt0_new[ival_scnd][ival_first] = gamma_inst_vs_v2_ky_tt0[ival_first][ival_scnd]
                    gamma_inst_fromSum_vs_v2_ky_tt0_new[ival_scnd][ival_first] = gamma_inst_fromSum_vs_v2_ky_tt0[ival_first][ival_scnd]
                    theta0_star_for_inst_vs_v2_ky_tt0_new[ival_scnd][ival_first] = theta0_star_for_inst_vs_v2_ky_tt0[ival_first][ival_scnd]

                firstdim_vs_v2_ky[ival_scnd][ival_first] = firstdim_vs_v2_ky_tt0[ival_first][ival_scnd]
                Qratio_avg_vs_v2_ky[ival_scnd][ival_first] = Qratio_avg_vs_v2_ky_tt0[ival_first][ival_scnd]
                g_exb_vs_v2_ky[ival_scnd][ival_first] = g_exb_vs_v2_ky_tt0[ival_first][ival_scnd]
                
                if dmid_for_plots_vs_ky is not None:

                    # Find the correct chain
                    idmid = 0
                    try:
                        while dmid_list_vs_v2_ky_tt0[ival_first][ival_scnd][idmid] != dmid_for_plots_vs_ky:
                            idmid += 1
                    except:
                        idmid = 0

                    # Rearrange keeping only theta0 = 0
                    gamma_avg_vs_v2_ky[ival_scnd][ival_first] = gamma_avg_vs_v2_ky_tt0[ival_first][ival_scnd][idmid]
                    gamma_avg_fromSum_vs_v2_ky[ival_scnd][ival_first] = gamma_avg_fromSum_vs_v2_ky_tt0[ival_first][ival_scnd][idmid]
                    gamma_max_vs_v2_ky[ival_scnd][ival_first] = gamma_max_vs_v2_ky_tt0[ival_first][ival_scnd][idmid]
                    gamma_max_fromSum_vs_v2_ky[ival_scnd][ival_first] = gamma_max_fromSum_vs_v2_ky_tt0[ival_first][ival_scnd][idmid]

                else:

                    # Average over all chains
                    ndmid = len(gamma_avg_vs_v2_ky_tt0[ival_first][ival_scnd])
                    for idmid in range(ndmid):
                        gamma_avg_vs_v2_ky[ival_scnd][ival_first] += gamma_avg_vs_v2_ky_tt0[ival_first][ival_scnd][idmid]/ndmid
                    ndmid = len(gamma_avg_fromSum_vs_v2_ky_tt0[ival_first][ival_scnd])
                    for idmid in range(ndmid):
                        gamma_avg_fromSum_vs_v2_ky[ival_scnd][ival_first] += gamma_avg_fromSum_vs_v2_ky_tt0[ival_first][ival_scnd][idmid]/ndmid
                    ndmid = len(gamma_max_vs_v2_ky_tt0[ival_first][ival_scnd])
                    for idmid in range(ndmid):
                        gamma_max_vs_v2_ky[ival_scnd][ival_first] += gamma_max_vs_v2_ky_tt0[ival_first][ival_scnd][idmid]/ndmid
                    ndmid = len(gamma_max_fromSum_vs_v2_ky_tt0[ival_first][ival_scnd])
                    for idmid in range(ndmid):
                        gamma_max_fromSum_vs_v2_ky[ival_scnd][ival_first] += gamma_max_fromSum_vs_v2_ky_tt0[ival_first][ival_scnd][idmid]/ndmid



        # Overwrite old arrays
        dmid_list_vs_v2_ky_tt0 = dmid_list_vs_v2_ky_tt0_new
        itheta0_list_vs_v2_ky_tt0 = itheta0_list_vs_v2_ky_tt0_new
        theta0_vs_v2_ky_tt0 = theta0_vs_v2_ky_tt0_new
        firstdim_vs_v2_ky_tt0 = firstdim_vs_v2_ky_tt0_new
        gamma_avg_vs_v2_ky_tt0 = gamma_avg_vs_v2_ky_tt0_new
        gamma_avg_fromSum_vs_v2_ky_tt0 = gamma_avg_fromSum_vs_v2_ky_tt0_new
        gamma_max_vs_v2_ky_tt0 = gamma_max_vs_v2_ky_tt0_new
        gamma_max_fromSum_vs_v2_ky_tt0 = gamma_max_fromSum_vs_v2_ky_tt0_new
        Qratio_avg_vs_v2_ky_tt0 = Qratio_avg_vs_v2_ky_tt0_new
        g_exb_vs_v2_ky_tt0 = g_exb_vs_v2_ky_tt0_new
        if gamma_inst_pres:
            gamma_inst_vs_v2_ky_tt0 = gamma_inst_vs_v2_ky_tt0_new
            gamma_inst_fromSum_vs_v2_ky_tt0 = gamma_inst_fromSum_vs_v2_ky_tt0_new
            theta0_star_for_inst_vs_v2_ky_tt0 = theta0_star_for_inst_vs_v2_ky_tt0_new

        # Get rid of extra dimension if ndim=1
        #if ndim == ONE:

        #    dmid_list_vs_v2_ky_tt0 = dmid_list_vs_v2_ky_tt0[0]
        #    itheta0_list_vs_v2_ky_tt0 = itheta0_list_vs_v2_ky_tt0[0]
        #    theta0_vs_v2_ky_tt0 = theta0_vs_v2_ky_tt0[0]
        #    firstdim_vs_v2_ky_tt0 = firstdim_vs_v2_ky_tt0[0]
        #    firstdim_vs_v2_ky = firstdim_vs_v2_ky[0]
        #    gamma_avg_vs_v2_ky_tt0 = gamma_avg_vs_v2_ky_tt0[0]
        #    gamma_avg_fromSum_vs_v2_ky_tt0 = gamma_avg_fromSum_vs_v2_ky_tt0[0]
        #    gamma_avg_vs_v2_ky = gamma_avg_vs_v2_ky[0]
        #    gamma_avg_fromSum_vs_v2_ky = gamma_avg_fromSum_vs_v2_ky[0]
        #    gamma_max_vs_v2_ky_tt0 = gamma_max_vs_v2_ky_tt0[0]
        #    gamma_max_fromSum_vs_v2_ky_tt0 = gamma_max_fromSum_vs_v2_ky_tt0[0]
        #    gamma_max_vs_v2_ky = gamma_max_vs_v2_ky[0]
        #    gamma_max_fromSum_vs_v2_ky = gamma_max_fromSum_vs_v2_ky[0]
        #    Qratio_avg_vs_v2_ky_tt0 = Qratio_avg_vs_v2_ky_tt0[0]
        #    Qratio_avg_vs_v2_ky = Qratio_avg_vs_v2_ky[0]
        #    if gamma_inst_pres:
        #        #gamma_inst_vs_v2_ky_tt0 = gamma_inst_vs_v2_ky_tt0[0] # NDCmod
        #        #gamma_inst_fromSum_vs_v2_ky_tt0 = gamma_inst_fromSum_vs_v2_ky_tt0[0] # NDCmod
        #        theta0_star_for_inst_vs_v2_ky_tt0 = theta0_star_for_inst_vs_v2_ky_tt0[0]

    # Save data vs var2,ky with all theta0
    vardict = {}
    vardict['dmid_list_vs_v2_ky_tt0'] = dmid_list_vs_v2_ky_tt0
    vardict['itheta0_list_vs_v2_ky_tt0'] = itheta0_list_vs_v2_ky_tt0
    vardict['theta0_vs_v2_ky_tt0'] = theta0_vs_v2_ky_tt0
    vardict['firstdim_vs_v2_ky_tt0'] = firstdim_vs_v2_ky_tt0
    vardict['gamma_avg_vs_v2_ky_tt0'] = gamma_avg_vs_v2_ky_tt0
    vardict['gamma_avg_fromSum_vs_v2_ky_tt0'] = gamma_avg_fromSum_vs_v2_ky_tt0
    vardict['gamma_max_vs_v2_ky_tt0'] = gamma_max_vs_v2_ky_tt0
    vardict['gamma_max_fromSum_vs_v2_ky_tt0'] = gamma_max_fromSum_vs_v2_ky_tt0
    vardict['Qratio_avg_vs_v2_ky_tt0'] = Qratio_avg_vs_v2_ky_tt0
    vardict['g_exb_vs_v2_ky_tt0'] = g_exb_vs_v2_ky_tt0
    if gamma_inst_pres:
        vardict['theta0_star_for_inst_vs_v2_ky_tt0'] = theta0_star_for_inst_vs_v2_ky_tt0
        vardict['gamma_inst_vs_v2_ky_tt0'] = gamma_inst_vs_v2_ky_tt0
        vardict['gamma_inst_fromSum_vs_v2_ky_tt0'] = gamma_inst_fromSum_vs_v2_ky_tt0
    write_to_file(vardict)

    # Here we check if any run has nonzero g_exb
    try:
        has_g_exb = False in [0 in g_exb_vs_v2_ky[i] for i in range(len(g_exb_vs_v2_ky))]
    except:
        has_g_exb = True in [g_exb_vs_v2_ky[i]!=0.0 for i in range(len(g_exb_vs_v2_ky))]


    if sav_for_tprimigexb_scan:

        tprimi = round(fac_tprimi*tprimi_orig,3)
        vardict = {}
        vardict['g_exb'] = [g_exb_vs_v2_ky[iv2][0] for iv2 in range(seconddim.size)]
        vardict['tprimi'] = tprimi
        vardict['ky'] = firstdim_vs_v2_ky[0][:]
        vardict['gamma_max'] = [max(gamma_max_fromSum_vs_v2_ky[iv2][:]) for iv2 in range(seconddim.size)]
        vardict['iky_atmax_gammax'] = [gamma_max_fromSum_vs_v2_ky[iv2][:].index(max(gamma_max_fromSum_vs_v2_ky[iv2][:])) for iv2 in range(seconddim.size)] 
        vardict['gamma_avg'] = [max(gamma_avg_fromSum_vs_v2_ky[iv2][:]) for iv2 in range(seconddim.size)]
        vardict['iky_atmax_gamavg'] = [gamma_avg_fromSum_vs_v2_ky[iv2][:].index(max(gamma_avg_fromSum_vs_v2_ky[iv2][:])) for iv2 in range(seconddim.size)] 

        #datfile_name = 'postproc/tprimigexb_scan_tprimi_' + str(tprimi) + '.dat'
        datfile_name = 'postproc/tprimigexb_scan_tprimi_' + str(tprimi) + '.dat'
        with open(datfile_name, 'wb') as outfile: # 'wb' stands for write bytes
            pickle.dump(vardict,outfile)





    # Plotting

    # Plots vs ky and vs seconddim

    if ndim == ONE:

        pdfname = 'postproc/linbox_gamma_avg_scan_'+firstdim_var+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel(firstdim_label)
        if not has_g_exb:
            plt.ylabel('$\\gamma [v_{\\textrm{\large th},i}/a]$')
        else:
            plt.ylabel('$\\langle\\gamma\\rangle_t [v_{\\textrm{\large th},i}/a]$')
        plt.plot(firstdim_vs_v2_ky[0], gamma_avg_vs_v2_ky[0], linewidth=2.0, color='k')
        if use_my_ylim:
            plt.ylim(my_ylim_avg)
        plt.savefig(pdfname)

        pdfname = 'postproc/linbox_gamoverksq_avg_scan_'+firstdim_var+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel(firstdim_label)
        if not has_g_exb:
            plt.ylabel('$\\gamma/k_y^2 [v_{\\textrm{\large th},i}\\rho_i^2/a]$')
        else:
            plt.ylabel('$\\langle\\gamma\\rangle_t/k_y^2 [v_{\\textrm{\large th},i}\\rho_i^2/a]$')
        plt.plot(firstdim_vs_v2_ky[0], [g/k**2 for g,k in zip(gamma_avg_vs_v2_ky[0],firstdim_vs_v2_ky[0])], linewidth=2.0, color='k')
        if use_my_ylim:
            plt.ylim(my_ylim_avg_gamoverksq)
        plt.savefig(pdfname)

        pdfname = 'postproc/linbox_gamma_max_scan_'+firstdim_var+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel(firstdim_label)
        if not has_g_exb:
            plt.ylabel('$\\gamma [v_{\\textrm{\large th},i}/a]$')
        else:
            plt.ylabel('$\\langle\\gamma\\rangle_t [v_{\\textrm{\large th},i}/a]$')
        plt.plot(firstdim_vs_v2_ky[0], gamma_max_vs_v2_ky[0], linewidth=2.0, color='k')
        if use_my_ylim:
            plt.ylim(my_ylim_max)
        plt.savefig(pdfname)

        # Same plots, obtained from sum(phi2) instead of max(phi2)

        pdfname = 'postproc/linbox_gamma_avg_fromSum_scan_'+firstdim_var+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel(firstdim_label)
        if not has_g_exb:
            plt.ylabel('$\\gamma [v_{\\textrm{\large th},i}/a]$')
        else:
            plt.ylabel('$\\langle\\gamma\\rangle_t [v_{\\textrm{\large th},i}/a]$')
        plt.plot(firstdim_vs_v2_ky[0], gamma_avg_fromSum_vs_v2_ky[0], linewidth=2.0, color='k')
        if use_my_ylim:
            plt.ylim(my_ylim_avg)
        plt.savefig(pdfname)

        pdfname = 'postproc/linbox_gamma_max_fromSum_scan_'+firstdim_var+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel(firstdim_label)
        if not has_g_exb:
            plt.ylabel('$\\gamma [v_{\\textrm{\large th},i}/a]$')
        else:
            plt.ylabel('$\\langle\\gamma\\rangle_t [v_{\\textrm{\large th},i}/a]$')
        plt.plot(firstdim_vs_v2_ky[0], gamma_max_fromSum_vs_v2_ky[0], linewidth=2.0, color='k')
        if use_my_ylim:
            plt.ylim(my_ylim_max)
        plt.savefig(pdfname)

    elif ndim == TWO:

        color_vs_v2_ky = plt.cm.gnuplot_r(np.linspace(0.05,0.9,seconddim.size))
        if scan_with_single_ky or seconddim.size==1:
            color_vs_v2_ky = ['k']

        # <gamma>_t vs ky

        if not has_g_exb:
            pdfname = 'postproc/linbox_gamma_scan_'+firstdim_var+'_'+seconddim_var+'.pdf'
        else:
            pdfname = 'postproc/linbox_gamma_avg_scan_'+firstdim_var+'_'+seconddim_var+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel(firstdim_label)
        if not has_g_exb:
            plt.ylabel('$\\gamma [v_{\\textrm{\large th},i}/a]$')
        else:
            plt.ylabel('$\\langle\\gamma\\rangle_t [v_{\\textrm{\large th},i}/a]$')
        for ival in range(valdim):
            if scan_with_single_ky:
                plt.plot(seconddim, gamma_avg_vs_v2_ky, linewidth=2.0, color=color_vs_v2_ky[ival])
            else:
                plt.plot(firstdim_vs_v2_ky[ival], gamma_avg_vs_v2_ky[ival], linewidth=2.0, color=color_vs_v2_ky[ival])
            if ndim==TWO:
                my_legend.append(seconddim_label + '$=' + str(round(seconddim[ival],3)) + '$')
        if use_my_ylim:
            plt.ylim(my_ylim_avg)
        if use_my_xlim:
            plt.xlim(my_xlim)
        if ndim==TWO:
            legend = plt.legend(my_legend, frameon = True, fancybox = False, fontsize=13, ncol=3,
                    handlelength=0.75, columnspacing=1.0)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        frame.set_linewidth(0.5)
        frame.set_alpha(1)
        plt.savefig(pdfname)
        print(gamma_avg_vs_v2_ky) # NDCDEL

        # <gamma>_t/ky^2 vs ky

        if not has_g_exb:
            pdfname = 'postproc/linbox_gamoverksq_scan_'+firstdim_var+'_'+seconddim_var+'.pdf'
        else:
            pdfname = 'postproc/linbox_gamoverksq_avg_scan_'+firstdim_var+'_'+seconddim_var+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel(firstdim_label)
        if not has_g_exb:
            plt.ylabel('$\\gamma/k_y^2$ [$v_{\\textrm{\large th},i}\\rho_i^2/a$]')
        else:
            plt.ylabel('$\\langle\\gamma\\rangle_t/k_y^2$ [$v_{\\textrm{\large th},i}\\rho_i^2/a$]')
        for ival in range(valdim):
            if scan_with_single_ky:
                plt.plot(seconddim, [g/k**2 for g,k in zip(gamma_avg_vs_v2_ky,seconddim)], linewidth=2.0, color=color_vs_v2_ky[ival])
            else:
                plt.plot(firstdim_vs_v2_ky[ival], [g/k**2 for g,k in zip(gamma_avg_vs_v2_ky[ival],firstdim_vs_v2_ky[ival])], linewidth=2.0, color=color_vs_v2_ky[ival])
            if ndim==TWO:
                my_legend.append(seconddim_label + '$=' + str(round(seconddim[ival],3)) + '$')
        if use_my_ylim:
            plt.ylim(my_ylim_avg_gamoverksq)
        if use_my_xlim:
            plt.xlim(my_xlim)
        if ndim==TWO:
            legend = plt.legend(my_legend, frameon = True, fancybox = False,
                    handlelength=1.0)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        frame.set_linewidth(0.5)
        frame.set_alpha(1)
        plt.savefig(pdfname)

        # Same plots obtained with sum(phi2) instead of max(phi2)

        # <gamma>_t vs ky from sum(phi)

        if not has_g_exb:
            pdfname = 'postproc/linbox_gamma_fromSum_scan_'+firstdim_var+'_'+seconddim_var+'.pdf'
        else:
            pdfname = 'postproc/linbox_gamma_avg_fromSum_scan_'+firstdim_var+'_'+seconddim_var+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel(firstdim_label)
        if not has_g_exb:
            plt.ylabel('$\\gamma [v_{\\textrm{\large th},i}/a]$')
        else:
            plt.ylabel('$\\langle\\gamma\\rangle_t [v_{\\textrm{\large th},i}/a]$')
        for ival in range(valdim):
            if scan_with_single_ky:
                plt.plot(seconddim, gamma_avg_fromSum_vs_v2_ky, linewidth=2.0, color=color_vs_v2_ky[ival])
            else:
                plt.plot(firstdim_vs_v2_ky[ival], gamma_avg_fromSum_vs_v2_ky[ival], linewidth=2.0, color=color_vs_v2_ky[ival])
            if ndim==TWO:
                my_legend.append(seconddim_label + '$=' + str(round(seconddim[ival],3)) + '$')
        if use_my_ylim:
            plt.ylim(my_ylim_avg)
        if use_my_xlim:
            plt.xlim(my_xlim)
        if ndim==TWO:
            legend = plt.legend(my_legend, frameon = True, fancybox = False, fontsize=13, ncol=3,
                    handlelength=0.75, columnspacing=1.0)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        frame.set_linewidth(0.5)
        frame.set_alpha(1)
        plt.savefig(pdfname)

        # <gamma>_t/ky^2 vs ky from sum(phi)

        if not has_g_exb:
            pdfname = 'postproc/linbox_gamoverksq_fromSum_scan_'+firstdim_var+'_'+seconddim_var+'.pdf'
        else:
            pdfname = 'postproc/linbox_gamoverksq_avg_fromSum_scan_'+firstdim_var+'_'+seconddim_var+'.pdf'
        plt.figure()
        my_legend = []
        plt.grid(True)
        plt.xlabel(firstdim_label)
        if not has_g_exb:
            plt.ylabel('$\\gamma/k_y^2$ [$v_{\\textrm{\large th},i}\\rho_i^2/a$]')
        else:
            plt.ylabel('$\\langle\\gamma\\rangle_t/k_y^2$ [$v_{\\textrm{\large th},i}\\rho_i^2/a$]')
        for ival in range(valdim):
            if scan_with_single_ky:
                plt.plot(seconddim, [g/k**2 for g,k in zip(gamma_avg_fromSum_vs_v2_ky,seconddim)], linewidth=2.0, color=color_vs_v2_ky[ival])
            else:
                plt.plot(firstdim_vs_v2_ky[ival], [g/k**2 for g,k in zip(gamma_avg_fromSum_vs_v2_ky[ival],firstdim_vs_v2_ky[ival])], linewidth=2.0, color=color_vs_v2_ky[ival])
            if ndim==TWO:
                my_legend.append(seconddim_label + '$=' + str(round(seconddim[ival],3)) + '$')
        if use_my_ylim:
            plt.ylim(my_ylim_avg_gamoverksq)
        if use_my_xlim:
            plt.xlim(my_xlim)
        if ndim==TWO:
            legend = plt.legend(my_legend, frameon = True, fancybox = False,
                    handlelength=1.0)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        frame.set_linewidth(0.5)
        frame.set_alpha(1)
        plt.savefig(pdfname)

        if has_g_exb:

            # gamma_max vs ky

            pdfname = 'postproc/linbox_gamma_max_scan_'+firstdim_var+'_'+seconddim_var+'.pdf'
            plt.figure()
            my_legend = []
            plt.grid(True)
            plt.xlabel(firstdim_label)
            plt.ylabel('$\\gamma_{\\textrm{\\large max}}\ [v_{\\textrm{\large th},i}/a]$')
            for ival in range(valdim):
                if scan_with_single_ky:
                    plt.plot(seconddim, gamma_max_vs_v2_ky, linewidth=2.0, color=color_vs_v2_ky[ival])
                else:
                    plt.plot(firstdim_vs_v2_ky[ival], gamma_max_vs_v2_ky[ival], linewidth=2.0, color=color_vs_v2_ky[ival])
                if ndim==TWO:
                    my_legend.append(seconddim_label + '$=' + str(round(seconddim[ival],3)) + '$')
            if use_my_ylim:
                plt.ylim(my_ylim_max)
            if use_my_xlim:
                plt.xlim(my_xlim)
            if ndim==TWO:
                legend = plt.legend(my_legend, frameon = True, fancybox = False, fontsize=13, ncol=3,
                        handlelength=0.75, columnspacing=1.0)
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('black')
            frame.set_linewidth(0.5)
            frame.set_alpha(1)
            plt.savefig(pdfname)
            print('') # NDCDEL
            print(gamma_max_vs_v2_ky) # NDCDEL

            # gamma_max/ky^2 vs ky

            pdfname = 'postproc/linbox_gamoverksq_max_scan_'+firstdim_var+'_'+seconddim_var+'.pdf'
            plt.figure()
            my_legend = []
            plt.grid(True)
            plt.xlabel(firstdim_label)
            plt.ylabel('$\\gamma_{\\textrm{\\large max}}/k_y^2$ [$v_{\\textrm{\large th},i}\\rho_i^2/a$]')
            for ival in range(valdim):
                if scan_with_single_ky:
                    plt.plot(seconddim, [g/k**2 for g,k in zip(gamma_max_vs_v2_ky,seconddim)], linewidth=2.0, color=color_vs_v2_ky[ival])
                else:
                    plt.plot(firstdim_vs_v2_ky[ival], [g/k**2 for g,k in zip(gamma_max_vs_v2_ky[ival],firstdim_vs_v2_ky[ival])], linewidth=2.0, color=color_vs_v2_ky[ival])
                if ndim==TWO:
                    my_legend.append(seconddim_label + '$=' + str(round(seconddim[ival],3)) + '$')
            if use_my_ylim:
                plt.ylim(my_ylim_max_gamoverksq)
            if use_my_xlim:
                plt.xlim(my_xlim)
            if ndim==TWO:
                legend = plt.legend(my_legend, frameon = True, fancybox = False,
                        handlelength=1.0)
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('black')
            frame.set_linewidth(0.5)
            frame.set_alpha(1)
            plt.savefig(pdfname)

            # Same plot obtained with sum(phi2) instead of max(phi2)

            # gamma_max vs ky from sum(phi)

            pdfname = 'postproc/linbox_gamma_max_fromSum_scan_'+firstdim_var+'_'+seconddim_var+'.pdf'
            plt.figure()
            my_legend = []
            plt.grid(True)
            plt.xlabel(firstdim_label)
            plt.ylabel('$\\gamma_{\\textrm{\\large max}}\ [v_{\\textrm{\large th},i}/a]$')
            for ival in range(valdim):
                if scan_with_single_ky:
                    plt.plot(seconddim, gamma_max_fromSum_vs_v2_ky, linewidth=2.0, color=color_vs_v2_ky[ival])
                else:
                    plt.plot(firstdim_vs_v2_ky[ival], gamma_max_fromSum_vs_v2_ky[ival], linewidth=2.0, color=color_vs_v2_ky[ival])
                if ndim==TWO:
                    my_legend.append(seconddim_label + '$=' + str(round(seconddim[ival],3)) + '$')
            if use_my_ylim:
                plt.ylim(my_ylim_max)
            if use_my_xlim:
                plt.xlim(my_xlim)
            if ndim==TWO:
                legend = plt.legend(my_legend, frameon = True, fancybox = False, fontsize=13, ncol=3,
                        handlelength=0.75, columnspacing=1.0)
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('black')
            frame.set_linewidth(0.5)
            frame.set_alpha(1)
            plt.savefig(pdfname)

            # gamma_max/ky^2 vs ky from sum(phi)

            pdfname = 'postproc/linbox_gamoverksq_max_fromSum_scan_'+firstdim_var+'_'+seconddim_var+'.pdf'
            plt.figure()
            my_legend = []
            plt.grid(True)
            plt.xlabel(firstdim_label)
            plt.ylabel('$\\gamma_{\\textrm{\\large max}}/k_y^2$ [$v_{\\textrm{\large th},i}\\rho_i^2/a$]')
            for ival in range(valdim):
                if scan_with_single_ky:
                    plt.plot(seconddim, [g/k**2 for g,k in zip(gamma_max_fromSum_vs_v2_ky,seconddim)], linewidth=2.0, color=color_vs_v2_ky[ival])
                else:
                    plt.plot(firstdim_vs_v2_ky[ival], [g/k**2 for g,k in zip(gamma_max_fromSum_vs_v2_ky[ival],firstdim_vs_v2_ky[ival])], linewidth=2.0, color=color_vs_v2_ky[ival])
                if ndim==TWO:
                    my_legend.append(seconddim_label + '$=' + str(round(seconddim[ival],3)) + '$')
            if use_my_ylim:
                plt.ylim(my_ylim_max_gamoverksq)
            if use_my_xlim:
                plt.xlim(my_xlim)
            if ndim==TWO:
                legend = plt.legend(my_legend, frameon = True, fancybox = False,
                        handlelength=1.0)
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('black')
            frame.set_linewidth(0.5)
            frame.set_alpha(1)
            plt.savefig(pdfname)

        if ndim==TWO:
            try:
                pdfname = 'postproc/linbox_qe_vs_qi_scan_'+seconddim_var+'.pdf'
                plt.figure()
                my_legend = []
                plt.grid(True)
                plt.xlabel(seconddim_label)
                plt.ylabel('$\\langle Q_e/Q_i\\rangle_t$')
                plt.plot(seconddim, Qratio_avg_vs_v2_ky, linewidth=2.0, color='k')
                plt.savefig(pdfname)
            except:
                print('Fluxes cannot be found in the output.')

    # Plot gamma vs (theta0, ky),
    # for every value of the second dimension in the scan.
    
    if firstdim_var == 'ky' and not (has_g_exb and not gamma_inst_pres):

        # Preparing to stitch multiple pdfs together
        tmp_pdf_id = 1
        pdflist = []
        tmp_pdf_id_fromSum = 1
        pdflist_fromSum = []

        # Here we assume that the scan uses a fixed set of ky.
        ky = np.array(firstdim_vs_v2_ky_tt0[0])
        naky = ky.size

        for ival in range(valdim):

            gamma_min = 1e20
            gamma_max = -1e20
            gamma_min_fromSum = 1e20
            gamma_max_fromSum = -1e20

            theta0 = []
            gamma = []
            theta0_fromSum = []
            gamma_fromSum = []

            for iky in range(naky):

                if not has_g_exb:

                    theta0.append([])
                    gamma.append([])
                    theta0_fromSum.append([])
                    gamma_fromSum.append([])

                    ntheta0 = len(itheta0_list_vs_v2_ky_tt0[ival][iky])

                    for iitheta0 in range(ntheta0):

                        this_theta0 = theta0_vs_v2_ky_tt0[ival][iky][itheta0_list_vs_v2_ky_tt0[ival][iky][iitheta0]]
                        this_gamma = gamma_avg_vs_v2_ky_tt0[ival][iky][iitheta0]

                        theta0[iky].append(this_theta0)
                        gamma[iky].append(this_gamma)

                        # Update min and max gamma
                        if this_gamma < gamma_min:
                            gamma_min = this_gamma
                        if this_gamma > gamma_max:
                            gamma_max = this_gamma


                        # Do the same, based on sum(phi2) instead of max(phi2)

                        this_gamma = gamma_avg_fromSum_vs_v2_ky_tt0[ival][iky][iitheta0]

                        gamma_fromSum[iky].append(this_gamma)

                        # Update min and max gamma
                        if this_gamma < gamma_min_fromSum:
                            gamma_min_fromSum = this_gamma
                        if this_gamma > gamma_max_fromSum:
                            gamma_max_fromSum = this_gamma

                else:

                    if dmid_for_plots_vs_ky is not None:

                        theta0.append(theta0_star_for_inst_vs_v2_ky_tt0[ival][iky][dmid_for_plots_vs_ky])
                        gamma.append(gamma_inst_vs_v2_ky_tt0[ival][iky][dmid_for_plots_vs_ky])

                    else:

                        # Average over chains
                        tt0_avg = 0
                        ndmid = len(theta0_star_for_inst_vs_v2_ky_tt0[ival][iky])
                        # tt0 is stored in np.array, so following works:
                        for idmid in range(ndmid):
                            tt0_avg += theta0_star_for_inst_vs_v2_ky_tt0[ival][iky][idmid]/ndmid
                        theta0.append(tt0_avg)
                        # gamma is in a list, so do it differently
                        numt = len(gamma_inst_vs_v2_ky_tt0[ival][iky][0])
                        g_avg = [0]*numt
                        ndmid = len(gamma_inst_vs_v2_ky_tt0[ival][iky])
                        for idmid in range(ndmid):
                            for it in range(numt):
                                g_avg[it] += gamma_inst_vs_v2_ky_tt0[ival][iky][idmid][it]/ndmid
                        gamma.append(g_avg)

                    # Update min and max gamma
                    for idx in range(len(gamma[-1])):

                        this_gamma = gamma[-1][idx]
                        if this_gamma < gamma_min:
                            gamma_min = this_gamma
                        if this_gamma > gamma_max:
                            gamma_max = this_gamma
                    

                    # Do the same, based on sum(phi2) instead of max(phi2)

                    if dmid_for_plots_vs_ky is not None:

                        gamma_fromSum.append(gamma_inst_fromSum_vs_v2_ky_tt0[ival][iky][dmid_for_plots_vs_ky])

                    else:

                        # Average over chains
                        numt = len(gamma_inst_fromSum_vs_v2_ky_tt0[ival][iky][0])
                        g_avg = [0]*numt
                        ndmid = len(gamma_inst_fromSum_vs_v2_ky_tt0[ival][iky])
                        for idmid in range(ndmid):
                            for it in range(numt):
                                g_avg[it] += gamma_inst_fromSum_vs_v2_ky_tt0[ival][iky][idmid][it]/ndmid
                        gamma_fromSum.append(g_avg)

                    # Update min and max gamma
                    for idx in range(len(gamma_fromSum[-1])):

                        this_gamma = gamma_fromSum[-1][idx]
                        if this_gamma < gamma_min_fromSum:
                            gamma_min_fromSum = this_gamma
                        if this_gamma > gamma_max_fromSum:
                            gamma_max_fromSum = this_gamma

            if not has_g_exb:
                xlabel = '$\\theta_0$'
            else:
                xlabel = '$\\theta_0^*$'
            ylabel = '$\\rho_ik_y$'
            title = '$\\gamma\ [v_{\\textrm{\large th},i}/a]$'
            if ndim==TWO:
                title = title + ', ' + seconddim_label + '$= {:.3f}$'.format(seconddim[ival])
            if fix_cbarlim:
                cbarmin = my_cbarmin
                cbarmax = my_cbarmax
            else:
                cbarmin = gamma_min
                cbarmax = gamma_max
            # NUmber of points in refined x-grid
            ngrid_fine = 1001
            gplot.plot_2d_uneven_xgrid(theta0, ky, gamma, -pi, pi, \
                    cbarmin, cbarmax, xlabel, ylabel, title, ngrid_fine)

            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            plt.savefig('postproc/'+tmp_pdfname+'.pdf')
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
        

            # Same plot obtained with sum(phi2) instead of max(phi2)

            gplot.plot_2d_uneven_xgrid(theta0, ky, gamma_fromSum, -pi, pi, \
                    cbarmin, cbarmax, xlabel, ylabel, title, ngrid_fine)

            tmp_pdfname = 'tmp'+str(tmp_pdf_id_fromSum)+'_fromSum'
            plt.savefig('postproc/'+tmp_pdfname+'.pdf')
            pdflist_fromSum.append(tmp_pdfname)
            tmp_pdf_id_fromSum = tmp_pdf_id_fromSum+1

        # Stitch the pdfs together

        merged_pdfname = 'linbox_gam_vs_theta0_ky'
        merge_pdfs(pdflist, merged_pdfname, 'postproc/')

        merged_pdfname = 'linbox_gam_fromSum_vs_theta0_ky'
        merge_pdfs(pdflist_fromSum, merged_pdfname, 'postproc/')






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





def read_data(fname, valtree, scandim, ival_firstdim, dmid_list_vs_v2_ky, itheta0_list_vs_v2_ky, theta0_vs_v2_ky, \
        firstdim_vs_v2_ky, gamma_avg_vs_v2_ky, gamma_avg_fromSum_vs_v2_ky, gamma_max_vs_v2_ky, gamma_max_fromSum_vs_v2_ky, \
        Qratio_avg_vs_v2_ky, g_exb_vs_v2_ky, \
        gamma_inst_vs_v2_ky, gamma_inst_fromSum_vs_v2_ky, theta0_star_for_inst_vs_v2_ky):

    # Iterate over every set of values taken by parameters in this dimension of the scan.
    for ival in range(scan[scandim][0].dim):
    
        # Name-base and patch to be modified for this ival.
        my_fname = fname

        # For every new value of firstdim, append elements
        if scandim==ONE and ival_firstdim!=ival:
            ival_firstdim = ival
            g_exb_vs_v2_ky.append([])
            dmid_list_vs_v2_ky.append([])
            itheta0_list_vs_v2_ky.append([])
            theta0_vs_v2_ky.append([])
            theta0_star_for_inst_vs_v2_ky.append([])
            firstdim_vs_v2_ky.append([])
            gamma_avg_vs_v2_ky.append([])
            gamma_avg_fromSum_vs_v2_ky.append([])
            gamma_max_vs_v2_ky.append([])
            gamma_max_fromSum_vs_v2_ky.append([])
            gamma_inst_vs_v2_ky.append([])
            gamma_inst_fromSum_vs_v2_ky.append([])
            Qratio_avg_vs_v2_ky.append([])

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
            g_exb_vs_v2_ky[ival_firstdim].append(my_vars['g_exb'])
            dmid_list_vs_v2_ky[ival_firstdim].append(my_vars['dmid_list'])
            itheta0_list_vs_v2_ky[ival_firstdim].append(my_vars['itheta0_list'])
            theta0_vs_v2_ky[ival_firstdim].append(my_vars['theta0'])
            firstdim_vs_v2_ky[ival_firstdim].append(my_vars[firstdim_var])
            gamma_avg_vs_v2_ky[ival_firstdim].append(my_vars['gamma_avg'])
            gamma_avg_fromSum_vs_v2_ky[ival_firstdim].append(my_vars['gamma_avg_fromSum'])
            gamma_max_vs_v2_ky[ival_firstdim].append(my_vars['gamma_max'])
            gamma_max_fromSum_vs_v2_ky[ival_firstdim].append(my_vars['gamma_max_fromSum'])
            Qratio_avg_vs_v2_ky[ival_firstdim].append(my_vars['Qratio_avg'])
            try:
                gamma_inst_vs_v2_ky[ival_firstdim].append(my_vars['gamma_inst'])
                gamma_inst_fromSum_vs_v2_ky[ival_firstdim].append(my_vars['gamma_inst_fromSum'])
                theta0_star_for_inst_vs_v2_ky[ival_firstdim].append(my_vars['theta0_star_for_inst'])
            except:
                pass
        # Or move on to the next dimension of the scan by calling function recursively
        else:
            next_scandim = increment_dim(scandim)
            read_data(my_fname, valtree, next_scandim, ival_firstdim, dmid_list_vs_v2_ky, itheta0_list_vs_v2_ky, theta0_vs_v2_ky, \
                    firstdim_vs_v2_ky, gamma_avg_vs_v2_ky, gamma_avg_fromSum_vs_v2_ky, gamma_max_vs_v2_ky, gamma_max_fromSum_vs_v2_ky, \
                    Qratio_avg_vs_v2_ky, g_exb_vs_v2_ky, \
                    gamma_inst_vs_v2_ky, gamma_inst_fromSum_vs_v2_ky, theta0_star_for_inst_vs_v2_ky)




def read_from_file(fname):

    datfile_name = 'postproc/' + fname + '.linbox.dat' ### USER ###

    my_vars = {}

    if os.path.getsize(datfile_name) > 0:
        with open(datfile_name, 'rb') as infile: # 'rb' stands for read bytes
            unpickler = pickle.Unpickler(infile)
            my_vars = unpickler.load()
            #my_vars = pickle.load(infile)
    else:
        print("Cannot open file: " + "'" + datfile_name + "'\n")

    return my_vars




def write_to_file(vardict):

    datfile_name = 'postproc/linbox_scan.dat'
    with open(datfile_name, 'wb') as outfile: # 'wb' stands for write bytes
        pickle.dump(vardict,outfile)



def merge_pdfs(in_namelist, out_name, out_dir):

    # read all tmp pdfs to be merged
    merger = PdfFileMerger()
    for pdfname in in_namelist:
        file_name = out_dir + pdfname + '.pdf'
        with open(file_name, 'rb') as pdffile:
            merger.append(PdfFileReader(pdffile))

    # write and save output pdf
    out_name = out_dir + out_name + '.pdf'
    merger.write(out_name)

    # remove tmp pdfs
    for pdfname in in_namelist:
        file_name = out_dir + pdfname
        os.system('rm -f '+file_name)

    plt.cla()
    plt.clf()




# Execute main
if __name__ == '__main__':
    main()
