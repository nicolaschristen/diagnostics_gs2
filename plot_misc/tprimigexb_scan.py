from matplotlib import pyplot as plt
import numpy as np
from math import pi
import sys
import pickle

# Add path to directory where scan-files are stored
sys.path.insert(1, '/marconi/home/userexternal/nchriste/codes/scan_gs2/paramfiles')
# Add path to directory where gs2_plotting is stored
sys.path.insert(1, '/marconi/home/userexternal/nchriste/codes/diagnostics_gs2')

import gs2_plotting as gplot

## rpsi = 0.51
#~ tprimi_orig = 1.7392
#~ shat = 0.57383

## rpsi = 0.6
#~ tprimi_orig = 1.8052
#~ shat = 0.83467

## rpsi = 0.7
#~ tprimi_orig = 1.9706
#~ shat = 1.2726

## rpsi = 0.8
tprimi_orig = 2.4461
shat = 1.8761

g_exbfac = 1.0

## rpsi = 0.51
#~ fac_tprimi = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
## rpsi = 0.6, 0.7, 0.8
fac_tprimi = [0.8, 0.9, 1.0, 1.1, 1.2]

ntprimi = len(fac_tprimi)
tprimi = np.zeros(ntprimi)
for itp in range(ntprimi):
    tprimi[itp] = round(fac_tprimi[itp]*tprimi_orig, 3)

if g_exbfac != 0.0:

    ## rpsi = 0.51
    #~ ngexbsmall = 4
    #~ ngexbstd = 14
    #~ g_exb = np.array([-0.0157671, -0.0210228, -0.0262785, -0.0315342,
           #~ -0.0367899 , -0.0420456 , -0.0473013 , -0.052557  , -0.0578127 ,
           #~ -0.0630684 , -0.0683241 , -0.0788355 , -0.09197475, -0.105114  ,
           #~ -0.11825325, -0.1313925 , -0.14453175, -0.157671])

    ## rpsi = 0.6
    #~ ngexbsmall = 0
    #~ g_exb = np.array([-0.053262 , -0.0585882, -0.0639144, -0.0692406])
    #~ ngexbstd = g_exb.size

    ## rpsi = 0.7
    #~ ngexbsmall = 0
    #~ g_exb = np.array([-0.049255 , -0.0541805, -0.059106 , -0.0640315])
    #~ ngexbstd = g_exb.size

    ## rpsi = 0.8
    ngexbsmall = 0
    g_exb = np.array([-0.0453408, -0.0510084, -0.056676 , -0.0623436, -0.0680112, -0.0736788])
    ngexbstd = g_exb.size

else:

    ## rpsi = 0.51
    ngexbsmall = 12
    ngexbstd = 0
    g_exb = np.array([-0.01313925, -0.0262785 , -0.03941775, -0.052557  , -0.06569625,
           -0.0788355 , -0.09197475, -0.105114  , -0.11825325, -0.1313925 ,
           -0.14453175, -0.157671])

is_gexb_small = []
for i in range(ngexbsmall):
    is_gexb_small.append(True)
for i in range(ngexbstd):
    is_gexb_small.append(False)

g_exb = np.abs(g_exb)
ngexb = g_exb.size

if g_exbfac != 0.0:
    Tf = 2*pi*shat/g_exb
else:
    Tf = np.zeros(g_exb.size)

gamma_max = np.zeros((ntprimi, ngexb))
gmax_Tf = np.zeros((ntprimi, ngexb))
ky_at_max_gammax = np.zeros((ntprimi, ngexb))
gamma_avg = np.zeros((ntprimi, ngexb))
gavg_Tf = np.zeros((ntprimi, ngexb))
ky_at_max_gamavg = np.zeros((ntprimi, ngexb))

for itp in range(ntprimi):

    for igb in range(ngexb):

        igb_subset = igb

        if is_gexb_small[igb]:
            datfile_name = 'tprimigexb_scan_tprimi_' + str(tprimi[itp]) + '_smallgexb.dat'
        else:
            datfile_name = 'tprimigexb_scan_tprimi_' + str(tprimi[itp]) + '.dat'
            igb_subset -= ngexbsmall
        vardict = {}
        with open(datfile_name, 'rb') as infile:
            vardict = pickle.load(infile)

        gamma_max[itp,igb] = vardict['gamma_max'][igb_subset]
        gmax_Tf[itp,igb] = vardict['gamma_max'][igb_subset]*Tf[igb]
        ky_at_max_gammax[itp,igb] = vardict['ky'][vardict['iky_atmax_gammax'][igb_subset]]
        gamma_avg[itp,igb] = vardict['gamma_avg'][igb_subset]
        gavg_Tf[itp,igb] = vardict['gamma_avg'][igb_subset]*Tf[igb]
        ky_at_max_gamavg[itp,igb] = vardict['ky'][vardict['iky_atmax_gamavg'][igb_subset]]

### Plotting

gplot.set_plot_defaults()

# Setting up grids for 2d_uneven plots
y_grid = tprimi
x_grid = [[g_exb[igb] for igb in range(ngexb)] for itp in range(ntprimi)]

# gamma_max

## rpsi = 0.51
#~ mycbarmin = -0.05
#~ mycbarmax = 0.15
## rpsi = 0.6
#~ mycbarmin = -0.05
#~ mycbarmax = 0.25
## rpsi = 0.7
#~ mycbarmin = -0.05
#~ mycbarmax = 0.25
## rpsi = 0.8
mycbarmin = -0.05
mycbarmax = 0.35
z = [[gamma_max[itp,igb] for igb in range(ngexb)] for itp in range(ntprimi)]
gplot.plot_2d_uneven_xgrid( x_grid, y_grid, z,
        xmin = min(g_exb), xmax = max(g_exb),
        cbarmin = mycbarmin, cbarmax = mycbarmax,
        xlabel = '$\\vert\\gamma_E\\vert$ [$v_{\\textrm{\\huge th}}/a$]',
        ylabel = '$a/L_{T_i}$',
        title = '$\\gamma_{\\textrm{\\huge max}}$ [$v_{\\textrm{\\huge th}}/a$]',
        x_is_twopi=False )

plt.savefig('gamma_max.pdf')

# gamma_max * Tf

if g_exbfac != 0.0:
    
    ## rpsi = 0.51
    #~ mycbarmin = 0.0
    #~ mycbarmax = 15.0
    ## rpsi = 0.6
    #~ mycbarmin = 0.9
    #~ mycbarmax = 45.0
    ## rpsi = 0.7
    #~ mycbarmin = 0.9
    #~ mycbarmax = 45.0
    ## rpsi = 0.8
    mycbarmin = 0.9
    mycbarmax = 60.0

    z = [[gmax_Tf[itp,igb] for igb in range(ngexb)] for itp in range(ntprimi)]
    gplot.plot_2d_uneven_xgrid( x_grid, y_grid, z,
            xmin = min(g_exb), xmax = max(g_exb),
            cbarmin = mycbarmin, cbarmax = mycbarmax,
            xlabel = '$\\vert\\gamma_E\\vert$ [$v_{\\textrm{\\huge th}}/a$]',
            ylabel = '$a/L_{T_i}$',
            title = '$T_{\\textrm{\\huge F}}\\times\\gamma_{\\textrm{\\huge max}}$',
            x_is_twopi=False,
            clrmap = 'RdBu_c_one',
            zticks = [1.0, 5.0, 10.0, 15.0] )

    plt.savefig('Tf_gmax.pdf')

# ky with max(gamma_max)

z = [[ky_at_max_gammax[itp,igb] for igb in range(ngexb)] for itp in range(ntprimi)]
gplot.plot_2d_uneven_xgrid( x_grid, y_grid, z,
        xmin = min(g_exb), xmax = max(g_exb),
        cbarmin = 0, cbarmax = 1.0,
        xlabel = '$\\vert\\gamma_E\\vert$ [$v_{\\textrm{\\huge th}}/a$]',
        ylabel = '$a/L_{T_i}$',
        title = '$\\rho_ik_y$ of $\\max(\\gamma_{\\textrm{\\huge max}})$',
        x_is_twopi=False,
        clrmap='Reds')

plt.savefig('ky_at_max_gammax.pdf')

# gamma_avg

## rpsi = 0.51
#~ mycbarmin = -0.02
#~ mycbarmax = 0.04
## rpsi = 0.6
#~ mycbarmin = -0.01
#~ mycbarmax = 0.01
## rpsi = 0.7
#~ mycbarmin = -0.01
#~ mycbarmax = 0.01
## rpsi = 0.8
mycbarmin = -0.0075
mycbarmax = 0.005

z = [[gamma_avg[itp,igb] for igb in range(ngexb)] for itp in range(ntprimi)]
gplot.plot_2d_uneven_xgrid( x_grid, y_grid, z,
        xmin = min(g_exb), xmax = max(g_exb),
        cbarmin = mycbarmin, cbarmax = mycbarmax,
        xlabel = '$\\vert\\gamma_E\\vert$ [$v_{\\textrm{\\huge th}}/a$]',
        ylabel = '$a/L_{T_i}$',
        title = '$\\langle\\gamma\\rangle_t$ [$v_{\\textrm{\\huge th}}/a$]',
        x_is_twopi=False )

plt.savefig('gamma_avg.pdf')

# gamma_avg * Tf

if g_exbfac != 0.0:
    
    ## rpsi = 0.51
    #~ mycbarmin = 0.0
    #~ mycbarmax = 4.0
    ## rpsi = 0.6
    #~ mycbarmin = 0.0
    #~ mycbarmax = 1.1
    ## rpsi = 0.7
    #~ mycbarmin = 0.0
    #~ mycbarmax = 1.1
    ## rpsi = 0.8
    mycbarmin = 0.0
    mycbarmax = 1.1

    z = [[gavg_Tf[itp,igb] for igb in range(ngexb)] for itp in range(ntprimi)]
    gplot.plot_2d_uneven_xgrid( x_grid, y_grid, z,
            xmin = min(g_exb), xmax = max(g_exb),
            cbarmin = mycbarmin, cbarmax = mycbarmax,
            xlabel = '$\\vert\\gamma_E\\vert$ [$v_{\\textrm{\\huge th}}/a$]',
            ylabel = '$a/L_{T_i}$',
            title = '$T_{\\textrm{\\huge F}}\\times\\langle\\gamma\\rangle_t$',
            x_is_twopi=False,
            clrmap='RdBu_c_one',
            zticks = [0.0, 1.0, 2.0, 3.0, 4.0],
            zticks_labels = ['$\leq$ 0','1','2','3','4'])
    
    plt.savefig('Tf_gavg.pdf')

# ky with max(gamma_avg)

z = [[ky_at_max_gamavg[itp,igb] for igb in range(ngexb)] for itp in range(ntprimi)]
gplot.plot_2d_uneven_xgrid( x_grid, y_grid, z,
        xmin = min(g_exb), xmax = max(g_exb),
        cbarmin = 0, cbarmax = 1,
        xlabel = '$\\vert\\gamma_E\\vert$ [$v_{\\textrm{\\huge th}}/a$]',
        ylabel = '$a/L_{T_i}$',
        title = '$\\rho_ik_y$ of $\\max\\langle\\gamma\\rangle_t$',
        x_is_twopi=False,
        clrmap='Reds')

plt.savefig('ky_at_max_gamavg.pdf')
