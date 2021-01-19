import pyfilm
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import copy as cp
import gs2_plotting as gplot
import pickle
from scipy.integrate import simps



def my_task_single(ifile, run, myin, myout, mygrids, mytime):


    compare_dir = '/home/e607/e607/nchriste/work/jet_Ntwopi_vs_dky/rpsi_0.51/gexb_+100_tprimi_+20/1_scan/19_compare_zonal/'

    gamma_avg = 0.01 # Taken from Marconi plot of gavg vs (gexb, tprim) @ gexb = 0.1, tprim = 2

    #files = ['Ntwopi_1_dky_std']
    #labels = ['$N_{2\\pi}=1$, jwtist=4']

    files = ['Ntwopi_3_dky_std']
    labels = ['$N_{2\\pi}=3$, jwtist=4']

    #files = ['Ntwopi_4_dky_half']
    #labels = ['$N_{2\\pi}=4$, jwtist=2']

    nfile = len(files)

    xgrid = [{} for ifile in range(nfile)]
    field_zonal = [{} for ifile in range(nfile)]
    field_zonal_avg = [{} for ifile in range(nfile)]
    shear_zonal = [{} for ifile in range(nfile)]
    shear_zonal_avg = [{} for ifile in range(nfile)]
    shear_zonal_eff_avg = [{} for ifile in range(nfile)]

    for ifile in range(nfile):

        datfile_name = compare_dir + files[ifile] + '.fields_real_space.dat'
        with open(datfile_name,'rb') as datfile:
            tmpdat = pickle.load(datfile)

        xgrid[ifile] = tmpdat['xgrid']
        field_zonal[ifile] = tmpdat['field_zonal']
        field_zonal_avg[ifile] = tmpdat['field_zonal_avg']
        shear_zonal[ifile] = tmpdat['shear_zonal']
        shear_zonal_avg[ifile] = tmpdat['shear_zonal_avg']

        datfile_name = compare_dir + files[ifile] + '.time.dat'
        with open(datfile_name,'rb') as datfile:
            tmpdat = pickle.load(datfile)

        time_steady = tmpdat.time_steady

        shear_zonal_eff_avg[ifile] = eff_avg(gamma_avg, time_steady, shear_zonal[ifile])

    plt.figure()
    for ifile in range(nfile):
        plt.plot(xgrid[ifile], field_zonal_avg[ifile], label=labels[ifile])
    plt.xlabel('$x/\\rho_i$')
    plt.ylabel('$\\langle\\varphi_Z\\rangle_{t}(\\theta=0)$')
    plt.grid(True)
    plt.legend(prop={'size': 12},
               ncol=2,
               frameon=True,
               fancybox=False,
               framealpha=1.0,
               handlelength=1)
    plt.savefig(compare_dir + 'compare_zonal_field_avg.pdf')

    plt.figure()
    for ifile in range(nfile):
        plt.plot(xgrid[ifile], shear_zonal_avg[ifile], label=labels[ifile])
    plt.xlabel('$x/\\rho_i$')
    plt.ylabel('$\\langle\\gamma_Z\\rangle_{t}(\\theta=0)$ [$v_{th}/a$]')
    plt.grid(True)
    plt.legend(prop={'size': 12},
               ncol=2,
               frameon=True,
               fancybox=False,
               framealpha=1.0,
               handlelength=1)
    plt.savefig(compare_dir + 'compare_zonal_shear_avg.pdf')

    plt.figure()
    for ifile in range(nfile):
        plt.plot(xgrid[ifile], shear_zonal_eff_avg[ifile], label=labels[ifile])
    plt.xlabel('$x/\\rho_i$')
    plt.ylabel('$\\langle\\gamma_Z\\rangle_{t}(\\theta=0)$ [$v_{th}/a$]')
    plt.grid(True)
    plt.legend(prop={'size': 12},
               ncol=2,
               frameon=True,
               fancybox=False,
               framealpha=1.0,
               handlelength=1)
    plt.savefig(compare_dir + 'compare_zonal_shear_eff_avg.pdf')



def eff_avg(gamma_max, time, func):

    tau = 1/gamma_max
    delt = time[1]-time[0]

    nt = time.size
    nx = func.shape[1]

    # First: coarse-grain average over tauNL
    func_eff = np.zeros((nt,nx))
    for ix in range(nx):
        for it in range(nt):
            it_min = max(0, it-int(round(0.5*tau//delt)))
            it_max = min(nt-1, it+int(round(0.5*tau//delt)))
            func_eff[it,ix] = simps(func[it_min:it_max,ix], x=time[it_min:it_max]) / (time[it_max]-time[it_min])

    # Then: time average
    func_eff_avg = np.zeros(nx)
    for ix in range(nx):
        func_eff_avg[ix] = simps(func_eff[:,ix], x=time) / (time[-1]-time[0])

    return func_eff_avg
