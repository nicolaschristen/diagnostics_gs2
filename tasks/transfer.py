import pyfilm
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import copy as cp


def my_task_single(ifile, run, myin, myout, mygrids, mytime):

    transfer = np.squeeze(myout['zonal_transfer'][..., 0] + 1j*myout['zonal_transfer'][..., 1])
    # Reorder kx to be increasing
    transfer = np.concatenate((transfer[:,:,mygrids.nxmid:], transfer[:,:,:mygrids.nxmid]), axis=2)
    # Swap kx and ky axes
    transfer = np.swapaxes(transfer,1,2)
    # Modulus squared
    transfer2 = np.abs(transfer)**2
    # Normalise to max at every time step
    transfer2_nrm = cp.deepcopy(transfer2)
    for it in range(mytime.ntime):
        transfer2_nrm[it,...] = transfer2[it,...] / np.amax(transfer2[it,...])

    # Transfer for zonal mode, excluding kx=0
    kxnozero = np.concatenate((mygrids.kx[:mygrids.nxmid-1], mygrids.kx[mygrids.nxmid:]))
    print(kxnozero)
    transfer2z_kxnozero = np.concatenate((transfer2[:,:mygrids.nxmid-1,0],transfer2[:,mygrids.nxmid:,0]), axis=1)
    transfer2z_nrm_kxnozero = np.concatenate((transfer2_nrm[:,:mygrids.nxmid-1,0],transfer2_nrm[:,mygrids.nxmid:,0]), axis=1)

    # Define the frame rate such that
    # 1 second in the movie = 5 a/vthi
    avth_per_sec = 5

    # Check whether delt changes during the selected time.
    # If it does, display a warning message.
    dt = mytime.time[mytime.it_min+1] - mytime.time[mytime.it_min]
    dt_changed = False
    for it in range(mytime.it_min, mytime.it_max-1):
        if abs(dt - (mytime.time[it+1]-mytime.time[it])) > 1e-6:
            dt_changed = True
    if dt_changed:
        print('\n\nWARNING: delt changes during the selected time window, movies might appear to accelerate.\n\n')

    fps = int(round( avth_per_sec / dt ))

    # Plot transfer vs (kx,ky)

    plt_opts = {'cmap': mpl.cm.RdBu_r,
                'extend': 'both'}
    opts = {'ncontours': 21,
            'cbar_ticks': 5,
            'cbar_label': '',
            'cbar_tick_format': '%.1E',
            'xlabel':'$\\rho_i k_x$',
            'ylabel':'$\\rho_i k_y$',
            'nprocs':20,
            'film_dir': run.out_dir,
            'file_name': run.fnames[ifile] + '_nonlin_transfer',
            'fps': fps}
    opts['title'] = ['$\\sum_s\\langle\\int d^3v\ $NL$\\rangle_\\theta$'\
                     '  at  '\
                     '$t={:.2f}$'.format(mytime.time[it])\
                     +' [$a/v_{th}$]'\
                     for it in range(mytime.ntime)]

    plt.rc('font', size=18)
    pyfilm.pyfilm.make_film_2d(mygrids.kx, mygrids.ky, transfer2,
                               plot_options = plt_opts,
                               options = opts)
    plt.rc('font', size=30)


    # Plot transfer at ky=0, excluding kx=0

    plt_opts = {'marker': 'o',
                'linewidth': 1}
    opts = {'xlabel':'$\\rho_i k_x$',
            'ylabel':'$\\log\\sum_s\\langle\\int d^3v\ $NL$\\rangle_\\theta$',
            'nprocs':20,
            'film_dir': run.out_dir,
            'grid': True,
            'fps': 20}
    opts['title'] = ['$t={:.2f}$'.format(mytime.time[it])\
                     +' [$a/v_{th}$]'\
                     for it in range(2,mytime.ntime)]
    opts['file_name'] = run.fnames[ifile] + 'nonlin_transfer_zonal_vs_kx'

    plt.rc('font', size=18)
    pyfilm.pyfilm.make_film_1d(kxnozero, np.log(transfer2z_nrm_kxnozero[2:,...]),
                               plot_options = plt_opts,
                               options = opts)
    plt.rc('font', size=30)


    # Plot <transfer>_t at ky=0, excluding kx=0

    transfer2z_kxnozero_avg = np.zeros(transfer2z_kxnozero.shape[1])
    for ix in range(transfer2z_kxnozero.shape[1]):
        transfer2z_kxnozero_avg[ix] = mytime.timeavg(transfer2z_nrm_kxnozero[:,ix])
    plt.figure()
    plt.semilogy(kxnozero, transfer2z_kxnozero_avg)
    plt.xlabel('$\\rho_i k_x$')
    plt.ylabel('$\\sum_s\\langle\\int d^3v\ $NL$\\rangle_\\theta$')
    plt.grid(True)
    plt.savefig(run.out_dir + run.fnames[ifile] + 'nonlin_transferavg_zonal_vs_kx' + '.pdf')
