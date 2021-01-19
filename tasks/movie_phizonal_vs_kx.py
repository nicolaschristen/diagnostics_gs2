import pyfilm
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import copy as cp


def my_task_single(ifile, run, myin, myout, mygrids, mytime):

    phi2z = np.squeeze(myout['phi2_by_mode'][:,0,:])
    # Reorder kx to be increasing, leaving out kx=0
    phi2z = np.concatenate((phi2z[:,mygrids.nxmid:], phi2z[:,1:mygrids.nxmid]), axis=1)
    # Normalise to max at every time step
    phi2z_nrm = cp.deepcopy(phi2z)
    for it in range(mytime.ntime):
        phi2z_nrm[it,...] = phi2z_nrm[it,...] / np.amax(phi2z_nrm[it,...])

    # kx grid excluding zero
    kxnozero = np.concatenate((mygrids.kx[:mygrids.nxmid-1], mygrids.kx[mygrids.nxmid:]))

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

    # Options for movies
    plt_opts = {'marker': 'o',
                'linewidth': 1}
    opts = {'xlabel':'$\\rho_i k_x$',
            'nprocs':20,
            'film_dir': run.out_dir,
            'file_name': run.fnames[ifile] + '_phizonal_vs_kx',
            'grid': True,
            'fps': fps}
    opts['title'] = ['$\\log\\langle\\vert\\varphi_Z\\vert^2\\rangle_\\theta$'\
                     '  at  '\
                     '$t={:.2f}$'.format(mytime.time[it])\
                     +' [$a/v_{th}$]'\
                     for it in range(mytime.it_min,mytime.it_max)]

    plt.rc('font', size=18)
    pyfilm.pyfilm.make_film_1d(kxnozero, np.log(phi2z[mytime.it_min:mytime.it_max,:]),
                               plot_options = plt_opts,
                               options = opts)
    plt.rc('font', size=30)


