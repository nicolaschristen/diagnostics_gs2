import pyfilm
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import copy as cp


def my_task_single(ifile, run, myin, myout, mygrids, mytime):

    phi2kxky = myout['phi2_by_mode']
    # Reorder kx to be increasing
    phi2kxky = np.concatenate((phi2kxky[:,:,mygrids.nxmid:], phi2kxky[:,:,:mygrids.nxmid]), axis=2)
    # Swap kx and ky axes
    phi2kxky = np.swapaxes(phi2kxky,1,2)
    # Normalise to max at every time step
    phi2kxky_nrm = cp.deepcopy(phi2kxky)
    phi2kxky_nrm_nozonal = cp.deepcopy(phi2kxky[:,:,1:])
    for it in range(mytime.ntime):
        phi2kxky_nrm[it,...] = phi2kxky_nrm[it,...] / np.amax(phi2kxky_nrm[it,...])
        phi2kxky_nrm_nozonal[it,...] = phi2kxky_nrm_nozonal[it,...] / np.amax(phi2kxky_nrm_nozonal[it,...])

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
            'fps': fps}
    opts['title'] = ['$\\langle\\vert\\varphi\\vert^2\\rangle_\\theta$'\
                     '  at  '\
                     '$t={:.2f}$'.format(mytime.time[it])\
                     +' [$a/v_{th}$]'\
                     for it in range(mytime.it_min,mytime.it_max)]

    plt.rc('font', size=18)
    opts['file_name'] = run.fnames[ifile] + '_phi2_vs_kx_ky'
    pyfilm.pyfilm.make_film_2d(mygrids.kx, mygrids.ky, phi2kxky_nrm[mytime.it_min:mytime.it_max],
                               plot_options = plt_opts,
                               options = opts)
    opts['file_name'] = run.fnames[ifile] + '_phi2_vs_kx_ky_nozonal'
    pyfilm.pyfilm.make_film_2d(mygrids.kx, mygrids.ky[1:], phi2kxky_nrm_nozonal[mytime.it_min:mytime.it_max,...],
                               plot_options = plt_opts,
                               options = opts)
    plt.rc('font', size=30)

