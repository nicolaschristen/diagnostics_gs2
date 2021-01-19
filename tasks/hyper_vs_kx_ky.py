import pyfilm
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import copy as cp
import gs2_plotting as gplot


def my_task_single(ifile, run, myin, myout, mygrids, mytime):

    plot_snap_at_theta_0 = True
    make_movies = False

    hyper = myout['hypervisc_exp']
    # Reorder kx to be increasing
    hyper = np.concatenate((hyper[:,:,mygrids.nxmid:,:], hyper[:,:,:mygrids.nxmid,:]), axis=2)
    # Arrange into [t,theta,kx,ky]
    hyper = np.swapaxes(hyper,1,3)

    # kx grid excluding zero
    kxnozero = np.concatenate((mygrids.kx[:mygrids.nxmid-1], mygrids.kx[mygrids.nxmid:]))
    # Zonal hyper excluding kx=0
    hyperz_kxnozero = np.concatenate((hyper[:,:,:mygrids.nxmid-1,0],hyper[:,:,mygrids.nxmid:,0]),axis=2)

    # Check whether delt changes during the selected time.
    # If it does, display a warning message.
    dt = mytime.time[mytime.it_min+1] - mytime.time[mytime.it_min]
    dt_changed = False
    for it in range(mytime.it_min, mytime.it_max-1):
        if abs(dt - (mytime.time[it+1]-mytime.time[it])) > 1e-6:
            dt_changed = True
    if dt_changed:
        print('\n\nWARNING: delt changes during the selected time window, movies might appear to accelerate.\n\n')


    #######################
    # Snapshot at theta=0 #
    #######################

    plt.figure()
    hypersnap = 1e-8*hyper[-1,(mygrids.ntheta-1)//2,:,:]
    hypersnap = np.swapaxes(hypersnap,0,1)
    hypermax = np.amax(hypersnap)
    hypermin = np.amin(hypersnap)
    gplot.plot_2d(hypersnap, kxnozero, mygrids.ky, hypermin, hypermax,
                  xlab='kx', ylab='ky', title='-log(hyper)/dt', cmp='RdBu_r')
    gplot.save_plot('hypersnap',run,ifile)

    ##################
    # Movies vs time #
    ##################

    if make_movies:

        plt.rc('font', size=18)

        # All vs kx,ky
        
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
                'fps': 40}

        for itheta in range(mygrids.ntheta):
            opts['title'] = ['1000x (1-Hyper)'\
                             '  at  '\
                             '$t={:.1f}$'.format(mytime.time[it])\
                             +' [$a/v_{th}$]'\
                             for it in range(mytime.it_min,mytime.it_max)]
            opts['file_name'] = run.fnames[ifile] + '_hyper_vs_kx_ky_at_theta_' + str(mygrids.theta[itheta])
            pyfilm.pyfilm.make_film_2d(mygrids.kx, mygrids.ky, 1000*(1-hyper[mytime.it_min:mytime.it_max,itheta,:,:]),
                                       plot_options = plt_opts,
                                       options = opts)

        # Zonal vs kx

        plt_opts = {'marker': 'o',
                    'linewidth': 1}
        opts = {'xlabel':'$\\rho_i k_x$',
                'nprocs':20,
                'film_dir': run.out_dir,
                'grid': True,
                'fps': 40}
        
        for itheta in range(mygrids.ntheta):

            opts['title'] = ['1000x (1-Hyper)'\
                             '  at  '\
                             '$t={:.1f}$'.format(mytime.time[it])\
                             +' [$a/v_{th}$]'\
                             for it in range(mytime.it_min,mytime.it_max)]
            opts['file_name'] = run.fnames[ifile] + '_hyperzonal_vs_kx_at_theta_' + str(mygrids.theta[itheta])

            pyfilm.pyfilm.make_film_1d(kxnozero, 1000*(1-hyperz_kxnozero[mytime.it_min:mytime.it_max,itheta,:]),
                                       plot_options = plt_opts,
                                       options = opts)

        plt.rc('font', size=30)

