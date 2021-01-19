import pyfilm
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import copy as cp
import gs2_plotting as gplot
# For movies
import matplotlib.animation as anim
import sys


def my_task_single(ifile, run, myin, myout, mygrids, mytime):





    # vvv INPUT PARAMETERS vvv #

    # Spike test case, to check spike at kx = +/-2.6:
    # iky = 0,1 @ ispike = 3
    # iky = 2 @ ispike = 1
    # iky = 4 @ ispike = 0
    iky_toplot = [0]
    ispike_toplot = [3]
    negkx = False

    make_plots = True
    make_movies = True
    nrm_movies = True
    
    # ^^^ INPUT PARAMETERS ^^^ #





    # Check whether delt changes during the selected time.
    # If it does, display a warning message.
    dt = mytime.time[mytime.it_min+1] - mytime.time[mytime.it_min]
    dt_changed = False
    for it in range(mytime.it_min, mytime.it_max-1):
        if abs(dt - (mytime.time[it+1]-mytime.time[it])) > 1e-6:
            dt_changed = True
    if dt_changed:
        print('\n\nWARNING: delt changes during the selected time window, movies might appear to slow down or accelerate.\n\n')

    # Options for movies
    plt_opts = {'marker': 'o',
                'linewidth': 1}
    opts = {'xlabel':'$\\theta$',
            'ylabel':'$\\vert\\varphi\\vert^2$',
            'nprocs':20,
            'film_dir': run.out_dir,
            'grid': True,
            'fps': 40}

    for iky in iky_toplot:

        phi = np.squeeze(myout['phi_t'][:,iky,:,:,:])
        phi = phi[..., 0] + 1j*phi[..., 1]
        phi2 = np.abs(phi)**2

        # Reorder kx to be increasing.
        # For zonal modes, leave out kx=0
        if iky != 0:
            idx_start = 0
        else:
            idx_start = 1
        kxsorted = np.concatenate((mygrids.kx[:mygrids.nxmid-idx_start], mygrids.kx[mygrids.nxmid:]))
        phi2 = np.concatenate((phi2[:,mygrids.nxmid:,:], phi2[:,idx_start:mygrids.nxmid,:]), axis=1)

        # Normalise to max at every time step
        phi2_nrm = cp.deepcopy(phi2)
        for it in range(mytime.ntime):
            phi2_nrm[it,...] = phi2_nrm[it,...] / np.amax(phi2_nrm[it,...])

        if negkx:
            kxsign = -1
        else :
            kxsign = 1

        # Pick kx's at which plotting is done:
        # (n*iky*jtwist-1)*dkx, n*iky*jtwist*dkx, (n*iky*jtwist+1)*dkx
        jtwist = myin['kt_grids_box_parameters']['jtwist']
        ikx = kxsorted.size//2
        ikx_to_film = []
        if iky == 0:
            ikx -= kxsign

        # Build list of indices to make movies
        # And plot time averages in packs of three
        tmp_pdf_id = 1
        pdflist = []
        tmp_pdf_id_fromSum = 1
        pdflist_fromSum = []

        if iky == 0:
            njump = kxsign
        else:
            njump = iky*kxsign

        ispike = 0

        while ikx+njump*jtwist < kxsorted.size and ikx+njump*jtwist >= 0:

            ikx += njump*jtwist

            if ispike in ispike_toplot:

                ikx_to_film.append(ikx-kxsign)
                ikx_to_film.append(ikx)
                if ikx+kxsign < kxsorted.size and ikx+kxsign >= 0:
                    ikx_to_film.append(ikx+kxsign)

                if make_plots:

                    phi2avg = np.zeros(mygrids.ntheta)
                    for itheta in range(mygrids.ntheta):
                        phi2avg[itheta] = mytime.timeavg(phi2[:,ikx-kxsign,itheta])
                    plt.plot(mygrids.theta, phi2avg, marker='o', linewidth=1, label='$k_x=$'+str(round(kxsorted[ikx-kxsign],2)))
                    phi2avg = np.zeros(mygrids.ntheta)
                    for itheta in range(mygrids.ntheta):
                        phi2avg[itheta] = mytime.timeavg(phi2[:,ikx,itheta])
                    plt.plot(mygrids.theta, phi2avg, marker='o', linewidth=1, label='$k_x=$'+str(round(kxsorted[ikx],2)))
                    if ikx+kxsign < kxsorted.size and ikx+kxsign >= 0:
                        phi2avg = np.zeros(mygrids.ntheta)
                        for itheta in range(mygrids.ntheta):
                            phi2avg[itheta] = mytime.timeavg(phi2[:,ikx+kxsign,itheta])
                        plt.plot(mygrids.theta, phi2avg, marker='o', linewidth=1, label='$k_x=$'+str(round(kxsorted[ikx+kxsign],2)))
                    plt.grid(True)
                    plt.xlabel('$\\theta$')
                    plt.ylabel('$\\vert\\varphi\\vert^2$')
                    plt.legend()

                    tmp_pdfname = 'tmp' + str(tmp_pdf_id)
                    plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
                    pdflist.append(tmp_pdfname)
                    tmp_pdf_id = tmp_pdf_id+1
                    plt.cla()
                    plt.clf()

                # Make movies

                def movie(xvar, func1, func2, func3, lab1, lab2, lab3, moviename, nrm_movies=False):
                
                    myfig = plt.figure(figsize=(12,8))
                    
                    # adjust movie name
                    moviename = run.out_dir + moviename + '_' + run.fnames[ifile] + '.mp4'
                    
                    print("\nCreating movie ...")
                    
                    # intialise artists
                
                    lfunc1, = plt.plot([],[], marker='o', color=gplot.mybluestd, linewidth=1.0)
                    lfunc1.set_label(lab1)
                
                    lfunc2, = plt.plot([],[], marker='X', color=gplot.myredstd, linewidth=1.0)
                    lfunc2.set_label(lab2)
                
                    lfunc3, = plt.plot([],[], marker='^', color=gplot.myyellow, linewidth=1.0)
                    lfunc3.set_label(lab3)
                    
                    # Labels, limits, legend
                    plt.xlabel('$\\theta$')
                    plt.grid(True)
                    ax = plt.gca()
                    leg = gplot.legend_matlab()
                    ax.legend(loc='upper right')
                    
                    # Update lines
                    def update_plot(data):
                    
                        # Unpack data from yield_data
                        t, dat1, dat2, dat3, datmax = data
                        nrmfac = 1
                        if nrm_movies and datmax!=0.0:
                            nrmfac = datmax
                        # Update data
                        lfunc1.set_data(xvar,dat1/nrmfac)
                        lfunc2.set_data(xvar,dat2/nrmfac)
                        lfunc3.set_data(xvar,dat3/nrmfac)
                        # Update title
                        ttl = '$k_x={:.2f}$'.format(kxsorted[ikx])\
                                + ', $k_y={:.2f}$'.format(mygrids.ky[iky])\
                                + '  at  '\
                                + '$t={:.2f}$'.format(t)\
                                +' [$a/v_{th}$]'
                        plt.gca().set_title(ttl)
                        plt.gca().set_xlim(min(xvar),max(xvar))
                        plt.gca().set_ylim(0.0,datmax/nrmfac)
                    
                        return lfunc1, lfunc2, lfunc3
                    
                    # "yield" = "return, and next time function is called, start from there"
                    def yield_data():
                    
                        for it in range(mytime.ntime):
                    
                            sys.stdout.write("\r{0}".format("\tFrame : "+str(it)+"/"+str(mytime.ntime-1))) # comment out if using HPC queues
                    
                            # find min and max of functions to plot for each time-step
                            funcmax = np.max(func1[it,:])
                            tmpmax = np.max(func2[it,:])
                            if tmpmax > funcmax:
                                funcmax = tmpmax
                            tmpmax = np.max(func3[it,:])
                            if tmpmax > funcmax:
                                funcmax = tmpmax

                            yield mytime.time[it], func1[it,:], func2[it,:], func3[it,:], funcmax
                    
                    mov = anim.FuncAnimation(myfig, update_plot, frames=yield_data, blit=False, save_count=len(range(mytime.ntime)))
                    writer = anim.writers['ffmpeg'](fps=40,bitrate=1800)
                    mov.save(moviename,writer=writer,dpi=100)
                    plt.clf()
                    plt.cla()
                    
                    print("\n... movie completed.")
                    print('\n')

                if make_movies:

                    #plt.rc('font', size=18)
                    #for ikx in ikx_to_film:
                    #    opts['file_name'] = run.fnames[ifile] + '_phi_vs_theta_with_kx_' + str(round(kxsorted[ikx],2)) + '_ky_' + str(round(mygrids.ky[iky],2))
                    #    opts['title'] = ['$k_x={:.2f}$'.format(kxsorted[ikx])\
                    #                     + ', $k_y={:.2f}$'.format(mygrids.ky[iky])\
                    #                     + '  at  '\
                    #                     + '$t={:.2f}$'.format(mytime.time[it])\
                    #                     +' [$a/v_{th}$]'\
                    #                     for it in range(mytime.it_min+1,mytime.it_max)]
                    #    pyfilm.pyfilm.make_film_1d(mygrids.theta, phi2[mytime.it_min+1:mytime.it_max,ikx,:], \
                    #                               plot_options = plt_opts, \
                    #                               options = opts)
                    #plt.rc('font', size=30)

                    xvar = mygrids.theta
                    func1 = phi2[:,ikx-kxsign,:]
                    func2 = phi2[:,ikx,:]
                    func3 = phi2[:,ikx+kxsign,:]
                    lab1 = '$k_x={:.2f}$'.format(kxsorted[ikx-kxsign])
                    lab2 = '$k_x={:.2f}$'.format(kxsorted[ikx])
                    lab3 = '$k_x={:.2f}$'.format(kxsorted[ikx+kxsign])
                    moviename = 'phi_vs_theta_with_ky_' + str(round(mygrids.ky[iky],2)) + '_spike_' + str(ispike)
                    if kxsign == -1:
                        moviename += '_negkx'
                    movie(xvar, func1, func2, func3, lab1, lab2, lab3, moviename, nrm_movies)

            ispike += 1

        # Merge pdfs and save
        if make_plots:
            merged_pdfname = 'phi2avg_vs_theta_with_ky_'+str(round(mygrids.ky[iky],2))
            if negkx:
                merged_pdfname += '_negkx'
            gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)




