import gs2_plotting as gplot
import gs2_fft as gfft
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pyfilm
import pickle
from scipy.optimize import leastsq
from math import pi



def my_task_single(ifile, run, myin, myout, mygrids, mytime):






    # vvv USER PARAMETERS vvv
    
    # Plot snapshots of fields in real space?
    plot_snap = True

    # Plot zonal quantities in real space?
    plot_zonal = True
    
    # Make movies in real space?
    make_movies = True

    # Time index for plots at specific time
    it_snap = -1
    
    # ^^^ USER PARAMETERS ^^^








    # Dictionary to save data for future use
    mydict = {}

    # Total number of ky's
    ny_full = 2*mygrids.ny-1

    # Check if this simulation had flow shear
    try:
        if myin['dist_fn_knobs']['g_exb'] != 0.0:
            flow_shear = True
        else:
            flow_shear = False
    except:
        flow_shear = False




    # Copy fields at ig=igomega (outboard midplane by default)

    # Electrostatic potential
    phi_all_t = form_complex('phi_igomega_by_mode', myout)
    # Total perturbed densities
    dens_all_t = form_complex('ntot_igomega_by_mode', myout)
    densi_all_t = dens_all_t[:,0,...]
    dense_all_t = dens_all_t[:,1,...]
    # Temperatures
    temp_all_t = form_complex('tpar_igomega_by_mode', myout) \
            + form_complex('tperp_igomega_by_mode', myout)
    tempi_all_t = temp_all_t[:,0,...]
    tempe_all_t = temp_all_t[:,1,...]




    # For cases with flow shear, check if kx_shift is known at every t-step

    if flow_shear:
        kxs = myout['kx_shift']
        if kxs is None or (kxs.ndim == 1 and make_movies):
            print('\n\nWARNING: this simulation has flow shear, \n'\
                  'but kx_shift was not printed over time. \n'\
                  'Movies will feature discrete jumps in time.\n\n')
    else:
        kxs = None




    ###########################
    # Snapshots in real space #
    ###########################

    if plot_snap:

        # Select time step for snapshot

        phi = phi_all_t[it_snap, ...]
        dens = dens_all_t[it_snap, ...]
        densi = densi_all_t[it_snap, ...]
        dense = dense_all_t[it_snap, ...]
        temp = temp_all_t[it_snap, ...]
        tempi = tempi_all_t[it_snap, ...]
        tempe = tempe_all_t[it_snap, ...]

        if flow_shear:
            if kxs is not None and kxs.ndim == 2:
                kxs_snap = np.squeeze(kxs[it_snap,:])
            else:
                kxs_snap = kxs
        else:
            kxs_snap = None




        # Inverse Fourier transform to direct space

        xxx, xxx, phi_snap_fft = gfft.fft_gs2( phi, mygrids.nx, mygrids.ny, mygrids.ky,
                                       kx_shift = kxs_snap,
                                       x = mygrids.xgrid_fft )
        xxx, xxx, densi_snap_fft = gfft.fft_gs2( densi, mygrids.nx, mygrids.ny, mygrids.ky,
                                                 kx_shift = kxs_snap,
                                                 x = mygrids.xgrid_fft )
        xxx, xxx, dense_snap_fft = gfft.fft_gs2( dense, mygrids.nx, mygrids.ny, mygrids.ky,
                                                 kx_shift = kxs_snap,
                                                 x = mygrids.xgrid_fft )
        xxx, xxx, tempi_snap_fft = gfft.fft_gs2( tempi, mygrids.nx, mygrids.ny, mygrids.ky,
                                                 kx_shift = kxs_snap,
                                                 x = mygrids.xgrid_fft )
        xxx, xxx, tempe_snap_fft = gfft.fft_gs2( tempe, mygrids.nx, mygrids.ny, mygrids.ky,
                                                 kx_shift = kxs_snap,
                                                 x = mygrids.xgrid_fft )



        # Plotting

        tmp_pdf_id = 1
        pdflist = []
        tmp_pdf_id_fromSum = 1
        pdflist_fromSum = []

        ttl = '$\\varphi$ [$\\rho_{{\\star}} T_i/e$]'\
              '  at  '\
              '$t={:.2f}$'.format(mytime.time[it_snap])
        ttl += ' [$a/v_{{th}}$]'
        gplot.plot_2d( phi_snap_fft, mygrids.xgrid, mygrids.ygrid,
                      np.amin(phi_snap_fft), np.amax(phi_snap_fft),
                      xlab = '$x/\\rho_i$', ylab = '$y/\\rho_i$',
                      title = ttl,
                      cmp = 'RdBu_c' )

        tmp_pdfname = 'tmp' + str(tmp_pdf_id)
        plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

        ttl = '$\\delta n_i$ [$\\rho_{{\\star}} n_i$]'\
              '  at  '\
              '$t={:.2f}$'.format(mytime.time[it_snap])
        ttl += ' [$a/v_{{th}}$]'
        gplot.plot_2d( densi_snap_fft, mygrids.xgrid, mygrids.ygrid,
                      np.amin(densi_snap_fft), np.amax(densi_snap_fft),
                      xlab = '$x/\\rho_i$', ylab = '$y/\\rho_i$',
                      title = ttl,
                      cmp = 'RdBu_c' )

        tmp_pdfname = 'tmp' + str(tmp_pdf_id)
        plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

        ttl = '$\\delta n_e$ [$\\rho_{{\\star}} n_i$]'\
              '  at  '\
              '$t={:.2f}$'.format(mytime.time[it_snap])
        ttl += ' [$a/v_{{th}}$]'
        gplot.plot_2d( dense_snap_fft, mygrids.xgrid, mygrids.ygrid,
                      np.amin(dense_snap_fft), np.amax(dense_snap_fft),
                      xlab = '$x/\\rho_i$', ylab = '$y/\\rho_i$',
                      title = ttl,
                      cmp = 'RdBu_c' )

        tmp_pdfname = 'tmp' + str(tmp_pdf_id)
        plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

        ttl = '$\\delta T_i$ [$\\rho_{{\\star}} T_i$]'\
              '  at  '\
              '$t={:.2f}$'.format(mytime.time[it_snap])
        ttl += ' [$a/v_{{th}}$]'
        gplot.plot_2d( tempi_snap_fft, mygrids.xgrid, mygrids.ygrid,
                      np.amin(tempi_snap_fft), np.amax(tempi_snap_fft),
                      xlab = '$x/\\rho_i$', ylab = '$y/\\rho_i$',
                      title = ttl,
                      cmp = 'RdBu_c' )

        tmp_pdfname = 'tmp' + str(tmp_pdf_id)
        plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

        ttl = '$\\delta T_e$ [$\\rho_{{\\star}} T_i$]'\
              '  at  '\
              '$t={:.2f}$'.format(mytime.time[it_snap])
        ttl += ' [$a/v_{{th}}$]'
        gplot.plot_2d( tempe_snap_fft, mygrids.xgrid, mygrids.ygrid,
                      np.amin(tempe_snap_fft), np.amax(tempe_snap_fft),
                      xlab = '$x/\\rho_i$', ylab = '$y/\\rho_i$',
                      title = ttl,
                      cmp = 'RdBu_c' )

        tmp_pdfname = 'tmp' + str(tmp_pdf_id)
        plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1




        # Merge pdfs and save
        
        merged_pdfname = 'fields_snap_real_space'
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)




        # Update dictionary

        to_add = { 'phi_snap_fft':phi_snap_fft,
                   'densi_snap_fft':densi_snap_fft,
                   'dense_snap_fft':dense_snap_fft,
                   'tempi_snap_fft':tempi_snap_fft,
                   'tempe_snap_fft':tempe_snap_fft,
                   'time':mytime.time,
                   'it_snap':it_snap,
                   'xgrid':mygrids.xgrid,
                   'ygrid':mygrids.ygrid }
        mydict.update(to_add)

 




    ####################
    # Zonal flow shear #
    ####################

    if plot_zonal:

        # Theta index of outboard midplane
        ig_mid = mygrids.ntheta//2



        # Compute zonal field in real space

        field_zonal = np.zeros((mytime.ntime_steady, mygrids.nx))
        it_g = 0
        for it in range(mytime.it_min, mytime.it_max):
            field_vs_kx = np.squeeze(phi_all_t[it,0,:])
            field_zonal[it_g,:] = np.real(np.fft.ifft(field_vs_kx))
            it_g += 1

        # Average over time
        field_zonal_avg = np.zeros(mygrids.nx)
        for ix in range(mygrids.nx):
            ft = np.squeeze(field_zonal[:,ix])
            field_zonal_avg[ix] = mytime.timeavg( ft,
                                                  use_ft_full = True)
        # Reorder to have growing x axis
        field_zonal_avg = np.concatenate((field_zonal_avg[mygrids.nxmid:],\
                                          field_zonal_avg[:mygrids.nxmid]))



        # Compute zonal flow in real space

        flow_zonal = np.zeros((mytime.ntime_steady, mygrids.nx))
        it_g = 0
        for it in range(mytime.it_min, mytime.it_max):
            field_vs_kx = -1j * mygrids.kx_gs2 * np.squeeze(phi_all_t[it,0,:])
            flow_zonal[it_g,:] = np.real(np.fft.ifft(field_vs_kx))
            it_g += 1

        # Average over time
        flow_zonal_avg = np.zeros(mygrids.nx)
        for ix in range(mygrids.nx):
            ft = np.squeeze(flow_zonal[:,ix])
            flow_zonal_avg[ix] = mytime.timeavg( ft,
                                                  use_ft_full = True)
        # Reorder to have growing x axis
        flow_zonal_avg = np.concatenate((flow_zonal_avg[mygrids.nxmid:],\
                                         flow_zonal_avg[:mygrids.nxmid]))



        # Compute zonal flow shear in real space

        # Factor to get gamma_Z [vthi/a]
        fac = -0.5 * myin['theta_grid_parameters']['rhoc'] \
                / (myin['theta_grid_parameters']['Rmaj']*myin['theta_grid_parameters']['qinp']) \
                * myout['gds22'][ig_mid] / myin['theta_grid_parameters']['shat']**2 \
                * myout['gds2'][ig_mid]**0.5

        # Fit zonal flow with sine to get derivative
        optfunc = lambda x: x[0]*np.sin(x[1]*mygrids.xgrid+x[2]) - flow_zonal_avg
        guess = [ np.max(flow_zonal_avg), \
                  2*pi/(2*np.max(mygrids.xgrid)), \
                  0]
        A,omeg,phase = leastsq(optfunc, guess)[0]
        flow_fit = A*np.sin(omeg*mygrids.xgrid+phase)

        g_zonal = np.zeros((mytime.ntime_steady, mygrids.nx))
        it_g = 0
        for it in range(mytime.it_min, mytime.it_max):
            field_vs_kx = mygrids.kx_gs2**2 * np.squeeze(phi_all_t[it,0,:])
            g_zonal[it_g,:] = fac * np.real(np.fft.ifft(field_vs_kx))
            it_g += 1

        # Average over time
        g_zonal_avg = np.zeros(mygrids.nx)
        for ix in range(mygrids.nx):
            ft = np.squeeze(g_zonal[:,ix])
            g_zonal_avg[ix] = mytime.timeavg( ft,
                                              use_ft_full = True)
        # Reorder to have growing x axis
        g_zonal_avg = np.concatenate((g_zonal_avg[mygrids.nxmid:],\
                                      g_zonal_avg[:mygrids.nxmid]))

        # NDCTEST
        print(fac)
        g_zonal_avg = fac * deriv_c(mygrids.xgrid,flow_fit)

        

        # Plotting

        tmp_pdf_id = 1
        pdflist = []
        tmp_pdf_id_fromSum = 1
        pdflist_fromSum = []

        plt.plot(mygrids.xgrid, field_zonal_avg, linewidth=2)
        plt.xlabel('$x/\\rho_i$')
        ylab = '$\\langle\\varphi_{Z}\\rangle_{turb}$'\
               ' [$\\rho_{\\star T_i/e}$]'
        plt.ylabel(ylab)
        plt.grid(True)

        tmp_pdfname = 'tmp' + str(tmp_pdf_id)
        plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
        plt.cla()
        plt.clf()

        plt.plot(mygrids.xgrid, flow_zonal_avg, linewidth=2, label=None)
        lfit, = plt.plot(mygrids.xgrid, flow_fit, linewidth=2, color='r', label='fit')
        plt.legend()
        plt.xlabel('$x/\\rho_i$')
        ylab = '$\\langle V_{Z}\\rangle_{turb}$'\
               ' [$v_{th}$]'
        plt.ylabel(ylab)
        plt.grid(True)

        tmp_pdfname = 'tmp' + str(tmp_pdf_id)
        plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
        plt.cla()
        plt.clf()

        plt.plot(mygrids.xgrid, g_zonal_avg, linewidth=2)
        plt.xlabel('$x/\\rho_i$')
        ylab = '$\\langle\\gamma_{Z}\\rangle_{turb}$'\
               ' [$v_{th}/a$]'
        plt.ylabel(ylab)
        plt.grid(True)

        tmp_pdfname = 'tmp' + str(tmp_pdf_id)
        plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1


        # Merge pdfs and save
        merged_pdfname = 'zonal_real_space'
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

        to_add = { 'g_zonal':g_zonal_avg,
                   'field_zonal':field_zonal_avg,
                   'flow_zonal':flow_zonal_avg,
                   'time':mytime.time,
                   'it_snap':it_snap,
                   'xgrid':mygrids.xgrid,
                   'ygrid':mygrids.ygrid }
        mydict.update(to_add)




    ###############
    # Make movies #
    ###############


    if make_movies:


        # Inverse 2D FFT over selected time window

        phi_full_fft = np.zeros((mytime.ntime_steady, mygrids.nx, ny_full))
        densi_full_fft = np.zeros((mytime.ntime_steady, mygrids.nx, ny_full))
        dense_full_fft = np.zeros((mytime.ntime_steady, mygrids.nx, ny_full))
        tempi_full_fft = np.zeros((mytime.ntime_steady, mygrids.nx, ny_full))
        tempe_full_fft = np.zeros((mytime.ntime_steady, mygrids.nx, ny_full))

        for it in range(mytime.it_min, mytime.it_max):

            itcut = it - mytime.it_min

            if flow_shear:
                if kxs is not None and kxs.ndim == 2:
                    kxs_now = np.squeeze(kxs[it,:])
                else:
                    kxs_now = None
            else:
                kxs_now = None

            # Potential
            xxx, xxx, field_tmp = gfft.fft_gs2( phi_all_t[it, ...],
                                                mygrids.nx, mygrids.ny, mygrids.ky, 
                                                kx_shift = kxs_now,
                                                x = mygrids.xgrid_fft )
            for ix in range(mygrids.nx):
                for iy in range(ny_full):
                    phi_full_fft[itcut,ix,iy] = field_tmp[iy,ix]

            # Ion density
            xxx, xxx, field_tmp = gfft.fft_gs2( densi_all_t[it, ...],
                                                mygrids.nx, mygrids.ny, mygrids.ky, 
                                                kx_shift = kxs_now,
                                                x = mygrids.xgrid_fft )
            for ix in range(mygrids.nx):
                for iy in range(ny_full):
                    densi_full_fft[itcut,ix,iy] = field_tmp[iy,ix]

            # Electron density
            xxx, xxx, field_tmp = gfft.fft_gs2( dense_all_t[it, ...],
                                                mygrids.nx, mygrids.ny, mygrids.ky, 
                                                kx_shift = kxs_now,
                                                x = mygrids.xgrid_fft )
            for ix in range(mygrids.nx):
                for iy in range(ny_full):
                    dense_full_fft[itcut,ix,iy] = field_tmp[iy,ix]

            # Ion temperature
            xxx, xxx, field_tmp = gfft.fft_gs2( tempi_all_t[it, ...],
                                                mygrids.nx, mygrids.ny, mygrids.ky, 
                                                kx_shift = kxs_now,
                                                x = mygrids.xgrid_fft )
            for ix in range(mygrids.nx):
                for iy in range(ny_full):
                    tempi_full_fft[itcut,ix,iy] = field_tmp[iy,ix]

            # Electron temperature
            xxx, xxx, field_tmp = gfft.fft_gs2( tempe_all_t[it, ...],
                                                mygrids.nx, mygrids.ny, mygrids.ky, 
                                                kx_shift = kxs_now,
                                                x = mygrids.xgrid_fft )
            for ix in range(mygrids.nx):
                for iy in range(ny_full):
                    tempe_full_fft[itcut,ix,iy] = field_tmp[iy,ix]



        # Options for movies

        plt_opts = {'cmap': mpl.cm.RdBu_r,
                    'extend': 'both'}

        opts = {'ncontours': 21,
                'cbar_ticks': 5,
                'cbar_label': '',
                'cbar_tick_format': '%.1E',
                'xlabel':'$x /\\rho_i$',
                'ylabel':'$y /\\rho_i$',
                'film_dir': run.out_dir}


        # Create movies

        # Make movie of potential
        plt.rc('font', size=18)
        opts['title'] = ['$\\varphi$ [$\\rho_{{\\star}} T_i/e$]'\
                         '  at  '\
                         '$t={:.2f}$'.format(mytime.time[it])\
                         +' [$a/v_{th}$]'\
                         for it in range(mytime.it_min, mytime.it_max+1)]
        opts['file_name'] = run.fnames[ifile]+ '_fields_real_space_phi'
        pyfilm.pyfilm.make_film_2d(mygrids.xgrid, mygrids.ygrid, phi_full_fft,
                                   plot_options = plt_opts,
                                   options = opts)
        plt.rc('font', size=30)

        # Make movie of ion density
        plt.rc('font', size=18)
        opts['title'] = ['$\\delta n_i$ [$\\rho_{{\\star}} n_i$]'\
                         '  at  '\
                         '$t={:.2f}$'.format(mytime.time[it])\
                         +' [$a/v_{th}$]'\
                         for it in range(mytime.it_min, mytime.it_max+1)]
        opts['file_name'] = run.fnames[ifile]+ '_fields_real_space_densi'
        pyfilm.pyfilm.make_film_2d(mygrids.xgrid, mygrids.ygrid, densi_full_fft,
                                   plot_options = plt_opts,
                                   options = opts)
        plt.rc('font', size=30)

        # Make movie of electron density
        plt.rc('font', size=18)
        opts['title'] = ['$\\delta n_e$ [$\\rho_{{\\star}} n_i$]'\
                         '  at  '\
                         '$t={:.2f}$'.format(mytime.time[it])\
                         +' [$a/v_{th}$]'\
                         for it in range(mytime.it_min, mytime.it_max+1)]
        opts['file_name'] = run.fnames[ifile]+ '_fields_real_space_dense'
        pyfilm.pyfilm.make_film_2d(mygrids.xgrid, mygrids.ygrid, dense_full_fft,
                                   plot_options = plt_opts,
                                   options = opts)
        plt.rc('font', size=30)

        # Make movie of ion temperature
        plt.rc('font', size=18)
        opts['title'] = ['$\\delta T_i$ [$\\rho_{{\\star}} T_i$]'\
                         '  at  '\
                         '$t={:.2f}$'.format(mytime.time[it])\
                         +' [$a/v_{th}$]'\
                         for it in range(mytime.it_min, mytime.it_max+1)]
        opts['file_name'] = run.fnames[ifile]+ '_fields_real_space_tempi'
        pyfilm.pyfilm.make_film_2d(mygrids.xgrid, mygrids.ygrid, tempi_full_fft,
                                   plot_options = plt_opts,
                                   options = opts)
        plt.rc('font', size=30)

        # Make movie of electron temperature
        plt.rc('font', size=18)
        opts['title'] = ['$\\delta T_e$ [$\\rho_{{\\star}} T_i$]'\
                         '  at  '\
                         '$t={:.2f}$'.format(mytime.time[it])\
                         +' [$a/v_{th}$]'\
                         for it in range(mytime.it_min, mytime.it_max+1)]
        opts['file_name'] = run.fnames[ifile]+ '_fields_real_space_tempe'
        pyfilm.pyfilm.make_film_2d(mygrids.xgrid, mygrids.ygrid, tempe_full_fft,
                                   plot_options = plt_opts,
                                   options = opts)
        plt.rc('font', size=30)




        # Update dict

        to_add = { 'phi_full_fft':phi_full_fft,
                   'densi_full_fft':densi_full_fft,
                   'dense_full_fft':dense_full_fft,
                   'tempi_full_fft':tempi_full_fft,
                   'tempe_full_fft':tempe_full_fft,
                   'time':mytime.time_steady,
                   'xgrid':mygrids.xgrid,
                   'ygrid':mygrids.ygrid }
        mydict.update(to_add)




    ###################
    # Save quantities #
    ###################

    datfile_name = run.out_dir + run.fnames[ifile] + '.fields_real_space.dat'
    with open(datfile_name,'wb') as datfile:
        pickle.dump(mydict,datfile)





            
def form_complex(varname, myout):

    arr = myout[varname][..., 0] + 1j*myout[varname][..., 1]

    return arr



def deriv_c(x,y):

    yprim = np.zeros(x.size)

    dx = x[1]-x[0]
    yprim[0] = (y[1]-y[0])/dx
    for ix in range(1,x.size-1):
        yprim[ix] = (y[ix+1]-y[ix-1])/(2*dx)
    yprim[-1] = (y[-1]-y[-2])/dx

    return yprim
