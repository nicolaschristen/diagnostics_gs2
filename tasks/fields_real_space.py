import gs2_plotting as gplot
import gs2_fft as gfft
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pyfilm
import pickle
from math import pi
from scipy import interpolate
from matplotlib import rcParams
from matplotlib import rcdefaults
import matplotlib.animation as anim
from scipy.integrate import simps



def my_task_single(ifile, run, myin, myout, mygrids, mytime):





    # vvv USER PARAMETERS vvv
    
    # Plot snapshots of fields in real space?
    plot_snap = True

    # Time index for plots at specific time
    it_snap = -1

    # Plot zonal quantities in real space?
    plot_zonal = True

    # Where should kx be cut to get smoother zonal quantities?
    kxcut = 1.0

    # Which zonal quantity should be fitted with splines?
    to_fit = 'field' # 'field'/'flow'/'shear'

    # Factor to smooth out spline fit of zonal quantities
    smoothfac = 2
    
    # Make movies in real space?
    make_movies = True
    make_mov_phi = True
    make_mov_densi = False
    make_mov_dense = False
    make_mov_tempi = False
    make_mov_tempe = False
    make_mov_zonal = True
    make_mov_yavg = True

    # Pyfilm bug: creating a 1D movie after having called plt.savefig
    # will produce glitches in axis labels.
    # To avoid this: override plotting preferences.
    if make_movies and make_mov_zonal:
        if not plot_zonal:
            plot_zonal = True
        save_plots = False
    else:
        save_plots = True
    
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
        if save_plots:

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



        # Store spectra for field, flow, and shear

        # Only consider kx>0, since phi[-kx] = conj(phi[kx])
        kxpos = mygrids.kx_gs2[:mygrids.nxmid]
        phizonal_avg = 1j*np.zeros(mygrids.nxmid) # need an array of complex numbers
        for ix in range(mygrids.nxmid):
            ft = np.squeeze(phi_all_t[:,0,ix])
            phizonal_avg_real = mytime.timeavg(np.real(ft))
            phizonal_avg_imag = mytime.timeavg(np.imag(ft))
            phizonal_avg[ix] = phizonal_avg_real + 1j*phizonal_avg_imag

        # Spectrum of zonal field
        field_spectrum = np.abs(phizonal_avg)**2

        # Spectrum of zonal flow
        flow_spectrum = kxpos**2 * np.abs(phizonal_avg)**2

        # Spectrum of zonal shear
        shear_spectrum = kxpos**4 * np.abs(phizonal_avg)**2



        # For |kx| < kxcut, get grids and associated phi

        ikxcut = 0
        while mygrids.kx_gs2[ikxcut] < kxcut:
            ikxcut += 1
        ikxcut -= 1
        kx_gs2_cut = np.concatenate((mygrids.kx_gs2[:ikxcut+1], mygrids.kx_gs2[-ikxcut:]))
        phi_all_t_cut = np.concatenate((phi_all_t[:,:,:ikxcut+1], phi_all_t[:,:,-ikxcut:]),axis=2)
        nxcut = kx_gs2_cut.size
        nxmid_cut = nxcut//2+1
        xgrid_fft_cut = 2*pi*np.fft.fftfreq(nxcut,kx_gs2_cut[1])
        xgrid_cut = np.concatenate((xgrid_fft_cut[nxmid_cut:],xgrid_fft_cut[:nxmid_cut]))





        #---------------#
        #  zonal field  #
        #---------------#


        # Full field

        # First, FFT at every time step
        field_zonal = np.zeros((mytime.ntime_steady, mygrids.nx))
        it_steady = 0
        for it in range(mytime.it_min,mytime.it_max):
            ft = np.squeeze(phi_all_t[it,0,:])
            field_zonal[it_steady,:] = np.real(np.fft.ifft(ft)) * mygrids.nx
            it_steady += 1

        # Then average over time
        field_zonal_avg = np.zeros(mygrids.nx)
        for ix in range(mygrids.nx):
            field_zonal_avg[ix] = mytime.timeavg(np.squeeze(field_zonal[:,ix]), use_ft_full=True)

        # Finally, reorder to have growing x axis
        field_zonal = np.concatenate((field_zonal[:,mygrids.nxmid:],\
                                      field_zonal[:,:mygrids.nxmid]),axis=1)
        field_zonal_avg = np.concatenate((field_zonal_avg[mygrids.nxmid:],\
                                          field_zonal_avg[:mygrids.nxmid]))



        # Keeping only |kx| < kxcut

        field_zonal_cut = np.zeros((mytime.ntime_steady, nxcut))
        it_steady = 0
        for it in range(mytime.it_min,mytime.it_max):
            ft = np.squeeze(phi_all_t_cut[it,0,:])
            field_zonal_cut[it_steady,:] = np.real(np.fft.ifft(ft)) * nxcut
            it_steady += 1

        field_zonal_avg_cut = np.zeros(nxcut)
        for ix in range(nxcut):
            field_zonal_avg_cut[ix] = mytime.timeavg(np.squeeze(field_zonal_cut[:,ix]), use_ft_full=True)

        field_zonal_cut = np.concatenate((field_zonal_cut[:,mygrids.nxmid:],\
                                          field_zonal_cut[:,:mygrids.nxmid]),axis=1)
        field_zonal_avg_cut = np.concatenate((field_zonal_avg_cut[nxmid_cut:],\
                                             field_zonal_avg_cut[:nxmid_cut]))



        # Fitting the full x-profile with splines

        if to_fit == 'field':

            # Every time step
            field_zonal_fit = np.zeros((mytime.ntime_steady, mygrids.xgrid_fine.size))
            splinerep_field = []
            it_steady = 0
            for it in range(mytime.it_min,mytime.it_max):
                ft = np.squeeze(field_zonal[it_steady,:])
                spl_smooth = np.max(ft)/smoothfac
                splinerep_field.append(interpolate.splrep(mygrids.xgrid, ft, s=spl_smooth))
                field_zonal_fit[it_steady,:] = interpolate.splev(mygrids.xgrid_fine, splinerep_field[-1])
                it_steady += 1

            # Time-average
            spl_smooth = np.max(field_zonal_avg)/smoothfac
            splinerep_field_avg = interpolate.splrep(mygrids.xgrid, field_zonal_avg, s=spl_smooth)
            field_zonal_avg_fit = interpolate.splev(mygrids.xgrid_fine, splinerep_field_avg)

            label_field_fit = 'spline fit'





        #--------------#
        #  zonal flow  #
        #--------------#


        # Full field

        # First, FFT at every time step
        flow_zonal = np.zeros((mytime.ntime_steady, mygrids.nx))
        it_steady = 0
        for it in range(mytime.it_min,mytime.it_max):
            ft = -1j * mygrids.kx_gs2 * np.squeeze(phi_all_t[it,0,:])
            flow_zonal[it_steady,:] = np.real(np.fft.ifft(ft)) * mygrids.nx
            it_steady += 1

        # Then average over time
        flow_zonal_avg = np.zeros(mygrids.nx)
        for ix in range(mygrids.nx):
            flow_zonal_avg[ix] = mytime.timeavg(np.squeeze(flow_zonal[:,ix]), use_ft_full=True)

        # Finally, reorder to have growing x axis
        flow_zonal = np.concatenate((flow_zonal[:,mygrids.nxmid:],\
                                     flow_zonal[:,:mygrids.nxmid]),axis=1)
        flow_zonal_avg = np.concatenate((flow_zonal_avg[mygrids.nxmid:],\
                                          flow_zonal_avg[:mygrids.nxmid]))




        # Keeping only |kx| < kxcut

        flow_zonal_cut = np.zeros((mytime.ntime_steady, nxcut))
        it_steady = 0
        for it in range(mytime.it_min,mytime.it_max):
            ft = -1j * kx_gs2_cut * np.squeeze(phi_all_t_cut[it,0,:])
            flow_zonal_cut[it_steady,:] = np.real(np.fft.ifft(ft)) * nxcut
            it_steady += 1

        flow_zonal_avg_cut = np.zeros(nxcut)
        for ix in range(nxcut):
            flow_zonal_avg_cut[ix] = mytime.timeavg(np.squeeze(flow_zonal_cut[:,ix]), use_ft_full=True)

        flow_zonal_cut = np.concatenate((flow_zonal_cut[:,mygrids.nxmid:],\
                                         flow_zonal_cut[:,:mygrids.nxmid]),axis=1)
        flow_zonal_avg_cut = np.concatenate((flow_zonal_avg_cut[nxmid_cut:],\
                                             flow_zonal_avg_cut[:nxmid_cut]))



        # Fitting the full x-profile with splines

        # Either take a derivative from the fitted field
        if to_fit == 'field':

            # Every time step
            flow_zonal_fit = np.zeros((mytime.ntime_steady, mygrids.xgrid_fine.size))
            it_steady = 0
            for it in range(mytime.it_min,mytime.it_max):
                flow_zonal_fit[it_steady,:] = interpolate.splev(mygrids.xgrid_fine, splinerep_field[it_steady], der=1)
                it_steady += 1

            # Time-average
            flow_zonal_avg_fit = interpolate.splev(mygrids.xgrid_fine, splinerep_field_avg, der=1)
            label_flow_fit = 'd/dx of fitted field'

        # Or if the flow is smooth enough, fit the flow itself
        elif to_fit == 'flow':

            # Every time step
            flow_zonal_fit = np.zeros((mytime.ntime_steady, mygrids.xgrid_fine.size))
            splinerep_flow = []
            it_steady = 0
            for it in range(mytime.it_min,mytime.it_max):
                ft = np.squeeze(flow_zonal[it_steady,:])
                spl_smooth = np.max(ft)/smoothfac
                splinerep_flow.append(interpolate.splrep(mygrids.xgrid, ft, s=spl_smooth))
                flow_zonal_fit[it_steady,:] = interpolate.splev(mygrids.xgrid_fine, splinerep_flow[-1])
                it_steady += 1

            # Time-average
            spl_smooth = np.max(flow_zonal_avg)/smoothfac
            splinerep_flow_avg = interpolate.splrep(mygrids.xgrid, flow_zonal_avg, s=spl_smooth)
            flow_zonal_avg_fit = interpolate.splev(mygrids.xgrid_fine, splinerep_flow_avg)

            label_flow_fit = 'spline fit'





        #---------------#
        #  zonal shear  #
        #---------------#


        # Factor to get gamma_Z [vthi/a]
        fac = 0.5 \
                * myout['gds2'][ig_mid]**-0.5 \
                * (myout['gds22'][ig_mid]*myout['gds2'][ig_mid] - myout['gds21'][ig_mid]**2) \
                * (myin['theta_grid_parameters']['rhoc']/myin['theta_grid_parameters']['shat'])**2


        # Full field

        # First, FFT at every time step
        shear_zonal = np.zeros((mytime.ntime_steady, mygrids.nx))
        it_steady = 0
        for it in range(mytime.it_min,mytime.it_max):
            ft = mygrids.kx_gs2**2 * np.squeeze(phi_all_t[it,0,:])
            shear_zonal[it_steady,:] = fac * np.real(np.fft.ifft(ft)) * mygrids.nx
            it_steady += 1

        # Then average over time
        shear_zonal_avg = np.zeros(mygrids.nx)
        for ix in range(mygrids.nx):
            shear_zonal_avg[ix] = mytime.timeavg(np.squeeze(shear_zonal[:,ix]), use_ft_full=True)

        # Finally, reorder to have growing x axis
        shear_zonal = np.concatenate((shear_zonal[:,mygrids.nxmid:],\
                                      shear_zonal[:,:mygrids.nxmid]),axis=1)
        shear_zonal_avg = np.concatenate((shear_zonal_avg[mygrids.nxmid:],\
                                          shear_zonal_avg[:mygrids.nxmid]))



        # Keeping only |kx| < kxcut

        shear_zonal_cut = np.zeros((mytime.ntime_steady, nxcut))
        it_steady = 0
        for it in range(mytime.it_min,mytime.it_max):
            ft = kx_gs2_cut**2 * np.squeeze(phi_all_t_cut[it,0,:])
            shear_zonal_cut[it_steady,:] = fac * np.real(np.fft.ifft(ft)) * nxcut
            it_steady += 1

        shear_zonal_avg_cut = np.zeros(nxcut)
        for ix in range(nxcut):
            shear_zonal_avg_cut[ix] = mytime.timeavg(np.squeeze(shear_zonal_cut[:,ix]), use_ft_full=True)

        shear_zonal_cut = np.concatenate((shear_zonal_cut[:,mygrids.nxmid:],\
                                          shear_zonal_cut[:,:mygrids.nxmid]),axis=1)
        shear_zonal_avg_cut = np.concatenate((shear_zonal_avg_cut[nxmid_cut:],\
                                              shear_zonal_avg_cut[:nxmid_cut]))



        # Fitting the full x-profile with splines

        # Either take the second derivative from the fitted field
        if to_fit == 'field':

            # Every time step
            shear_zonal_fit = np.zeros((mytime.ntime_steady, mygrids.xgrid_fine.size))
            it_steady = 0
            for it in range(mytime.it_min,mytime.it_max):
                shear_zonal_fit[it_steady,:] = fac * interpolate.splev(mygrids.xgrid_fine, splinerep_field[it_steady], der=2)
                it_steady += 1

            # Time-average
            shear_zonal_avg_fit = fac * interpolate.splev(mygrids.xgrid_fine, splinerep_field_avg, der=2)

            label_shear_fit = 'd^2/dx^2 of fitted field'

        # Or take a derivative from the fitted flow
        elif to_fit == 'flow':

            # Every time step
            shear_zonal_fit = np.zeros((mytime.ntime_steady, mygrids.xgrid_fine.size))
            it_steady = 0
            for it in range(mytime.it_min,mytime.it_max):
                shear_zonal_fit[it_steady,:] = fac * interpolate.splev(mygrids.xgrid_fine, splinerep_flow[it_steady], der=1)
                it_steady += 1

            # Time-average
            shear_zonal_avg_fit = fac * interpolate.splev(mygrids.xgrid_fine, splinerep_flow_avg, der=1)

            label_shear_fit = 'd/dx of fitted flow'

        # Or if the shear is smooth enough, fit the shear itself
        elif to_fit == 'shear':

            # Every time step
            shear_zonal_fit = np.zeros((mytime.ntime_steady, mygrids.xgrid_fine.size))
            splinerep_shear = []
            it_steady = 0
            for it in range(mytime.it_min,mytime.it_max):
                ft = np.squeeze(shear_zonal[it_steady,:])
                spl_smooth = np.max(ft)/smoothfac
                splinerep_shear.append(interpolate.splrep(mygrids.xgrid, ft, s=spl_smooth))
                shear_zonal_fit[it_steady,:] = interpolate.splev(mygrids.xgrid_fine, splinerep_shear[-1])
                it_steady += 1

            # Time-average
            spl_smooth = np.max(shear_zonal_avg)/smoothfac
            splinerep_shear_avg = interpolate.splrep(mygrids.xgrid, shear_zonal_avg, s=spl_smooth)
            shear_zonal_avg_fit = interpolate.splev(mygrids.xgrid_fine, splinerep_shear_avg)

            label_shear_fit = 'spline fit'

        if flow_shear:

            gexb = myin['dist_fn_knobs']['g_exb']
            sheartot_zonal_fit = np.zeros((mytime.ntime_steady, mygrids.xgrid_fine.size))
            it_steady = 0
            for it in range(mytime.it_min,mytime.it_max):
                for ix in range(mygrids.xgrid_fine.size):
                    sheartot_zonal_fit[it_steady,ix] = shear_zonal_fit[it_steady,ix] + gexb
                it_steady += 1




        #------------#
        #  Plotting  #
        #------------#

        if save_plots:

            tmp_pdf_id = 1
            pdflist = []
            tmp_pdf_id_fromSum = 1
            pdflist_fromSum = []

            #
            # Zonal field
            #

            plt.semilogy(kxpos, field_spectrum, linewidth=2)
            plt.axvline(x=kxcut, color='k', linestyle='-')
            plt.xlabel('$\\rho_i k_x$')
            ylab = '$\\langle\\vert\\hat{\\varphi}\\vert^2_{Z}\\rangle_{t}$'
            plt.ylabel(ylab)
            plt.grid(True)

            tmp_pdfname = 'tmp' + str(tmp_pdf_id)
            plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
            plt.cla()
            plt.clf()

            plt.plot(mygrids.xgrid, field_zonal_avg, linewidth=2)
            lcut, = plt.plot(xgrid_cut, field_zonal_avg_cut, linewidth=2, color='k', label='cut kx', linestyle='--')
            if to_fit == 'field':
                lfit, = plt.plot(mygrids.xgrid_fine, field_zonal_avg_fit, linewidth=2, color='r', label=label_field_fit)
            plt.legend()
            plt.xlabel('$x/\\rho_i$')
            plt.title('Averaged over time')
            ylab = '$\\langle\\varphi_{Z}\\rangle_{t}$'\
                   ' [$\\rho_{\\star T_i/e}$]'
            plt.ylabel(ylab)
            plt.grid(True)

            tmp_pdfname = 'tmp' + str(tmp_pdf_id)
            plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
            plt.cla()
            plt.clf()

            plt.plot(mygrids.xgrid, field_zonal[0,:], linewidth=2)
            if to_fit == 'field':
                lfit, = plt.plot(mygrids.xgrid_fine, field_zonal_fit[0,:], linewidth=2, color='r', label=label_field_fit)
            plt.legend()
            plt.xlabel('$x/\\rho_i$')
            plt.title('Instantaneous')
            ylab = '$\\varphi_{Z}$'\
                   ' [$\\rho_{\\star T_i/e}$]'
            plt.ylabel(ylab)
            plt.grid(True)

            tmp_pdfname = 'tmp' + str(tmp_pdf_id)
            plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
            plt.cla()
            plt.clf()

            #
            # Zonal flow
            #

            plt.semilogy(kxpos, flow_spectrum, linewidth=2)
            plt.axvline(x=kxcut, color='k', linestyle='-')
            plt.xlabel('$\\rho_i k_x$')
            ylab = '$k_x^2\\langle\\vert\\hat{\\varphi}\\vert^2_{Z}\\rangle_{t}$'
            plt.ylabel(ylab)
            plt.grid(True)

            tmp_pdfname = 'tmp' + str(tmp_pdf_id)
            plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
            plt.cla()
            plt.clf()

            plt.plot(mygrids.xgrid, flow_zonal_avg, linewidth=2, label=None)
            lcut, = plt.plot(xgrid_cut, flow_zonal_avg_cut, linewidth=2, color='k', label='cut kx', linestyle='--')
            if to_fit == 'field' or to_fit == 'flow':
                lfit, = plt.plot(mygrids.xgrid_fine, flow_zonal_avg_fit, linewidth=2, color='r', label=label_flow_fit)
            plt.legend()
            plt.xlabel('$x/\\rho_i$')
            plt.title('Time averaged')
            ylab = '$-\\sum_{k_x}ik_x\\langle\\hat{\\varphi}_{Z}\\rangle_{t} e^{ik_x x}$'\
                   ' [$T_i/(ea)$]'
            plt.ylabel(ylab)
            plt.grid(True)

            tmp_pdfname = 'tmp' + str(tmp_pdf_id)
            plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
            plt.cla()
            plt.clf()

            plt.plot(mygrids.xgrid, flow_zonal[0,:], linewidth=2)
            if to_fit == 'field' or to_field == 'flow':
                lfit, = plt.plot(mygrids.xgrid_fine, flow_zonal_fit[0,:], linewidth=2, color='r', label=label_flow_fit)
            plt.legend()
            plt.xlabel('$x/\\rho_i$')
            plt.title('Instantaneous')
            ylab = '$-\\sum_{k_x}ik_x\\hat{\\varphi}_{Z} e^{ik_x x}$'\
                   ' [$T_i/(ea)$]'
            plt.ylabel(ylab)
            plt.grid(True)

            tmp_pdfname = 'tmp' + str(tmp_pdf_id)
            plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
            plt.cla()
            plt.clf()

            #
            # Zonal shear
            #

            plt.semilogy(kxpos, shear_spectrum, linewidth=2)
            plt.axvline(x=kxcut, color='k', linestyle='-')
            plt.xlabel('$\\rho_i k_x$')
            ylab = '$k_x^4\\langle\\vert\\hat{\\varphi}\\vert^2_{Z}\\rangle_{t}$'
            plt.ylabel(ylab)
            plt.grid(True)

            tmp_pdfname = 'tmp' + str(tmp_pdf_id)
            plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
            plt.cla()
            plt.clf()

            plt.plot(mygrids.xgrid, shear_zonal_avg, linewidth=2, label=None)
            lcut, = plt.plot(xgrid_cut, shear_zonal_avg_cut, linewidth=2, color='k', label='cut kx', linestyle='--')
            lfit, = plt.plot(mygrids.xgrid_fine, shear_zonal_avg_fit, linewidth=2, color='r', label=label_shear_fit)
            plt.legend()
            plt.xlabel('$x/\\rho_i$')
            plt.title('Time averaged')
            ylab = '$\\langle\\gamma_{Z}\\rangle_{t}$'\
                   ' [$v_{th}/a$]'
            plt.ylabel(ylab)
            plt.grid(True)

            tmp_pdfname = 'tmp' + str(tmp_pdf_id)
            plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
            plt.cla()
            plt.clf()

            plt.plot(mygrids.xgrid, shear_zonal[0,:], linewidth=2, label=None)
            lfit, = plt.plot(mygrids.xgrid_fine, shear_zonal_fit[0,:], linewidth=2, color='r', label=label_shear_fit)
            plt.legend()
            plt.xlabel('$x/\\rho_i$')
            plt.title('Instantaneous')
            ylab = '$\\gamma_{Z}$'\
                   ' [$v_{th}/a$]'
            plt.ylabel(ylab)
            plt.grid(True)

            tmp_pdfname = 'tmp' + str(tmp_pdf_id)
            plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
            plt.cla()
            plt.clf()


            # Merge pdfs and save
            merged_pdfname = 'zonal_real_space'
            gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

            to_add = { 'shear_zonal_avg':shear_zonal_avg,
                       'field_zonal_avg':field_zonal_avg,
                       'flow_zonal_avg':flow_zonal_avg,
                       'shear_zonal':shear_zonal,
                       'field_zonal':field_zonal,
                       'flow_zonal':flow_zonal,
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
            if make_mov_phi:

                xxx, xxx, field_tmp = gfft.fft_gs2( phi_all_t[it, ...],
                                                    mygrids.nx, mygrids.ny, mygrids.ky, 
                                                    kx_shift = kxs_now,
                                                    x = mygrids.xgrid_fft )
                for ix in range(mygrids.nx):
                    for iy in range(ny_full):
                        phi_full_fft[itcut,ix,iy] = field_tmp[iy,ix]

            # Ion density
            if make_mov_densi:

                xxx, xxx, field_tmp = gfft.fft_gs2( densi_all_t[it, ...],
                                                    mygrids.nx, mygrids.ny, mygrids.ky, 
                                                    kx_shift = kxs_now,
                                                    x = mygrids.xgrid_fft )
                for ix in range(mygrids.nx):
                    for iy in range(ny_full):
                        densi_full_fft[itcut,ix,iy] = field_tmp[iy,ix]

            # Electron density
            if make_mov_dense:

                xxx, xxx, field_tmp = gfft.fft_gs2( dense_all_t[it, ...],
                                                    mygrids.nx, mygrids.ny, mygrids.ky, 
                                                    kx_shift = kxs_now,
                                                    x = mygrids.xgrid_fft )
                for ix in range(mygrids.nx):
                    for iy in range(ny_full):
                        dense_full_fft[itcut,ix,iy] = field_tmp[iy,ix]

            # Ion temperature
            if make_mov_tempi:

                xxx, xxx, field_tmp = gfft.fft_gs2( tempi_all_t[it, ...],
                                                    mygrids.nx, mygrids.ny, mygrids.ky, 
                                                    kx_shift = kxs_now,
                                                    x = mygrids.xgrid_fft )
                for ix in range(mygrids.nx):
                    for iy in range(ny_full):
                        tempi_full_fft[itcut,ix,iy] = field_tmp[iy,ix]

            # Electron temperature
            if make_mov_tempe:

                xxx, xxx, field_tmp = gfft.fft_gs2( tempe_all_t[it, ...],
                                                    mygrids.nx, mygrids.ny, mygrids.ky, 
                                                    kx_shift = kxs_now,
                                                    x = mygrids.xgrid_fft )
                for ix in range(mygrids.nx):
                    for iy in range(ny_full):
                        tempe_full_fft[itcut,ix,iy] = field_tmp[iy,ix]



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



        # Options for 2D movies

        plt_opts = {'cmap': mpl.cm.RdBu_r,
                    'extend': 'both'}

        opts = {'ncontours': 21,
                'cbar_ticks': 5,
                'cbar_label': '',
                'cbar_tick_format': '%.1E',
                'xlabel':'$x /\\rho_i$',
                'ylabel':'$y /\\rho_i$',
                'nprocs':20,
                'film_dir': run.out_dir,
                'fps': fps}


        # Create movies

        # Make movie of potential
        if make_mov_phi:

            plt.rc('font', size=18)
            opts['title'] = ['$\\varphi$ [$\\rho_{{\\star}} T_i/e$]'\
                             '  at  '\
                             '$t={:.2f}$'.format(mytime.time[it])\
                             +' [$a/v_{th}$]'\
                             for it in range(mytime.it_min, mytime.it_max)]
            opts['file_name'] = run.fnames[ifile]+ '_fields_real_space_phi'
            opts['cbar_ticks'] = np.array([np.amin(phi_full_fft), np.amin(phi_full_fft)/2, 0.0, np.amax(phi_full_fft)/2, np.amax(phi_full_fft)])
            pyfilm.pyfilm.make_film_2d(mygrids.xgrid, mygrids.ygrid, phi_full_fft,
                                       plot_options = plt_opts,
                                       options = opts)
            plt.rc('font', size=30)

        # Make movie of ion density
        if make_mov_densi:

            plt.rc('font', size=18)
            opts['title'] = ['$\\delta n_i$ [$\\rho_{{\\star}} n_i$]'\
                             '  at  '\
                             '$t={:.2f}$'.format(mytime.time[it])\
                             +' [$a/v_{th}$]'\
                             for it in range(mytime.it_min, mytime.it_max)]
            opts['file_name'] = run.fnames[ifile]+ '_fields_real_space_densi'
            pyfilm.pyfilm.make_film_2d(mygrids.xgrid, mygrids.ygrid, densi_full_fft,
                                       plot_options = plt_opts,
                                       options = opts)
            plt.rc('font', size=30)

        # Make movie of electron density
        if make_mov_dense:

            plt.rc('font', size=18)
            opts['title'] = ['$\\delta n_e$ [$\\rho_{{\\star}} n_i$]'\
                             '  at  '\
                             '$t={:.2f}$'.format(mytime.time[it])\
                             +' [$a/v_{th}$]'\
                             for it in range(mytime.it_min, mytime.it_max)]
            opts['file_name'] = run.fnames[ifile]+ '_fields_real_space_dense'
            pyfilm.pyfilm.make_film_2d(mygrids.xgrid, mygrids.ygrid, dense_full_fft,
                                       plot_options = plt_opts,
                                       options = opts)
            plt.rc('font', size=30)

        # Make movie of ion temperature
        if make_mov_tempi:

            plt.rc('font', size=18)
            opts['title'] = ['$\\delta T_i$ [$\\rho_{{\\star}} T_i$]'\
                             '  at  '\
                             '$t={:.2f}$'.format(mytime.time[it])\
                             +' [$a/v_{th}$]'\
                             for it in range(mytime.it_min, mytime.it_max)]
            opts['file_name'] = run.fnames[ifile]+ '_fields_real_space_tempi'
            pyfilm.pyfilm.make_film_2d(mygrids.xgrid, mygrids.ygrid, tempi_full_fft,
                                       plot_options = plt_opts,
                                       options = opts)
            plt.rc('font', size=30)

        # Make movie of electron temperature
        if make_mov_tempe:

            plt.rc('font', size=18)
            opts['title'] = ['$\\delta T_e$ [$\\rho_{{\\star}} T_i$]'\
                             '  at  '\
                             '$t={:.2f}$'.format(mytime.time[it])\
                             +' [$a/v_{th}$]'\
                             for it in range(mytime.it_min, mytime.it_max)]
            opts['file_name'] = run.fnames[ifile]+ '_fields_real_space_tempe'
            pyfilm.pyfilm.make_film_2d(mygrids.xgrid, mygrids.ygrid, tempe_full_fft,
                                       plot_options = plt_opts,
                                       options = opts)
            plt.rc('font', size=30)




        # Update dict

        to_add = { 'time_steady':mytime.time_steady,
                   'xgrid':mygrids.xgrid,
                   'ygrid':mygrids.ygrid }
        mydict.update(to_add)
        if make_mov_phi:
            to_add = {'phi_full_fft':phi_full_fft}
            mydict.update(to_add)
        if make_mov_densi:
            to_add = {'densi_full_fft':densi_full_fft}
            mydict.update(to_add)
        if make_mov_dense:
            to_add = {'dense_full_fft':dense_full_fft}
            mydict.update(to_add)
        if make_mov_tempi:
            to_add = {'tempi_full_fft':tempi_full_fft}
            mydict.update(to_add)
        if make_mov_tempe:
            to_add = {'tempe_full_fft':tempe_full_fft}
            mydict.update(to_add)




        # Make 1D movies of y-averaged quantities
        if make_mov_yavg:

            # Options for 1D movies
            plt_opts = {'marker': None,
                        'linewidth': 2,
                        'linestyle': '-'}

            opts = {'xlabel':'$x /\\rho_i$',
                    'title':['$t={:.2f}$'.format(mytime.time[it])\
                             +' [$a/v_{th}$]'\
                             for it in range(mytime.it_min, mytime.it_max)],
                    'xlim':[np.amin(mygrids.xgrid),np.amax(mygrids.xgrid)],
                    'film_dir': run.out_dir,
                    'grid': True,
                    'fps': fps}

            plt.rc('font', size=18)

            # Compute y-averaged ion temperature and fit it with splines
            it_steady = 0
            tempi_yavg = np.zeros((mytime.ntime_steady,mygrids.nx))
            tempi_yavg_fit = np.zeros((mytime.ntime_steady,mygrids.nx))
            tprimi_yavg_fit = np.zeros((mytime.ntime_steady,mygrids.nx))
            splinerep_tempi_yavg = []
            fac = myin['theta_grid_parameters']['qinp']/(myin['theta_grid_parameters']['rhoc']*myout['drhodpsi'])
            for it in range(mytime.it_min,mytime.it_max):
                for ix in range(mygrids.nx):
                    tempi_yavg[it_steady,ix] = simps(np.squeeze(tempi_full_fft[it_steady,ix,:]),mygrids.ygrid) \
                            / (mygrids.ygrid[-1]-mygrids.ygrid[0])
                ft = np.squeeze(tempi_yavg[it_steady,:])
                spl_smooth = np.max(ft)/smoothfac
                splinerep_tempi_yavg.append(interpolate.splrep(mygrids.xgrid, ft, s=spl_smooth))
                tempi_yavg_fit[it_steady,:] = interpolate.splev(mygrids.xgrid, splinerep_tempi_yavg[-1])
                tprimi_yavg_fit[it_steady,:] = -1*fac/tempi_yavg_fit[it_steady,:] * \
                        interpolate.splev(mygrids.xgrid, splinerep_tempi_yavg[-1], der=1)
                it_steady += 1


            # Make a movie of it
            opts['ylim'] = [np.amin(tempi_yavg),np.amax(tempi_yavg)]
            opts['ylabel'] = '$\\langle\\delta T_i\\rangle_y$ [$\\rho_{{\\star}} T_i$]'
            opts['file_name'] = run.fnames[ifile]+ '_tempi_yavg_vs_x'
            plt_opts['color'] = gplot.mybluestd
            pyfilm.pyfilm.make_film_1d(mygrids.xgrid, tempi_yavg,
                                       plot_options = plt_opts,
                                       options = opts)
            # And one of its spline fit
            opts['ylim'] = [np.amin(tempi_yavg_fit),np.amax(tempi_yavg_fit)]
            opts['ylabel'] = '$\\langle\\delta T_i\\rangle_y$ [$\\rho_{{\\star}} T_i$]'
            opts['file_name'] = run.fnames[ifile]+ '_tempi_yavg_spline_vs_x'
            plt_opts['color'] = gplot.myredstd
            pyfilm.pyfilm.make_film_1d(mygrids.xgrid, tempi_yavg_fit,
                                       plot_options = plt_opts,
                                       options = opts)
            # And one of the derivative of its spline fit
            opts['ylim'] = [-100,100]
            opts['ylabel'] = '$-\\frac{1}{\\langle\\delta T_i\\rangle_y}\\frac{\\partial \\langle\\delta T_i\\rangle_y}{\\partial r_\\psi}$ [1/$\\rho_i$]'
            opts['file_name'] = run.fnames[ifile]+ '_tprimi_yavg_spline_vs_x'
            plt_opts['color'] = gplot.myredstd
            pyfilm.pyfilm.make_film_1d(mygrids.xgrid, tprimi_yavg_fit,
                                       plot_options = plt_opts,
                                       options = opts)

            plt.rc('font', size=30)




        # Make 1D movies of zonal quantities vs x
        if make_mov_zonal:

            # Options for 1D movies
            plt_opts = {'marker': None,
                        'linewidth': 2,
                        'linestyle': '-'}

            opts = {'xlabel':'$x /\\rho_i$',
                    'title':['$t={:.2f}$'.format(mytime.time[it])\
                             +' [$a/v_{th}$]'\
                             for it in range(mytime.it_min, mytime.it_max)],
                    'xlim':[np.amin(mygrids.xgrid),np.amax(mygrids.xgrid)],
                    'film_dir': run.out_dir,
                    'grid': True,
                    'fps': fps}

            plt.rc('font', size=18)

            # NDCTEST
            #x = np.linspace(0,2*pi)
            #nt = 100
            #y = np.zeros((nt,x.size))
            #for it in range(nt):
            #    y[it,:] = np.sin(x+pi/nt*it)
            #print('Done')
            #
            #pyfilm.make_film_1d(x, y, options={'encoder':'ffmpeg','file_name':'test'})
            # NDCTEST

            # Movie of zonal field vs x
            opts['ylim'] = [np.amin(field_zonal),np.amax(field_zonal)]
            opts['ylabel'] = '$\\varphi_{{Z}}$ [$\\rho_{{\\star}} T_i/e$]'
            opts['file_name'] = run.fnames[ifile]+ '_zonal_field_vs_x'
            plt_opts['color'] = gplot.mybluestd
            pyfilm.pyfilm.make_film_1d(mygrids.xgrid, field_zonal,
                                       plot_options = plt_opts,
                                       options = opts)
            # And of its spline fit
            opts['ylim'] = [np.amin(field_zonal_fit),np.amax(field_zonal_fit)]
            opts['ylabel'] = '$\\varphi_{{Z}}$ [$\\rho_{{\\star}} T_i/e$]'
            opts['file_name'] = run.fnames[ifile]+ '_zonal_field_spline_vs_x'
            plt_opts['color'] = gplot.myredstd
            pyfilm.pyfilm.make_film_1d(mygrids.xgrid_fine, field_zonal_fit,
                                       plot_options = plt_opts,
                                       options = opts)

            # Movie of zonal flow vs x
            opts['ylim'] = [np.amin(flow_zonal),np.amax(flow_zonal)]
            opts['ylabel'] = '$-\\sum_{k_x}ik_x\\hat{\\varphi}_{Z} e^{ik_x x}$ [$T_i/(ea)$]'
            opts['file_name'] = run.fnames[ifile]+ '_zonal_flow_vs_x'
            plt_opts['color'] = gplot.mybluestd
            pyfilm.pyfilm.make_film_1d(mygrids.xgrid, flow_zonal,
                                       plot_options = plt_opts,
                                       options = opts)
            # And of its spline fit
            opts['ylim'] = [np.amin(flow_zonal_fit),np.amax(flow_zonal_fit)]
            opts['ylabel'] = '$\\varphi_{{Z}}$ [$\\rho_{{\\star}} T_i/e$]'
            opts['file_name'] = run.fnames[ifile]+ '_zonal_flow_spline_vs_x'
            plt_opts['color'] = gplot.myredstd
            pyfilm.pyfilm.make_film_1d(mygrids.xgrid_fine, flow_zonal_fit,
                                       plot_options = plt_opts,
                                       options = opts)

            # Movie of zonal shear vs x
            opts['ylim'] = [np.amin(shear_zonal),np.amax(shear_zonal)]
            opts['ylabel'] = '$\\gamma_{Z}$ [$v_{th}/a$]'
            opts['file_name'] = run.fnames[ifile]+ '_zonal_shear_vs_x'
            plt_opts['color'] = gplot.mybluestd
            pyfilm.pyfilm.make_film_1d(mygrids.xgrid, shear_zonal,
                                       plot_options = plt_opts,
                                       options = opts)
            # Of its spline fit
            opts['ylim'] = [np.amin(shear_zonal_fit),np.amax(shear_zonal_fit)]
            opts['ylabel'] = '$\\gamma_{Z}$ [$v_{th}/a$]'
            opts['file_name'] = run.fnames[ifile]+ '_zonal_shear_spline_vs_x'
            plt_opts['color'] = gplot.myredstd
            pyfilm.pyfilm.make_film_1d(mygrids.xgrid_fine, shear_zonal_fit,
                                       plot_options = plt_opts,
                                       options = opts)
            # And of its spline fit + gexb
            if flow_shear:
                opts['ylim'] = [np.amin(sheartot_zonal_fit),np.amax(sheartot_zonal_fit)]
                opts['ylabel'] = '$\\gamma_{Z}+\\gamma_E$ [$v_{th}/a$]'
                opts['file_name'] = run.fnames[ifile]+ '_zonal_sheartot_spline_vs_x'
                plt_opts['color'] = gplot.myredstd
                pyfilm.pyfilm.make_film_1d(mygrids.xgrid_fine, sheartot_zonal_fit,
                                           plot_options = plt_opts,
                                           options = opts)

            plt.rc('font', size=30)







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
