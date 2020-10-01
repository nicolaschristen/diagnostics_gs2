import gs2_plotting as gplot
import gs2_fft as gfft
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pyfilm
import pickle
from math import pi
from scipy import interpolate



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
    smoothfac = 500
    
    # Make movies in real space?
    make_movies = True
    make_mov_phi = True
    make_mov_densi = False
    make_mov_dense = False
    make_mov_tempi = False
    make_mov_tempe = False
    
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

        # Then average over time
        field_zonal_avg = np.zeros(mygrids.nx)
        for ix in range(mygrids.nx):
            field_zonal_avg[ix] = mytime.timeavg(np.squeeze(field_zonal[:,ix]), use_ft_full=True)

        # Finally, reorder to have growing x axis
        field_zonal_avg = np.concatenate((field_zonal_avg[mygrids.nxmid:],\
                                          field_zonal_avg[:mygrids.nxmid]))



        # Keeping only |kx| < kxcut

        field_zonal_cut = np.zeros((mytime.ntime_steady, nxcut))
        it_steady = 0
        for it in range(mytime.it_min,mytime.it_max):
            ft = np.squeeze(phi_all_t_cut[it,0,:])
            field_zonal_cut[it_steady,:] = np.real(np.fft.ifft(ft)) * nxcut

        field_zonal_avg_cut = np.zeros(nxcut)
        for ix in range(nxcut):
            field_zonal_avg_cut[ix] = mytime.timeavg(np.squeeze(field_zonal_cut[:,ix]), use_ft_full=True)

        field_zonal_avg_cut = np.concatenate((field_zonal_avg_cut[nxmid_cut:],\
                                             field_zonal_avg_cut[:nxmid_cut]))



        # Fitting the full x-profile with splines

        if to_fit == 'field':
            spl_smooth = np.max(field_zonal_avg)/smoothfac
            splinerep = interpolate.splrep(mygrids.xgrid, field_zonal_avg, s=spl_smooth)
            field_zonal_avg_fit = interpolate.splev(mygrids.xgrid, splinerep)
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

        # Then average over time
        flow_zonal_avg = np.zeros(mygrids.nx)
        for ix in range(mygrids.nx):
            flow_zonal_avg[ix] = mytime.timeavg(np.squeeze(flow_zonal[:,ix]), use_ft_full=True)

        # Finally, reorder to have growing x axis
        flow_zonal_avg = np.concatenate((flow_zonal_avg[mygrids.nxmid:],\
                                          flow_zonal_avg[:mygrids.nxmid]))




        # Keeping only |kx| < kxcut

        flow_zonal_cut = np.zeros((mytime.ntime_steady, nxcut))
        it_steady = 0
        for it in range(mytime.it_min,mytime.it_max):
            ft = -1j * kx_gs2_cut * np.squeeze(phi_all_t_cut[it,0,:])
            flow_zonal_cut[it_steady,:] = np.real(np.fft.ifft(ft)) * nxcut

        flow_zonal_avg_cut = np.zeros(nxcut)
        for ix in range(nxcut):
            flow_zonal_avg_cut[ix] = mytime.timeavg(np.squeeze(flow_zonal_cut[:,ix]), use_ft_full=True)

        flow_zonal_avg_cut = np.concatenate((flow_zonal_avg_cut[nxmid_cut:],\
                                             flow_zonal_avg_cut[:nxmid_cut]))



        # Fitting the full x-profile with splines

        # Either take a derivative from the fitted field
        if to_fit == 'field':
            flow_zonal_avg_fit = interpolate.splev(mygrids.xgrid, splinerep, der=1)
            label_flow_fit = 'd/dx of fitted field'
        # Or if the flow is smooth enough, fit the flow itself
        elif to_fit == 'flow':
            spl_smooth = np.max(flow_zonal_avg)/smoothfac
            splinerep = interpolate.splrep(mygrids.xgrid, flow_zonal_avg, s=spl_smooth)
            flow_zonal_avg_fit = interpolate.splev(mygrids.xgrid, splinerep)
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

        # Then average over time
        shear_zonal_avg = np.zeros(mygrids.nx)
        for ix in range(mygrids.nx):
            shear_zonal_avg[ix] = mytime.timeavg(np.squeeze(shear_zonal[:,ix]), use_ft_full=True)

        # Finally, reorder to have growing x axis
        shear_zonal_avg = np.concatenate((shear_zonal_avg[mygrids.nxmid:],\
                                          shear_zonal_avg[:mygrids.nxmid]))



        # Keeping only |kx| < kxcut

        shear_zonal_cut = np.zeros((mytime.ntime_steady, nxcut))
        it_steady = 0
        for it in range(mytime.it_min,mytime.it_max):
            ft = kx_gs2_cut**2 * np.squeeze(phi_all_t_cut[it,0,:])
            shear_zonal_cut[it_steady,:] = fac * np.real(np.fft.ifft(ft)) * nxcut

        shear_zonal_avg_cut = np.zeros(nxcut)
        for ix in range(nxcut):
            shear_zonal_avg_cut[ix] = mytime.timeavg(np.squeeze(shear_zonal_cut[:,ix]), use_ft_full=True)

        shear_zonal_avg_cut = np.concatenate((shear_zonal_avg_cut[nxmid_cut:],\
                                              shear_zonal_avg_cut[:nxmid_cut]))



        # Fitting the full x-profile with splines

        # Either take the second derivative from the fitted field
        if to_fit == 'field':
            shear_zonal_avg_fit = fac * interpolate.splev(mygrids.xgrid, splinerep, der=2)
            label_shear_fit = 'd^2/dx^2 of fitted field'
        # Or take a derivative from the fitted flow
        elif to_fit == 'flow':
            shear_zonal_avg_fit = fac * interpolate.splev(mygrids.xgrid, splinerep, der=1)
            label_shear_fit = 'd/dx of fitted flow'
        # Or if the shear is smooth enough, fit the shear itself
        elif to_fit == 'shear':
            spl_smooth = np.max(shear_zonal_avg)/smoothfac
            splinerep = interpolate.splrep(mygrids.xgrid, shear_zonal_avg, s=spl_smooth)
            shear_zonal_avg_fit = interpolate.splev(mygrids.xgrid, splinerep)
            label_shear_fit = 'spline fit'
        




        #------------#
        #  Plotting  #
        #------------#


        tmp_pdf_id = 1
        pdflist = []
        tmp_pdf_id_fromSum = 1
        pdflist_fromSum = []

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
            lfit, = plt.plot(mygrids.xgrid, field_zonal_avg_fit, linewidth=2, color='r', label=label_field_fit)
        plt.legend()
        plt.xlabel('$x/\\rho_i$')
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
            lfit, = plt.plot(mygrids.xgrid, flow_zonal_avg_fit, linewidth=2, color='r', label=label_flow_fit)
        plt.legend()
        plt.xlabel('$x/\\rho_i$')
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
        lfit, = plt.plot(mygrids.xgrid, shear_zonal_avg_fit, linewidth=2, color='r', label=label_shear_fit)
        plt.legend()
        plt.xlabel('$x/\\rho_i$')
        ylab = '$\\langle\\gamma_{Z}\\rangle_{t}$'\
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

        to_add = { 'shear_zonal':shear_zonal_avg,
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
        if make_mov_phi:

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
        if make_mov_densi:

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
        if make_mov_dense:

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
        if make_mov_tempi:

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
        if make_mov_tempe:

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

        to_add = { 'time':mytime.time_steady,
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
