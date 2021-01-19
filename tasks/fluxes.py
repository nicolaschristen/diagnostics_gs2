from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
import pickle
import copy as cp
from math import pi

import gs2_plotting as gplot
from plot_phi2_vs_time import plot_phi2_ky_vs_t

def my_single_task(ifile,run,myin,myout,mygrids,mytime,myfields,stitching=False):

    # Compute and save to dat file
    if not run.only_plot:
    
        nvpa = mygrids.nvpa
        ntheta = mygrids.ntheta
        nx = mygrids.nx
        ny = mygrids.ny
        nxmid = mygrids.nxmid

        islin = False
        if myin['nonlinear_terms_knobs']['nonlinear_mode']=='off':
            islin = True

        has_flowshear = False
        try:
            if myin['dist_fn_knobs']['g_exb'] != 0.:
                has_flowshear = True
        except:
            pass # ignore error if g_exb not defined

        nspec = myin['species_knobs']['nspec']
        spec_names = []
        for ispec in range(nspec):
            spec_names.append(myin['species_parameters_'+str(ispec+1)]['type'])
        naky = (myin['kt_grids_box_parameters']['ny']-1)//3 + 1
    
        theta = mygrids.theta
        theta0 = mygrids.theta0
        kx = mygrids.kx
        ky = mygrids.ky
        jtwist = myin['kt_grids_box_parameters']['jtwist']

        time = mytime.time
        time_steady = mytime.time_steady
        it_min = mytime.it_min
        it_max = mytime.it_max

        phi_bytheta_tfinal = myfields.phi_bytheta_tfinal
        
        phi2_avg = myfields.phi2_avg
        if myout['phi2_by_ky_present']:
            phi2_by_ky = myout['phi2_by_ky']
        
        if myout['phi2_by_mode_present']:
            phi2_kxky = np.concatenate((myout['phi2_by_mode'][:,:,nxmid:],myout['phi2_by_mode'][:,:,:nxmid]),axis=2)
            mydim = (mygrids.ny,mygrids.nx)
            phi2_kxky_tavg = np.zeros(mydim, dtype=float)
            phi2_kxky_nrm = np.zeros((mytime.ntime,ny,nx), dtype=float)
            phi2_kxky_tauwinavg = np.zeros((mytime.ntauwin,ny,nx))
            phi2_kxky_nrm_tauwinavg = np.zeros((mytime.ntauwin,ny,nx))
            phi2_kxky_tauwinsig = np.zeros((mytime.ntauwin,ny,nx))
            phi2_kxky_nrm_tauwinsig = np.zeros((mytime.ntauwin,ny,nx))
            for ik in range(ny):
                for it in range(nx):
                    phi2_kxky_tavg[ik,it] = mytime.timeavg(phi2_kxky[:,ik,it])
                    phi2_kxky_nrm[:,ik,it] = phi2_kxky[:,ik,it] / phi2_kxky_tavg[ik,it]
                    phi2_kxky_tauwinavg[:,ik,it] = mytime.tauwin_avg(phi2_kxky[:,ik,it])
                    phi2_kxky_nrm_tauwinavg[:,ik,it] = phi2_kxky_tauwinavg[:,ik,it] / phi2_kxky_tavg[ik,it]
                    phi2_kxky_tauwinsig[:,ik,it] = mytime.tauwin_sigma(phi2_kxky[:,ik,it], phi2_kxky_tauwinavg[:,ik,it])
                    phi2_kxky_nrm_tauwinsig[:,ik,it] = phi2_kxky_tauwinsig[:,ik,it] / phi2_kxky_tavg[ik,it]
        else:
            mydim = (mytime.ntime,mygrids.ny,mygrids.nx)
            phi2_kxky = np.zeros(mydim, dtype=float)
            phi2_kxky_nrm = np.zeros(mydim, dtype=float)
            mydim = (mygrids.ny,mygrids.nx)
            phi2_kxky_tavg = np.zeros(mydim, dtype=float)
            phi2_kxky_tauwinavg = np.zeros((mytime.ntauwin,ny,nx))
            phi2_kxky_tauwinsig = np.zeros((mytime.ntauwin,ny,nx))
            phi2_kxky_nrm_tauwinavg = np.zeros((mytime.ntauwin,ny,nx))
            phi2_kxky_nrm_tauwinsig = np.zeros((mytime.ntauwin,ny,nx))

        if myout['es_part_flux_present']:
            pflx = myout['es_part_flux']
        else:
            pflx = np.arange(1,dtype=float)

        if myout['es_heat_flux_present']:
            qflx = myout['es_heat_flux']
        else:
            qflx = np.arange(1,dtype=float)

        if myout['es_mom_flux_present']:
            vflx = myout['es_mom_flux']
        else:
            vflx = np.arange(1,dtype=float)

        if myout['es_energy_exchange_present']:
            xchange = myout['es_energy_exchange']
        else:
            xchange = np.arange(1,dtype=float)

        if myout['es_heat_flux_present'] and myout['es_mom_flux_present']:
            # avoid divide by zero with qflx
            # in this case, set pi/Q = 0
            dum = np.copy(qflx)
            zerotest = dum==0
            dum[zerotest] = vflx[zerotest]*100000
            pioq = np.copy(np.divide(vflx,dum))
        else:
            pioq = np.arange(1,dtype=float)

        # need to multiply this by rhoc/(g_exb*rmaj**2)
        #prandtl = np.copy(vflx[:,0]*tprim[0]/(qflx[:,0]*q)

        if myout['es_part_flux_by_mode_present']:
            pflx_kxky = np.concatenate((myout['es_part_flux_by_mode'][:,:,:,nxmid:],myout['es_part_flux_by_mode'][:,:,:,:nxmid]),axis=3)
            pflx_kxky_tavg = np.arange(myout['nspec']*nx*ny,dtype=float).reshape(myout['nspec'],ny,nx)
            for ispec in range(myout['nspec']):
                for ik in range(ny):
                    for it in range(nx):
                        pflx_kxky_tavg[ispec,ik,it] = mytime.timeavg(pflx_kxky[:,ispec,ik,it])
        else:
            mydim = (mytime.ntime,myout['nspec'],mygrids.ny,mygrids.nx)
            pflx_kxky = np.zeros(mydim, dtype=float)
            mydim = (myout['nspec'],mygrids.ny,mygrids.nx)
            pflx_kxky_tavg = np.zeros(mydim, dtype=float)

        if myout['es_heat_flux_by_mode_present']:
            qflx_kxky = np.concatenate((myout['es_heat_flux_by_mode'][:,:,:,nxmid:],myout['es_heat_flux_by_mode'][:,:,:,:nxmid]),axis=3)
            qflx_kxky_tavg = np.copy(pflx_kxky_tavg)
            for ispec in range(myout['nspec']):
                for ik in range(ny):
                    for it in range(nx):
                        qflx_kxky_tavg[ispec,ik,it] = mytime.timeavg(qflx_kxky[:,ispec,ik,it])
        else:
            mydim = (mytime.ntime,myout['nspec'],mygrids.ny,mygrids.nx)
            qflx_kxky = np.zeros(mydim, dtype=float)
            mydim = (myout['nspec'],mygrids.ny,mygrids.nx)
            qflx_kxky_tavg = np.zeros(mydim, dtype=float)

        if myout['es_mom_flux_by_mode_present']:
            vflx_kxky = np.concatenate((myout['es_mom_flux_by_mode'][:,:,:,nxmid:],myout['es_mom_flux_by_mode'][:,:,:,:nxmid]),axis=3)
            vflx_kxky_tavg = np.copy(pflx_kxky_tavg)
            for ispec in range(myout['nspec']):
                for ik in range(ny):
                    for it in range(nx):
                        vflx_kxky_tavg[ispec,ik,it] = mytime.timeavg(vflx_kxky[:,ispec,ik,it])
        else:
            mydim = (mytime.ntime,myout['nspec'],mygrids.ny,mygrids.nx)
            vflx_kxky = np.zeros(mydim, dtype=float)
            mydim = (myout['nspec'],mygrids.ny,mygrids.nx)
            vflx_kxky_tavg = np.zeros(mydim, dtype=float)

        pflx_vpth = myout['es_part_sym']
        pflx_vpth_tavg = np.arange(myout['nspec']*nvpa*ntheta,dtype=float).reshape(myout['nspec'],nvpa,ntheta)
        if myout['es_part_sym_present']:
            for ispec in range(myout['nspec']):
                for iv in range(nvpa):
                    for ig in range(ntheta):
                        pflx_vpth_tavg[ispec,iv,ig] = mytime.timeavg(pflx_vpth[:,ispec,iv,ig])
        
        qflx_vpth = myout['es_heat_sym']
        qflx_vpth_tavg = np.copy(pflx_vpth_tavg)
        if myout['es_heat_sym_present']:
            for ispec in range(myout['nspec']):
                for iv in range(nvpa):
                    for ig in range(ntheta):
                        qflx_vpth_tavg[ispec,iv,ig] = mytime.timeavg(qflx_vpth[:,ispec,iv,ig])
        
        vflx_vpth = myout['es_mom_sym'] 
        vflx_vpth_tavg = np.copy(pflx_vpth_tavg)
        if myout['es_mom_sym_present']:
            for ispec in range(myout['nspec']):
                for iv in range(nvpa):
                    for ig in range(ntheta):
                        vflx_vpth_tavg[ispec,iv,ig] = mytime.timeavg(vflx_vpth[:,ispec,iv,ig])
 
        # Save computed quantities
        datfile_name = run.out_dir + run.fnames[ifile] + '.fluxes.dat'
        mydict = {
                'pflx':pflx,'qflx':qflx,'vflx':vflx,'xchange':xchange,
                'pflx_kxky':pflx_kxky,'qflx_kxky':qflx_kxky,'vflx_kxky':vflx_kxky,
                'pflx_kxky_tavg':pflx_kxky_tavg,'qflx_kxky_tavg':qflx_kxky_tavg,'vflx_kxky_tavg':vflx_kxky_tavg,
                'pflx_vpth_tavg':pflx_vpth_tavg,'qflx_vpth_tavg':qflx_vpth_tavg,'vflx_vpth_tavg':vflx_vpth_tavg,
                'pioq':pioq,
                'nvpa':nvpa,'ntheta':ntheta,'nx':nx,'ny':ny,'nxmid':nxmid,
                'naky':naky,'kx':kx,'ky':ky,'theta':theta,'theta0':theta0,'jtwist':jtwist,
                'time':time,'time_steady':time_steady,'it_min':it_min,'it_max':it_max,
                'phi_bytheta_tfinal':phi_bytheta_tfinal, 'phi2_avg':phi2_avg,'phi2_by_ky':phi2_by_ky,
                'phi2_kxky_tavg':phi2_kxky_tavg,'phi2_kxky':phi2_kxky,'phi2_kxky_nrm':phi2_kxky_nrm,
                'phi2_kxky_tauwinavg':phi2_kxky_tauwinavg, 'phi2_kxky_tauwinsig':phi2_kxky_tauwinsig,
                'phi2_kxky_nrm_tauwinavg':phi2_kxky_nrm_tauwinavg, 'phi2_kxky_nrm_tauwinsig':phi2_kxky_nrm_tauwinsig,
                'islin':islin,'nspec':nspec,'spec_names':spec_names,'has_flowshear':has_flowshear}
        with open(datfile_name,'wb') as datfile:
            pickle.dump(mydict,datfile)

        # Save time obj
        datfile_name = run.out_dir + run.fnames[ifile] + '.time.dat'
        with open(datfile_name,'wb') as datfile:
            pickle.dump(mytime,datfile)

        # Save grid obj
        datfile_name = run.out_dir + run.fnames[ifile] + '.grids.dat'
        with open(datfile_name,'wb') as datfile:
            pickle.dump(mygrids,datfile)

    # Read from dat files
    else:

        datfile_name = run.out_dir + run.fnames[ifile] + '.fluxes.dat'
        with open(datfile_name,'rb') as datfile:
            mydict = pickle.load(datfile)

        datfile_name = run.out_dir + run.fnames[ifile] + '.time.dat'
        with open(datfile_name,'rb') as datfile:
            mytime = pickle.load(datfile)

        datfile_name = run.out_dir + run.fnames[ifile] + '.grids.dat'
        with open(datfile_name,'rb') as datfile:
            mygrids = pickle.load(datfile)
    
    if not run.no_plot and not stitching: # plot fluxes for this single file
        
        plot_fluxes(ifile,run,mytime,mydict)

def multi_task(run):

    # Set to True to plot simulations with different t[0]
    # and remove their time offsets.
    start_together = True

    leg_fontsize = 8

    nfile = len(run.fnames)

    fluxes = [{} for ifile in range(nfile)]
    times = [{} for ifile in range(nfile)]
    print('Running MULTI')

    for ifile in range(nfile):

        datfile_name = run.out_dir + run.fnames[ifile] + '.fluxes.dat'
        with open(datfile_name,'rb') as datfile:
            fluxes[ifile] = pickle.load(datfile)

        datfile_name = run.out_dir + run.fnames[ifile] + '.time.dat'
        with open(datfile_name,'rb') as datfile:
            times[ifile] = pickle.load(datfile)

    # Plotting phi spectra vs kx and ky
    plot_spectra(run, fluxes)

    # Plotting phi2 vs theta-theta0 at final time,
    # for ky~0.5 and ky~1.0
    kyplt = [0.5, 1.0]
    tt0plt = [0, pi]
    plot_phi2_vs_bloon(run, times, fluxes, kyplt, tt0plt)

    # Plotting time traces of phi2 for all modes
    fldname = 'phi2_kxky_nrm'
    zttl_nrm = '$\\langle\\vert\\hat{\\varphi}\\vert^2\\rangle_{\\theta, t}$'
    zttl = '$\\langle\\vert\\hat{\\varphi}\\vert^2\\rangle_\\theta$ / ' + zttl_nrm
    pltname = 'phi2all'
    zmin = 0.1
    zmax = 10.0
    plot_allmodes_vs_t(run, times, fluxes, fldname, zttl, pltname, zmin=zmin, zmax=zmax)

    # Plotting time traces of tauwinavg(phi2) for all modes
    fldname = 'phi2_kxky_nrm_tauwinavg'
    zttl_nrm = '$\\langle\\vert\\hat{\\varphi}\\vert^2\\rangle_{\\theta, t}$'
    zttl = '$\\overline{\\langle\\vert\\hat{\\varphi}\\vert^2\\rangle_\\theta}$ / ' + zttl_nrm
    pltname = 'phi2all_tauwinavg'
    zmin = 0.5
    zmax = 2.0
    plot_allmodes_vs_t(run, times, fluxes, fldname, zttl, pltname, zmin=zmin, zmax=zmax, tauwindow=True)

    # Plotting time traces of sigma(phi2) for all modes
    fldname = 'phi2_kxky_nrm_tauwinsig'
    zttl_nrm = '$\\langle\\vert\\hat{\\varphi}\\vert^2\\rangle_{\\theta, t}$'
    zttl = '$\\sigma (\\langle\\vert\\hat{\\varphi}\\vert^2\\rangle_\\theta )$ / ' + zttl_nrm
    pltname = 'phi2all_tauwinsig'
    zmin = 0.2
    zmax = 2.0
    plot_allmodes_vs_t(run, times, fluxes, fldname, zttl, pltname, zmin=zmin, zmax=zmax, tauwindow=True)


    # Plotting fluxes vs time
    for ispec in range(len(fluxes[0]['spec_names'])):

        time_offset = np.zeros(nfile)
        for ifile in range(nfile):
            time_offset[ifile] = times[ifile].time[0]
            if start_together:
                times[ifile].time -= time_offset[ifile]

        specname = fluxes[0]['spec_names'][ispec]
        if specname == 'ion':
            speclab = 'i'
        elif specname == 'electron':
            speclab = 'e'

        plt.figure()
        tmp_pdf_id = 1
        pdflist = []
        labs = []

        # Particle flux vs t, linear scale
        for ifile in range(nfile):
            plt.plot(times[ifile].time, fluxes[ifile]['pflx'][:,ispec])
            labs.append(run.flabels[ifile])
        plt.xlabel('$t$ [$a/v_th$]')
        plt.ylabel('$\Gamma_'+speclab+'/\Gamma_{gB}$')
        leg = plt.legend(labs,
                         prop={'size': leg_fontsize},
                         ncol=2,
                         frameon=True,
                         fancybox=False,
                         framealpha=1.0,
                         handlelength=1)
        plt.grid(True)
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

        # Particle flux vs t, semilogy scale
        for ifile in range(nfile):
            plt.semilogy(times[ifile].time, fluxes[ifile]['pflx'][:,ispec])
            labs.append(run.flabels[ifile])
        plt.xlabel('$t$ [$a/v_th$]')
        plt.ylabel('$\Gamma_'+speclab+'/\Gamma_{gB}$')
        leg = plt.legend(labs,
                         prop={'size': leg_fontsize},
                         ncol=2,
                         frameon=True,
                         fancybox=False,
                         framealpha=1.0,
                         handlelength=1)
        plt.grid(True)
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

        # Heat flux vs t, linear scale
        for ifile in range(nfile):
            plt.plot(times[ifile].time, fluxes[ifile]['qflx'][:,ispec])
            labs.append(run.flabels[ifile])
        plt.xlabel('$t$ [$a/v_th$]')
        plt.ylabel('$Q_'+speclab+'/Q_{gB}$')
        leg = plt.legend(labs,
                         prop={'size': leg_fontsize},
                         ncol=2,
                         frameon=True,
                         fancybox=False,
                         framealpha=1.0,
                         handlelength=1)
        plt.grid(True)
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

        # Heat flux vs t, semilogy scale
        for ifile in range(nfile):
            plt.semilogy(times[ifile].time, fluxes[ifile]['qflx'][:,ispec])
            labs.append(run.flabels[ifile])
        plt.xlabel('$t$ [$a/v_th$]')
        plt.ylabel('$Q_'+speclab+'/Q_{gB}$')
        leg = plt.legend(labs,
                         prop={'size': leg_fontsize},
                         ncol=2,
                         frameon=True,
                         fancybox=False,
                         framealpha=1.0,
                         handlelength=1)
        plt.grid(True)
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

        # Momentum flux vs t, linear scale
        for ifile in range(nfile):
            plt.plot(times[ifile].time, fluxes[ifile]['vflx'][:,ispec])
            labs.append(run.flabels[ifile])
        plt.xlabel('$t$ [$a/v_th$]')
        plt.ylabel('$\Pi_'+speclab+'/\Pi_{gB}$')
        leg = plt.legend(labs,
                         prop={'size': leg_fontsize},
                         ncol=2,
                         frameon=True,
                         fancybox=False,
                         framealpha=1.0,
                         handlelength=1)
        plt.grid(True)
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

        # Momentum flux vs t, semilogy scale
        for ifile in range(nfile):
            plt.semilogy(times[ifile].time, np.abs(fluxes[ifile]['vflx'][:,ispec]))
            labs.append(run.flabels[ifile])
        plt.xlabel('$t$ [$a/v_th$]')
        plt.ylabel('$\\vert\Pi_'+speclab+'\\vert/\Pi_{gB}$')
        leg = plt.legend(labs,
                         prop={'size': leg_fontsize},
                         ncol=2,
                         frameon=True,
                         fancybox=False,
                         framealpha=1.0,
                         handlelength=1)
        plt.grid(True)
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

        # Ratio of momentum to heat flux vs t
        for ifile in range(nfile):
            itmin = times[ifile].it_min
            itmax = times[ifile].it_max
            plt.plot(times[ifile].time_steady, fluxes[ifile]['pioq'][itmin:itmax,ispec])
            labs.append(run.flabels[ifile])
        plt.xlabel('$t$ [$a/v_th$]')
        plt.ylabel('$\Pi_'+speclab+'/Q_'+speclab+'$')
        leg = plt.legend(labs,
                         prop={'size': leg_fontsize},
                         ncol=2,
                         frameon=True,
                         fancybox=False,
                         framealpha=1.0,
                         handlelength=1)
        plt.grid(True)
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

        merged_pdfname = 'fluxes_' + specname + '_vs_t_' + run.scan_name
        gplot.merge_pdfs(pdflist, merged_pdfname, run)

        if start_together:
            for ifile in range(nfile):
                times[ifile].time += time_offset[ifile]


def stitching_fluxes(run):

    # Only executed if we want to plot the data
    if run.no_plot:
        return

    Nfile = len(run.fnames)
    full_fluxes = [dict() for ifile in range(Nfile)]
    full_time = [dict() for ifile in range(Nfile)]
    full_grids = [dict() for ifile in range(Nfile)]

    # Reading .dat file for each run
    # and calc how long the stitched array will be
    Nt_tot = 0
    for ifile in range(Nfile):
        datfile_name = run.out_dir + run.fnames[ifile] + '.fluxes.dat'
        with open(datfile_name,'rb') as datfile:
            full_fluxes[ifile] = pickle.load(datfile)
        datfile_name = run.out_dir + run.fnames[ifile] + '.time.dat'
        with open(datfile_name,'rb') as datfile:
            full_time[ifile] = pickle.load(datfile)
        Nt_tot += full_fluxes[ifile]['pflx'].shape[0]
        if ifile > 0:
            Nt_tot -= 1 # removing duplicate at restart point
    
    twin = run.twin

    # A lot of stuff is the same for all runs
    islin = full_fluxes[0]['islin']
    has_flowshear = full_fluxes[0]['has_flowshear']
    taumax = full_time[0].taumax
    nspec = full_fluxes[0]['nspec']
    spec_names = full_fluxes[0]['spec_names']
    nx = full_fluxes[0]['nx']
    ny = full_fluxes[0]['ny']
    naky = full_fluxes[0]['naky']
    kx = full_fluxes[0]['kx']
    ky = full_fluxes[0]['ky']
    jtwist = full_fluxes[0]['jtwist']
    theta = full_fluxes[0]['theta']
    theta0 = full_fluxes[0]['theta0']
    datfile_name = run.out_dir + run.fnames[0] + '.grids.dat'
    with open(datfile_name,'rb') as datfile:
        my_grids = pickle.load(datfile)

    # Quantities evaluated at the last time step
    phi_bytheta_tfinal = full_fluxes[-1]['phi_bytheta_tfinal']

    # Stitching the arrays together
    stitch_my_time = cp.deepcopy(full_time[0])
    stitch_my_time.ntime = Nt_tot
    stitch_my_time.time = np.zeros(Nt_tot)

    stitch_pflx = np.zeros((Nt_tot,nspec))
    stitch_qflx = np.zeros((Nt_tot,nspec))
    stitch_vflx = np.zeros((Nt_tot,nspec))
    stitch_pioq = np.zeros((Nt_tot,nspec))

    stitch_pflx_tavg = np.zeros(nspec)
    stitch_qflx_tavg = np.zeros(nspec)
    stitch_vflx_tavg = np.zeros(nspec)
    stitch_pioq_tavg = np.zeros(nspec)
    
    stitch_pflx_kxky = np.zeros((Nt_tot,nspec,ny,nx))
    stitch_qflx_kxky = np.zeros((Nt_tot,nspec,ny,nx))
    stitch_vflx_kxky = np.zeros((Nt_tot,nspec,ny,nx))
    stitch_pflx_kxky_tavg = np.zeros((nspec,ny,nx))
    stitch_qflx_kxky_tavg = np.zeros((nspec,ny,nx))
    stitch_vflx_kxky_tavg = np.zeros((nspec,ny,nx))

    stitch_phi2_avg = np.zeros(Nt_tot)
    stitch_phi2_by_ky = np.zeros((Nt_tot,naky))
    stitch_phi2_kxky = np.zeros((Nt_tot,ny,nx))
    stitch_phi2_kxky_tavg = np.zeros((ny,nx))

    it_tot = 0
    for ifile in range(Nfile):
        if ifile == 0:
            it_range = range(full_time[0].ntime)
        else:
            it_range = range(1,full_time[ifile].ntime) # excluding duplicate when restarting
        for it in it_range:
            stitch_my_time.time[it_tot] = full_time[ifile].time[it]
            for ispec in range(nspec):
                stitch_pflx[it_tot,ispec] = full_fluxes[ifile]['pflx'][it,ispec]
                stitch_qflx[it_tot,ispec] = full_fluxes[ifile]['qflx'][it,ispec]
                stitch_vflx[it_tot,ispec] = full_fluxes[ifile]['vflx'][it,ispec]
                stitch_pioq[it_tot,ispec] = full_fluxes[ifile]['pioq'][it,ispec]
                
                for ikx in range(nx):
                    for iky in range(ny):
                        stitch_pflx_kxky[it_tot,ispec,iky,ikx] = full_fluxes[ifile]['pflx_kxky'][it,ispec,iky,ikx]
                        stitch_qflx_kxky[it_tot,ispec,iky,ikx] = full_fluxes[ifile]['qflx_kxky'][it,ispec,iky,ikx]
                        stitch_vflx_kxky[it_tot,ispec,iky,ikx] = full_fluxes[ifile]['vflx_kxky'][it,ispec,iky,ikx]
                
            for ikx in range(nx):
                for iky in range(ny):
                    stitch_phi2_kxky[it_tot,iky,ikx] = full_fluxes[ifile]['phi2_kxky'][it,iky,ikx]
            
            stitch_phi2_avg[it_tot] = full_fluxes[ifile]['phi2_avg'][it]
            stitch_phi2_by_ky[it_tot,:] = full_fluxes[ifile]['phi2_by_ky'][it,:]
            it_tot += 1

    tmin = stitch_my_time.time[-1] * twin[0]
    stitch_my_time.it_min = 0
    while stitch_my_time.time[stitch_my_time.it_min] < tmin:
        stitch_my_time.it_min += 1
    tmax = stitch_my_time.time[-1] * twin[1]
    stitch_my_time.it_max = 0
    while stitch_my_time.time[stitch_my_time.it_max] < tmax and stitch_my_time.it_max < stitch_my_time.ntime-1:
        stitch_my_time.it_max += 1
    stitch_my_time.time_steady = stitch_my_time.time[stitch_my_time.it_min:stitch_my_time.it_max]
    stitch_my_time.ntime_steady = stitch_my_time.time_steady.size
    stitch_my_time.taumax = taumax
    stitch_my_time.ntauwin = int(stitch_my_time.time[-1]//(0.5*stitch_my_time.taumax)) - 1
    stitch_my_time.t_tauwinavg = stitch_my_time.tauwin_avg(stitch_my_time.time)

    stitch_phi2_kxky_nrm = np.zeros((Nt_tot,ny,nx))
    stitch_phi2_kxky_tauwinavg = np.zeros((stitch_my_time.ntauwin,ny,nx))
    stitch_phi2_kxky_nrm_tauwinavg = np.zeros((stitch_my_time.ntauwin,ny,nx))
    stitch_phi2_kxky_tauwinsig = np.zeros((stitch_my_time.ntauwin,ny,nx))
    stitch_phi2_kxky_nrm_tauwinsig = np.zeros((stitch_my_time.ntauwin,ny,nx))

    # Computing time averaged versions of stitched fluxes vs t
    for ispec in range(nspec):
        stitch_pflx_tavg[ispec] = stitch_my_time.timeavg(stitch_pflx[:,ispec])
        stitch_qflx_tavg[ispec] = stitch_my_time.timeavg(stitch_qflx[:,ispec])
        stitch_vflx_tavg[ispec] = stitch_my_time.timeavg(stitch_vflx[:,ispec])
    # Computing time averaged versions of stitched fluxes vs (kx,ky)
    for ik in range(ny):
        for it in range(nx):
            stitch_phi2_kxky_tavg[ik,it] = stitch_my_time.timeavg(stitch_phi2_kxky[:,ik,it])
            stitch_phi2_kxky_nrm[:,ik,it] = stitch_phi2_kxky[:,ik,it] / stitch_phi2_kxky_tavg[ik,it]
            stitch_phi2_kxky_tauwinavg[:,ik,it] = stitch_my_time.tauwin_avg(stitch_phi2_kxky[:,ik,it])
            stitch_phi2_kxky_nrm_tauwinavg[:,ik,it] = stitch_phi2_kxky_tauwinavg[:,ik,it] / stitch_phi2_kxky_tavg[ik,it]
            stitch_phi2_kxky_tauwinsig[:,ik,it] = stitch_my_time.tauwin_sigma(stitch_phi2_kxky[:,ik,it], stitch_phi2_kxky_tauwinavg[:,ik,it])
            stitch_phi2_kxky_nrm_tauwinsig[:,ik,it] = stitch_phi2_kxky_tauwinsig[:,ik,it] / stitch_phi2_kxky_tavg[ik,it]
            for ispec in range(nspec):
                stitch_pflx_kxky_tavg[ispec,ik,it] = stitch_my_time.timeavg(stitch_pflx_kxky[:,ispec,ik,it])
                stitch_qflx_kxky_tavg[ispec,ik,it] = stitch_my_time.timeavg(stitch_qflx_kxky[:,ispec,ik,it])
                stitch_vflx_kxky_tavg[ispec,ik,it] = stitch_my_time.timeavg(stitch_vflx_kxky[:,ispec,ik,it])

    # Save computed quantities
    stitch_flux_dict = {'pflx':stitch_pflx,'qflx':stitch_qflx,'vflx':stitch_vflx,'pioq':stitch_pioq,
            'nx':nx,'ny':ny,'islin':islin,'has_flowshear':has_flowshear,'nspec':nspec,'spec_names':spec_names,
            'naky':naky,'kx':kx,'ky':ky,'jtwist':jtwist,'theta':theta,'theta0':theta0,
            'phi_bytheta_tfinal':phi_bytheta_tfinal,'phi2_avg':stitch_phi2_avg,'phi2_by_ky':stitch_phi2_by_ky,
            'pflx_kxky_tavg':stitch_pflx_kxky_tavg,'qflx_kxky_tavg':stitch_qflx_kxky_tavg,
            'vflx_kxky_tavg':stitch_vflx_kxky_tavg,
            'phi2_kxky_tavg':stitch_phi2_kxky_tavg,'phi2_kxky':stitch_phi2_kxky,'phi2_kxky_nrm':stitch_phi2_kxky_nrm,
            'phi2_kxky_tauwinavg':stitch_phi2_kxky_tauwinavg, 'phi2_kxky_tauwinsig':stitch_phi2_kxky_tauwinsig,
            'phi2_kxky_nrm_tauwinavg':stitch_phi2_kxky_nrm_tauwinavg, 'phi2_kxky_nrm_tauwinsig':stitch_phi2_kxky_nrm_tauwinsig,
            'pflx_tavg':stitch_pflx_tavg,'qflx_tavg':stitch_qflx_tavg,
            'vflx_tavg':stitch_vflx_tavg}
    datfile_name = run.out_dir + run.scan_name + '.fluxes.dat'
    with open(datfile_name,'wb') as datfile:
        pickle.dump(stitch_flux_dict,datfile)

    # Save time obj
    datfile_name = run.out_dir + run.scan_name + '.time.dat'
    with open(datfile_name,'wb') as datfile:
        pickle.dump(stitch_my_time,datfile)

    # Plotting the stitched fluxes
    ifile = None
    plot_fluxes(ifile,run,stitch_my_time,stitch_flux_dict)

def plot_fluxes(ifile,run,mytime,mydict):

    ## SELECT PLOTTING CASE
    ollie_case = False

    if ollie_case:
        avg_in_title = False
    else:
        avg_in_title = True
        ylims = None
        label_ypos = None

    islin = mydict['islin']
    has_flowshear = mydict['has_flowshear']
    
    # t grid
    time = mytime.time
    time_steady = mytime.time_steady
    it_min = mytime.it_min
    it_max = mytime.it_max

    # k grids
    nx = mydict['nx']
    nxmid = nx//2 + 1
    ny = mydict['ny']
    naky = mydict['naky']
    kx = mydict['kx']
    ky = mydict['ky']
    jtwist = mydict['jtwist']
    theta0 = mydict['theta0']
    
     # theta grids
    theta = mydict['theta']

    # species
    nspec = mydict['nspec']
    spec_names = mydict['spec_names']

    # fluxes vs t
    pflx = mydict['pflx']
    qflx = mydict['qflx']
    vflx = mydict['vflx']
    pioq = mydict['pioq']

    # fluxes vs (kx,ky)
    pflx_kxky_tavg = mydict['pflx_kxky_tavg']
    qflx_kxky_tavg = mydict['qflx_kxky_tavg']
    vflx_kxky_tavg = mydict['vflx_kxky_tavg']

    # potential
    phi_bytheta_tfinal = mydict['phi_bytheta_tfinal']
    phi2_avg = mydict['phi2_avg']
    phi2_by_ky = mydict['phi2_by_ky']
    phi2_kxky = mydict['phi2_kxky']
    phi2_kxky_tavg = mydict['phi2_kxky_tavg']
    
    print()
    print(">>> producing plots of fluxes vs time...")

    print("-- plotting avg(phi2)")
    write_fluxes_vs_t = False
    tmp_pdf_id = 1
    pdflist = []
    if phi2_avg is not None:
        title = '$\\langle|\phi^{2}|\\rangle_{\\theta,k_x,k_y}$'
        if islin:
            title = '$\ln$'+title
            gplot.plot_1d(time,np.log(phi2_avg),'$t (a/v_{t})$',title)
        else:
            gplot.plot_1d(time,phi2_avg,'$t (a/v_{t})$',title, semilogy=True)
        # indicating area of saturation
        plt.xlim((time[0], time[-1]))
        plt.axvline(x=time_steady[0], color='grey', linestyle='-')
        ax = plt.gca()
        ax.axvspan(time_steady[0], time_steady[-1], alpha=0.1, color='grey')
        plt.grid(True)
        write_fluxes_vs_t = True
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
    print("-- plotting particle flux")
    if pflx is not None:
        title = '$\Gamma/\Gamma_{gB}$'
        if ollie_case:
            ylims = [-0.75, 0.75]
            #label_ypos = [-0.35,0.5,0.2] # for old algo
            label_ypos = [-0.2,0.35,0.17] # for new algo

        # Linear plot
        plot_flux_vs_t(islin,nspec,spec_names,mytime,pflx,title,ylims,label_ypos,avg_in_title)
        write_fluxes_vs_t = True
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

        # Semilogy plot
        title = '$\\vert$' + title + '$\\vert$'
        plot_flux_vs_t(islin,nspec,spec_names,mytime,np.abs(pflx),title,ylims,label_ypos,avg_in_title,semilogy=True)
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

    print("-- plotting heat flux")
    if qflx is not None:
        title = '$Q/Q_{gB}$'
        if ollie_case:
            ylims = [-0.1, 4.0]
            #label_ypos = [3.1,1.2,2.1] # for old algo
            label_ypos = [2.2,0.85,1.4] # for new algo
        
        # Linear plot
        plot_flux_vs_t(islin,nspec,spec_names,mytime,qflx,title,ylims,label_ypos,avg_in_title)
        write_fluxes_vs_t = True
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
        
        # Semilogy plot
        title = '$\\vert$' + title + '$\\vert$'
        plot_flux_vs_t(islin,nspec,spec_names,mytime,np.abs(qflx),title,ylims,label_ypos,avg_in_title,semilogy=True)
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

    print("-- plotting momentum flux")
    if vflx is not None:
        title = '$\Pi/\Pi_{gB}$'
        if ollie_case:
            ylims = [-0.2, 6.0]
            #label_ypos = [1.8,0.3,4.8] # for old algo
            label_ypos = [1.25,0.25,3.0] # for old algo
        else:
            ylims = None

        # Linear plot
        plot_flux_vs_t(islin,nspec,spec_names,mytime,vflx,title,ylims,label_ypos,avg_in_title)
        write_fluxes_vs_t = True
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
        
        # Semilogy plot
        title = '$\\vert$' + title + '$\\vert$'
        plot_flux_vs_t(islin,nspec,spec_names,mytime,np.abs(vflx),title,ylims,label_ypos,avg_in_title,semilogy=True)
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

    #if myout['es_energy_exchange_present']:
    #    title = 'energy exchange'
    #    gplot.plot_1d(mytime.time,self.xchange,"$t (v_t/a)$",title)
    #    write_fluxes_vs_t = True
    #    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    #    gplot.save_plot(tmp_pdfname, run, ifile)
    #    pdflist.append(tmp_pdfname)
    #    tmp_pdf_id = tmp_pdf_id+1
    print("-- plotting momentum/heat flux ratio")
    if pioq is not None:
        title = '$\Pi/Q$'
        #for idx in range(nspec):
        #    plt.plot(mytime.time_steady,pioq[it_min:it_max,idx],label=spec_names[idx])
        #plt.title(title)
        #plt.xlabel('$t\ [r_r/v_{thr}]$')
        #plt.legend()
        #plt.grid(True)
        plot_flux_vs_t(islin,nspec,spec_names,mytime,pioq,title,ylims,label_ypos,avg_in_title,only_steady=True)
        write_fluxes_vs_t = True
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1


    print("-- plotting field by ky")

    if phi2_kxky is not None:

        # Plot zonal velocity squared

        title = '$\sum_{k_x} k_x^2\\langle|\\hat{\\varphi}_{k_x,0}|^2\\rangle_{\\theta}$'
        zvelsq = np.zeros(mytime.ntime)
        for ikx in range(nx):
            zvelsq += kx[ikx]**2 * phi2_kxky[:,0,ikx]
        if islin:
            title = '$\\ln$' + title
            zvelsq = np.log(zvelsq)
        plt.semilogy(time, zvelsq, color=gplot.mybluestd)
        plt.xlabel('$t (a/v_t)$')
        plt.title(title)
        # indicating area of saturation
        plt.xlim((time[0], time[-1]))
        plt.axvline(x=time_steady[0], color='grey', linestyle='-')
        ax = plt.gca()
        ax.axvspan(time_steady[0], time_steady[-1], alpha=0.1, color='grey')
        plt.grid(True)
        write_fluxes_vs_t = True
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1


        # Plot zonal shear squared

        title = '$\sum_{k_x} k_x^4\\langle|\\hat{\\varphi}_{k_x,0}|^2\\rangle_{\\theta}$'
        zshearsq = np.zeros(mytime.ntime)
        for ikx in range(nx):
            zshearsq += kx[ikx]**4 * phi2_kxky[:,0,ikx]
        if islin:
            title = '$\\ln$' + title
            zshearsq = np.log(zshearsq)
        plt.semilogy(time, zshearsq, color=gplot.mybluestd)
        plt.xlabel('$t (a/v_t)$')
        plt.title(title)
        # indicating area of saturation
        plt.xlim((time[0], time[-1]))
        plt.axvline(x=time_steady[0], color='grey', linestyle='-')
        ax = plt.gca()
        ax.axvspan(time_steady[0], time_steady[-1], alpha=0.1, color='grey')
        plt.grid(True)
        write_fluxes_vs_t = True
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1


    if phi2_by_ky is not None:

        title = '$\\langle|\phi^{2}|\\rangle_{\\theta,k_x}$'
        # Create list of colors
        cmap = plt.get_cmap('nipy_spectral')
        my_colors = [cmap(i) for i in np.linspace(0,1,naky-1)]
        if islin:
            title = '$\\ln$' + title
            plt.semilogy(time, np.log(phi2_by_ky[:,0]),label='ky = '+'{:5.3f}'.format(ky[0]),linestyle='dashed',color='black')
            for iky in range(1,naky) :
                plt.semilogy(time, np.log(phi2_by_ky[:,iky]),label='ky = '+'{:5.3f}'.format(ky[iky]),color=my_colors[iky-1])
        else:
            plt.plot(time, phi2_by_ky[:,0],label='ky = '+'{:5.3f}'.format(ky[0]),linestyle='dashed',color='black')
            for iky in range(1,naky) :
                plt.semilogy(time, phi2_by_ky[:,iky],label='ky = '+'{:5.3f}'.format(ky[iky]),color=my_colors[iky-1])
        plt.xlabel('$t (a/v_t)$')
        plt.title(title)
        leg = plt.legend(prop={'size': 10}, ncol=3,frameon=True,fancybox=False,framealpha=1.0)
        leg.get_frame().set_facecolor('w')
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(1.0)
        # indicating area of saturation
        plt.xlim((time[0], time[-1]))
        plt.axvline(x=time_steady[0], color='grey', linestyle='-')
        ax = plt.gca()
        ax.axvspan(time_steady[0], time_steady[-1], alpha=0.1, color='grey')

        plt.grid(True)
        write_fluxes_vs_t = True
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1


        # Break down the previous plot into smaller sets of (kx,ky):
        # ibin*(kmax/nbin) < |k| < (ibin+1)*(kmax/nbin), with 0 <= ibin <= nbin

        # Set number of bins for kx
        nbinx = 3

        # Set number of bins for ky
        nbiny = 3

        if nx>nbinx and ny>nbiny:

            # Set up colors, linestyle and label
            cmap = plt.get_cmap('nipy_spectral')
            clrs = [cmap(i) for i in np.linspace(0,1,naky-1)]
            clrs.insert(0,'black')
            lstyle = ['dashed']
            for iky in range(naky-1):
                lstyle.append('solid')
            lab = []
            for iky in range(naky):
                lab.append('ky = '+'{:5.3f}'.format(ky[iky]))

            for ibiny in range(nbiny):

                # ky range for this bin
                kyminbin = ibiny*ky[-1]/nbiny
                kymaxbin = (ibiny+1)*ky[-1]/nbiny

                # Corresponding indices, with zonal for the first bin in y
                ikybin = np.where((ky>kyminbin) & (ky<=kymaxbin))[0]
                if ibiny == 0:
                    ikybin = np.insert(ikybin,0,0)

                for ibinx in range(nbinx):

                    # kx range for this bin
                    kxminbin = ibinx*kx[-1]/nbinx
                    kxmaxbin = (ibinx+1)*kx[-1]/nbinx

                    # Corresponding indices, with streamers for the first bin in x
                    ikxbin = np.where((np.abs(kx)>kxminbin) & (np.abs(kx)<=kxmaxbin))[0]
                    if ibinx == 0:
                        ikxbin = np.insert(ikxbin,ikxbin.size//2,0)

                    title = "$|k_x| \\in $" + " [{0:.2f},{1:.2f}]".format(kxminbin,kxmaxbin)

                    if islin:
                        title = '$\\ln$' + title

                    # Plot sum_kx(phi2) for every ky in the bin,
                    # summed over all kx's in the bin

                    plt.figure(figsize=(12,8))

                    for iky in ikybin:

                        # Average over all kx's in this bin
                        toplot = np.sum(phi2_kxky[:,iky,ikxbin],1)/ikxbin.size

                        # In linear runs, plot the log
                        if islin:
                            toplot = np.log(toplot)

                        plt.semilogy(time, toplot, label=lab[iky], linestyle=lstyle[iky], color=clrs[iky])

                    # Fine-tune plot
                    plt.xlabel('$t\ [r_r/v_{thr}]$')
                    plt.ylabel("$\\langle|\\hat{\\varphi}|^{2}\\rangle_{\\theta,k_x}$")
                    plt.title(title, fontsize=24)
                    leg = plt.legend(prop={'size': 16}, ncol=2,frameon=True,fancybox=False,framealpha=1.0)
                    leg.get_frame().set_facecolor('w')
                    leg.get_frame().set_edgecolor('k')
                    leg.get_frame().set_linewidth(1.0)
                    plt.grid(True)

                    # Save plot to collate later
                    write_fluxes_vs_t = True
                    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
                    gplot.save_plot(tmp_pdfname, run, ifile)
                    pdflist.append(tmp_pdfname)
                    tmp_pdf_id = tmp_pdf_id+1


    if write_fluxes_vs_t:
        merged_pdfname = 'fluxes_vs_t'
        if ifile==None: # This is the case when we stitch fluxes together
            merged_pdfname += '_'+run.scan_name
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

    print('complete')
    
    print()
    print('producing plots of fluxes vs (kx,ky)...', end='')

    write_fluxes_vs_kxky = False
    tmp_pdf_id = 1
    pdflist = []

    ## Plot energy spectrum kx*<phi2>_{t,theta,ky} vs kx
    energy_dens_x = np.squeeze(np.abs(kx)*np.sum(phi2_kxky_tavg[1:,:],axis=0))
    plt.semilogy(kx, energy_dens_x, color=gplot.myblue, linewidth=3.0, marker='o')
    [xmin,xmax] = plt.gca().get_xlim()
    xvecpos = np.linspace(xmax/4,xmax)
    xvecneg = np.linspace(xmin,xmin/4)
    [ymin,ymax] = plt.gca().get_ylim()
    yvec = np.linspace(ymin,ymax)
    fitpos = (xvecpos/kx[-1])**(-7.0/3.0) * energy_dens_x[-1]
    fitneg = np.abs((xvecneg/kx[0]))**(-7.0/3.0) * energy_dens_x[0]
    plt.semilogy(xvecpos, fitpos, color='k', linewidth=1.5, label='$(\\bar{k}_x\\rho_i)^{-7/3}$')
    plt.semilogy(xvecneg, fitneg, color='k', linewidth=1.5, label=None)
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))
    plt.grid(True)
    plt.xlabel('$\\bar{k}_{x}\\rho_i$')
    plt.ylabel('$\\sum_{k_y\\neq 0} \\vert\\bar{k}_x\\vert\\rho_i\\langle \\vert\\varphi\\vert ^2\\rangle_{t,\\theta}$')
    legend = plt.legend(frameon = True, fancybox = False)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    frame.set_linewidth(0.5)
    frame.set_alpha(1)
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run, ifile)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1

    ## Plot energy spectrum ky*<phi2>_{t,theta,kx} vs ky
    energy_dens_y = np.squeeze(ky*np.sum(phi2_kxky_tavg,axis=1))
    iky_energymax = np.argmax(energy_dens_y)
    plt.semilogy(ky, energy_dens_y, color=gplot.myblue, linewidth=3.0, marker='o')
    [xmin,xmax] = plt.gca().get_xlim()
    xvec = np.linspace(xmax/4,xmax)
    [ymin,ymax] = plt.gca().get_ylim()
    yvec = np.linspace(ymin,ymax)
    fit = (xvec/ky[-1])**(-7.0/3.0) * energy_dens_y[-1]
    plt.semilogy(xvec, fit, color='k', linewidth=1.5, label='$(k_y\\rho_i)^{-7/3}$')
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))
    plt.grid(True)
    plt.xlabel('$k_{y}\\rho_i$')
    plt.ylabel('$\\sum_{k_x} k_y\\rho_i\\langle \\vert\\varphi\\vert ^2\\rangle_{t,\\theta}$')
    legend = plt.legend(frameon = True, fancybox = False)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    frame.set_linewidth(0.5)
    frame.set_alpha(1)
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run, ifile)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1

    ## Plot phi2 averaged over t and theta, vs (kx,ky)
    dkx = abs(kx[1]-kx[0])
    for iky in range(naky):
        plt.semilogy(kx,1/dkx*phi2_kxky_tavg[iky,:], marker='o', color=gplot.myblue, \
                markersize=8, markerfacecolor=gplot.myblue, markeredgecolor=gplot.myblue, linewidth=2.0)
        plt.grid(True)
        plt.xlabel('$\\bar{k}_{x}\\rho_i$')
        plt.ylabel('$\\frac{1}{\\Delta k_x}\\langle\\vert\\varphi\\vert ^2\\rangle_{t,\\theta}$')
        plt.title('$k_y\\rho_i = $'+str(round(ky[iky],2)))
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id += 1

    # Then non-zonal modes on contour plot vs (kx,ky)
    plot_phi2_vs_kxky(kx,ky,phi2_kxky_tavg,has_flowshear)
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run, ifile)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1

    if pflx_kxky_tavg is not None:
        title = '$\Gamma_{GS2}$'
        for ispec in range(nspec):
            plot_flux_vs_kxky(ispec,spec_names,kx,ky,pflx_kxky_tavg,title,has_flowshear)
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
        write_fluxes_vs_kxky = True
    if qflx_kxky_tavg is not None:
        title = '$Q_{GS2}$'
        for ispec in range(nspec):
            plot_flux_vs_kxky(ispec,spec_names,kx,ky,qflx_kxky_tavg,title,has_flowshear)
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
        write_fluxes_vs_kxky = True
    if vflx_kxky_tavg is not None:
        title = '$\Pi_{GS2}$'
        for ispec in range(nspec):
            plot_flux_vs_kxky(ispec,spec_names,kx,ky,vflx_kxky_tavg,title,has_flowshear)
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
        write_fluxes_vs_kxky = True

    if write_fluxes_vs_kxky:
        merged_pdfname = 'fluxes_vs_kxky'
        if ifile==None: # This is the case when we stitch fluxes together
            merged_pdfname += '_'+run.scan_name
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

    print('complete')
    
    print()
    print('producing plots of potential vs (theta-theta0)...', end='')

    # Plot for the smallest non-zero ky,
    # for the ky with the maximum energy,
    # and for the largest ky.
    iky_to_plot = [1,iky_energymax,ny-1]
    phi2_bytheta_tfinal = np.abs(phi_bytheta_tfinal)**2

    for iky in iky_to_plot:

        tmp_pdf_id = 1
        pdflist = []

        for dmid in range(min(jtwist*iky, int(nx//2))):

            # Get chain of (theta-theta0) and associated phi2.
            bloonang, phi2bloon = get_bloon(theta,theta0,phi2_bytheta_tfinal,iky,dmid,jtwist)

            plt.semilogy(bloonang,phi2bloon,color=gplot.myblue,linewidth=3.0)
            plt.grid(True)
            plt.xlabel('$\\theta-\\theta_0$')
            plt.ylabel('$\\vert\\varphi\\vert^2$')
            plt.title('$k_y = '+str(round(ky[iky],2))+'$, $d_{mid} = '+str(dmid)+'$, at $t='+str(round(time[-1],3))+'$')

            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id += 1

        merged_pdfname = 'potential_vs_theta_theta0_iky_'+str(iky)
        if ifile==None: # This is the case when we stitch fluxes together
            merged_pdfname += '_'+run.scan_name
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

    # With linear scale
    iky_to_plot = [1,iky_energymax,ny-1]
    phi2_bytheta_tfinal = np.abs(phi_bytheta_tfinal)**2

    for iky in iky_to_plot:

        tmp_pdf_id = 1
        pdflist = []

        for dmid in range(min(jtwist*iky, int(nx//2))):

            # Get chain of (theta-theta0) and associated phi2.
            bloonang, phi2bloon = get_bloon(theta,theta0,phi2_bytheta_tfinal,iky,dmid,jtwist)

            plt.plot(bloonang,phi2bloon,color=gplot.myblue,linewidth=3.0)
            plt.grid(True)
            plt.xlabel('$\\theta-\\theta_0$')
            plt.ylabel('$\\vert\\varphi\\vert^2$')
            plt.title('$k_y = '+str(round(ky[iky],2))+'$, $d_{mid} = '+str(dmid)+'$, at $t='+str(round(time[-1],3))+'$')

            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id += 1

        merged_pdfname = 'potential_vs_theta_theta0_linear_iky_'+str(iky)
        if ifile==None: # This is the case when we stitch fluxes together
            merged_pdfname += '_'+run.scan_name
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

    print('complete')


def plot_flux_vs_t(islin,nspec,spec_names,mytime,flx,ylabel,ylims=None,my_label_ypos=None,avg_in_title=None,only_steady=False,semilogy=False):

    fig=plt.figure(figsize=(12,8))
    if islin:
        ylabel = '$\\ln($' + ylabel + '$)$'

    #my_colorlist = plt.cm.YlGnBu(np.linspace(0.5,1,nspec)) # for old algo
    #my_colorlist = plt.cm.YlOrBr(np.linspace(0.5,1,nspec)) # for new algo
    my_colorlist =  [gplot.mybluestd, gplot.myredstd, gplot.myyellow, gplot.mygreen] # for standard cases

    my_curves = []
    my_labels = ['$^2H$','$e^-$','$^{12}C$']
    #my_linestyle_list = ['-', '--', ':']
    my_linestyle_list = ['-', '-', '-']

    # Adapt to selected time window
    if not only_steady:
        tselect = mytime.time
        flxselect = flx
        # indicating area of saturation
        plt.axvline(x=mytime.time_steady[0], color='grey', linestyle='-')
        plt.axvline(x=mytime.time_steady[-1], color='grey', linestyle='-')
        ax = plt.gca()
        ax.axvspan(mytime.time_steady[0], mytime.time_steady[-1], alpha=0.1, color='grey')
    else:
        tselect = mytime.time_steady
        flxselect = flx[mytime.it_min:mytime.it_max,:]
    
    # plot time-traces for each species
    for idx in range(nspec):
        # get the time-averaged flux
        if islin:
            crv, = plt.plot(tselect,np.log(flxselect[:,idx]),color=my_colorlist[idx],linewidth=3.0, \
                    linestyle=my_linestyle_list[idx])
        else:
            if semilogy:
                crv, = plt.semilogy(tselect,flxselect[:,idx],color=my_colorlist[idx],linewidth=3.0, \
                        linestyle=my_linestyle_list[idx])
            else:
                crv, = plt.plot(tselect,flxselect[:,idx],color=my_colorlist[idx],linewidth=3.0, \
                        linestyle=my_linestyle_list[idx])
        my_curves.append(crv)
    
    # plot time-averages
    ttl = 'Time avg: '
    for idx in range(nspec):
        
        if not islin:
            
            flxavg = mytime.timeavg(flx[:,idx])
            
            # Annotate
            if my_label_ypos:
                note_str = 'avg = {:.2f}'.format(flxavg)

                xpos = mytime.time[-1]*0.82
                ypos = my_label_ypos[idx]
                #if ylims:
                #    #ypos = flx[round(len(mytime.time)*0.8),idx]+(ylims[1]-ylims[0])/15.
                #    ypos = flxavg+(ylims[1]-ylims[0])/15.
                #    if ypos > ylims[1]:
                #        #ypos = flx[round(len(mytime.time)*0.8),idx]-(ylims[1]-ylims[0])/12.
                #        ypos = flxavg-(ylims[1]-ylims[0])/12.
                #else:
                #    #ypos = flx[round(len(mytime.time)*0.8),idx]+(np.amax(flx)-np.amin(flx))/15.
                #    ypos = flxavg+(np.amax(flx)-np.amin(flx))/15.
                #    if ypos > np.amax(flx):
                #        #ypos = flx[round(len(mytime.time)*0.8),idx]-(np.amax(flx)-np.amin(flx))/12.
                #        ypos = flxavg-(np.amax(flx)-np.amin(flx))/12.

                note_xy = (xpos, ypos)
                note_coords = 'data'

                plt.annotate(note_str, xy=note_xy, xycoords=note_coords, color=my_colorlist[idx], \
                        fontsize=26, backgroundcolor='w', \
                        bbox=dict(facecolor='w', edgecolor=my_colorlist[idx], alpha=1.0))

            elif avg_in_title:

                my_labels[idx] += '  (avg: {:.1E})'.format(flxavg)

            print('flux avg for '+spec_names[idx]+': '+str(flxavg))

    plt.xlabel('$t [L/v_{th,i}]$')
    plt.ylabel(ylabel)
    plt.xlim([tselect[0],tselect[-1]])
    if ylims is not None:
        plt.ylim(ylims)
    plt.grid(True)
    #plt.title(ttl)

    if only_steady:
        legloc = 'best'
    elif semilogy:
        legloc = 'lower right'
    else:
        legloc = 'upper left'
    my_legend = plt.legend(my_curves,my_labels,frameon=True,fancybox=False,framealpha=1.0,loc=legloc)
    my_legend.get_frame().set_facecolor('w')
    my_legend.get_frame().set_edgecolor('k')
    my_legend.get_frame().set_linewidth(1.0)

    return fig

def plot_flux_vs_kxky(ispec,spec_names,kx,ky,flx,title,has_flowshear):

    from gs2_plotting import plot_2d

    if has_flowshear:
        xlab = '$\\bar{k}_{x}\\rho_i$'
    else:
        xlab = '$k_{x}\\rho_i$'
    ylab = '$k_{y}\\rho_i$'

    cmap = 'RdBu_r' # 'Reds','Blues'
    z = np.abs(flx[ispec,1:,:]) # absolute val and exclude zonal mode which is zero
    z_min, z_max = z.min(), z.max()
    
    title = 'Contributions to ' + title
    if ispec > 1:
        title += ' (impurity ' + str(ispec-1) + ')'
    else:
        title += ' (' + spec_names[ispec] + 's)'

    use_logcolor = True
    fig = plot_2d(z,kx,ky[1:],z_min,z_max,xlab,ylab,title,cmap,use_logcolor)

    return fig

def plot_phi2_vs_kxky(kx,ky,phi2,has_flowshear):

    from gs2_plotting import plot_2d

    title = '$\\langle\\vert\\varphi\\vert ^2\\rangle_{t,\\theta}$'
    title += ' $\\forall$ $k_y\\neq 0$'
    
    if has_flowshear:
        xlab = '$\\bar{k}_{x}\\rho_i$'
    else:
        xlab = '$k_{x}\\rho_i$'
    ylab = '$k_{y}\\rho_i$'

    cmap = 'RdBu_r'# 'RdBu_r','Blues'
    z = phi2[1:,:] # taking out zonal modes because they are much larger
    z_min, z_max = z.min(), z.max()

    use_logcolor = True
    fig = plot_2d(z,kx,ky[1:],z_min,z_max,xlab,ylab,title,cmap,use_logcolor)

    return fig

def plot_flux_vs_vpth(mygrids,flx,title):

    from gs2_plotting import plot_2d

    xlab = '$\\theta$'
    ylab = '$v_{\parallel}$'
    cmap = 'RdBu'
    for idx in range(flx.shape[0]):
        z = flx[idx,:,:]
        z_min, z_max = z.min(), z.max()
        fig = plot_2d(z,mygrids.theta,mygrids.vpa,z_min,z_max,xlab,ylab,title+' (is= '+str(idx+1)+')',cmap)

    return fig

def get_bloon(theta,theta0,phi2_bytheta_tfinal,iky,dmid,jtwist):

    ntheta = theta.size
    nx = phi2_bytheta_tfinal.shape[1]
    ikx0 = (nx-1)//2
    sgn_shat = np.sign(theta0[1,1]-theta0[1,0])

    # List of ikx's belonging to this chain,
    # in decreasing order which corresponds to
    # increasing(theta-thet0) for shat>0
    ikx_bloon = []
    ikx = ikx0 + dmid
    while ikx + jtwist*iky < nx:
        ikx += jtwist*iky
    while ikx >= 0:
        ikx_bloon.append(ikx)
        ikx -= jtwist*iky

    # If shat<0, need to invert the kx-indices list
    if sgn_shat == -1:
        ikx_bloon = list(reversed(ikx_bloon))

    bloonang = []
    phi2bloon = []
    
    for ilink in ikx_bloon:
        for itheta in range(ntheta):
            bloonang.append(theta[itheta]-theta0[iky,ilink])
            phi2bloon.append(phi2_bytheta_tfinal[iky,ilink,itheta])

    return bloonang, phi2bloon

def plot_allmodes_vs_t(run, time, flux, fldname, zttl, pltname=None, ifile=None, tauwindow=False, zmin=0.1, zmax=10, cmp='RdBu_c_one'):

    if pltname is None:
        pltname = fldname

    multi = ifile is None

    if multi:
        nrun = len(time)
    else:
        nrun = 1

    fig = plt.figure()
    tmp_pdf_id = 1
    pdflist = []

    for irun in range(nrun):

        if multi:
            flx = flux[irun]
            tm = time[irun]
            ttl = run.flabels[irun]
        else:
            flx = flux
            tm = time
            ttl = ''

        nx = flx['nx']
        kx = flx['kx']
        ny = flx['ny']
        ky = flx['ky']
        
        if tauwindow:
            nt = tm.ntauwin
            t = tm.t_tauwinavg
        else:
            nt = tm.ntime
            t = tm.time

        gap = 10

        nrows = (nx+gap)*ny
        fld = np.zeros((nrows,nt))
        yax = np.zeros(nrows)
        ylab = ['$(0.0,'+str(round(ky[iy],2))+')$' for iy in range(ny)]
        yaxtick = [(nx+gap)*iy+(nx+1)//2 for iy in range(ny)]
        for iy in range(ny):
            for ix in range(nx):
                imode = iy*(nx+gap) + ix
                yax[imode] = imode
                fld[imode,:] = flx[fldname][:,iy,ix]
            for iskip in range(gap):
                imode = iy*(nx+gap) + nx + iskip
                yax[imode] = imode
                fld[imode,:] = np.nan*np.ones(nt)

        if cmp == 'RdBu_c_one':
            if (zmax-1) > 5*(1-zmin):
                zticks = [zmin,1.0,(zmax-1)/3+1,2*(zmax-1)/3+1,zmax]
            else:
                zticks = [zmin,(1.0-zmin)/2+zmin,1.0,(zmax-1)/2+1,zmax]
        else:
            zticks = [(zmax-zmin)/5*i+zmin for i in range(6)]

        # Adapt time array for plot_2d_uneven_xgrid:
        t_by_ky = np.zeros((nrows, nt))
        for iy in range(nrows):
            t_by_ky[iy,:] = t
        gplot.plot_2d_uneven_xgrid(
                      t_by_ky, yax, fld,
                      t[0], t[-1],
                      zmin, zmax,
                      '$t$ [$a/v_{th}$]', '$(\\bar{k}_x,k_y)$', ttl,
                      x_is_twopi = False, ngrid_fine = 2*nt+1, clrmap = cmp,
                      zticks = zticks, zlabel = zttl,
                      yticks = yaxtick, yticklabels = ylab, ytickfontsize = 15)

        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        if multi:
            gplot.save_plot(tmp_pdfname, run)
        else:
            gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

    if multi:
        merged_pdfname = pltname + '_vs_t_' + run.scan_name
        gplot.merge_pdfs(pdflist, merged_pdfname, run)
    else:
        merged_pdfname = pltname + '_vs_t_' + run.fnames[ifile]
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

    return fig



def plot_phi2_vs_bloon(run, time, flux, kyplt, tt0plt, ifile=None):

    multi = ifile is None

    if multi:
        nrun = len(time)
    else:
        nrun = 1

    fig = plt.figure()
    tmp_pdf_id = 1
    pdflist = []

    for ky_wish in kyplt:

        for tt0_wish in tt0plt:

            create_phi2_vs_bloon(multi, nrun, run, time, flux, plt.plot, ky_wish, tt0_wish)

            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            if multi:
                gplot.save_plot(tmp_pdfname, run)
            else:
                gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1

            create_phi2_vs_bloon(multi, nrun, run, time, flux, plt.semilogy, ky_wish, tt0_wish)

            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            if multi:
                gplot.save_plot(tmp_pdfname, run)
            else:
                gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1

    if multi:
        merged_pdfname = 'phi2_vs_bloon_' + run.scan_name
        gplot.merge_pdfs(pdflist, merged_pdfname, run)
    else:
        merged_pdfname =  'phi2_vs_bloon_' + run.fnames[ifile]
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

    return fig

def create_phi2_vs_bloon(multi, nrun, run, time, flux, pltfunc, ky_wish, tt0_wish):

    labs = []

    for irun in range(nrun):

        if multi:
            flx = flux[irun]
            tm = time[irun]
            lab = run.flabels[irun]
        else:
            flx = flux
            tm = time
            lab = None

        jtwist = flx['jtwist']
        theta = flx['theta']
        theta0 = flx['theta0']
        phi2 = np.abs(flx['phi_bytheta_tfinal'])**2
        ky = flx['ky']
        
        ikynear, kynear = find_nearest(ky, ky_wish)

        # min|kx| / dkx in the chain
        nx = theta0.shape[1]
        dmid = int(round(jtwist*ikynear*tt0_wish/(2*pi)))
        dmid = min(int(nx//2), dmid)
        tt0near = 2*pi*dmid/(jtwist*ikynear)

        bloonang, phi2bloon = get_bloon(theta, theta0, phi2, ikynear, dmid, jtwist)
        phi2bloon = phi2bloon / np.amax(phi2bloon)

        pltfunc(bloonang, phi2bloon)

        if multi:
            lab = run.flabels[irun]
        else:
            lab = None
        labs.append(lab)

    plt.title('$k_y \\simeq ' + str(round(ky_wish,2)) + \
            ', \\min\\vert \\theta_0\\vert \\simeq ' + str(round(abs(tt0_wish),2)) + '$')
    plt.grid(True)
    plt.xlabel('$\\theta-\\theta_0$')
    plt.ylabel('$\\vert\\varphi\\vert^2\ /\ \\max\\vert\\varphi\\vert^2$')
    leg = plt.legend(labs,
                     prop={'size': 8},
                     ncol=1,
                     frameon=True,
                     fancybox=False,
                     framealpha=1.0,
                     handlelength=1)


def find_nearest(array, value):

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return idx, array[idx]



def plot_spectra(run, flux, ifile=None):

    leg_fontsize = 8

    multi = ifile is None

    if multi:
        nrun = len(flux)
    else:
        nrun = 1

    def create_energy_spectra(vs):

        xmin = np.nan
        xmax = np.nan
        ymin = np.nan
        ymax = np.nan

        for irun in range(nrun):

            if multi:
                flx = flux[irun]
                lab = run.flabels[irun]
                fitlab = None
            else:
                flx = flux
                lab = None
                if vs == 'kx':
                    fitlab = '$(\\bar{k}_x\\rho_i)^{-7/3}$'
                else:
                    fitlab = '$(k_y\\rho_i)^{-7/3}$'

            if vs == 'kx':
                
                k = flx['kx']
                dk = abs(k[1]-k[0])
                energy_dens = 1.0/dk * np.squeeze(np.abs(k)*np.sum(flx['phi2_kxky_tavg'][1:,:],axis=0))
                fit_both = True
                xlab = '$\\bar{k}_{x}\\rho_i$'
                ylab = '$\\frac{1}{\\Delta k_x}\\sum_{k_y\\neq 0} \\vert\\bar{k}_x\\vert\\rho_i\\langle \\vert\\varphi\\vert ^2\\rangle_{t,\\theta}$'

            elif vs == 'ky':
                
                k = flx['ky']
                dk = abs(k[1]-k[0])
                energy_dens = 1.0/dk * np.squeeze(np.abs(k)*np.sum(flx['phi2_kxky_tavg'],axis=1))
                fit_both = False
                xlab = '$k_{y}\\rho_i$'
                ylab = '$\\frac{1}{\\Delta k_y}\\sum_{k_x} k_y\\rho_i\\langle \\vert\\varphi\\vert ^2\\rangle_{t,\\theta}$'

            plt.semilogy(k, energy_dens, marker='o', label=lab, markersize=3)

            [this_xmin, this_xmax] = plt.gca().get_xlim()
            if not (xmin < this_xmin): xmin = this_xmin
            if not (xmax > this_xmax): xmax = this_xmax

            [this_ymin, this_ymax] = plt.gca().get_ylim()
            if not (ymin < this_ymin): ymin = this_ymin
            if not (ymax > this_ymax): ymax = this_ymax

            if irun == nrun-1:

                xvec = np.linspace(xmax/10,3*xmax)
                fit = (xvec/k[-1])**(-7.0/3.0) * energy_dens[-1]
                plt.semilogy(xvec, fit, color='k', linewidth=1.5, label=fitlab, linestyle='--')

                if fit_both:

                    xvec = np.linspace(3*xmin,xmin/10)
                    fit = np.abs((xvec/k[0]))**(-7.0/3.0) * energy_dens[0]
                    plt.semilogy(xvec, fit, color='k', linewidth=1.5, label=fitlab, linestyle='--')

        plt.xlim((xmin,xmax))
        plt.ylim((ymin,ymax))
        plt.grid(True)
        plt.xlabel(xlab)
        plt.ylabel(ylab, fontsize=15)

        plt.legend(prop={'size': leg_fontsize},
                   ncol=1,
                   frameon=True,
                   fancybox=False,
                   framealpha=1.0,
                   handlelength=1)

    def create_zonal_spectrum(to_plot):

        for irun in range(nrun):

            if multi:
                flx = flux[irun]
                lab = run.flabels[irun]
            else:
                flx = flux
                lab = None

            if to_plot == 'field':
                fac = 1
            elif to_plot == 'flow':
                fac = flx['kx']**2
            elif to_plot == 'shear':
                fac = flx['kx']**4

            dkx = abs(flx['kx'][1]-flx['kx'][0])
            plt.semilogy(flx['kx'],fac/dkx*flx['phi2_kxky_tavg'][0,:], marker='o', label=lab, markersize=3)

        plt.grid(True)
        plt.xlabel('$\\bar{k}_{x}\\rho_i$')
        if to_plot == 'field':
            ylab = '$\\frac{1}{\\Delta k_x}\\langle \\vert\\varphi_Z\\vert ^2\\rangle_{t,\\theta}$'
        elif to_plot == 'flow':
            ylab = '$\\frac{1}{\\Delta k_x}k_x^2\\langle \\vert\\varphi_Z\\vert ^2\\rangle_{t,\\theta}$'
        elif to_plot == 'shear':
            ylab = '$\\frac{1}{\\Delta k_x}k_x^4\\langle \\vert\\varphi_Z\\vert ^2\\rangle_{t,\\theta}$'
        plt.ylabel(ylab)
        plt.legend(prop={'size': leg_fontsize},
                   ncol=1,
                   frameon=True,
                   fancybox=False,
                   framealpha=1.0,
                   handlelength=1)

    fig = plt.figure()
    tmp_pdf_id = 1
    pdflist = []

    create_energy_spectra('kx')

    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    if multi:
        gplot.save_plot(tmp_pdfname, run)
    else:
        gplot.save_plot(tmp_pdfname, run, ifile)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id = tmp_pdf_id+1

    create_energy_spectra('ky')

    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    if multi:
        gplot.save_plot(tmp_pdfname, run)
    else:
        gplot.save_plot(tmp_pdfname, run, ifile)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id = tmp_pdf_id+1

    create_zonal_spectrum('field')

    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    if multi:
        gplot.save_plot(tmp_pdfname, run)
    else:
        gplot.save_plot(tmp_pdfname, run, ifile)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id = tmp_pdf_id+1

    create_zonal_spectrum('flow')

    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    if multi:
        gplot.save_plot(tmp_pdfname, run)
    else:
        gplot.save_plot(tmp_pdfname, run, ifile)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id = tmp_pdf_id+1

    create_zonal_spectrum('shear')

    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    if multi:
        gplot.save_plot(tmp_pdfname, run)
    else:
        gplot.save_plot(tmp_pdfname, run, ifile)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id = tmp_pdf_id+1

    if multi:
        merged_pdfname = 'spectra_' + run.scan_name
        gplot.merge_pdfs(pdflist, merged_pdfname, run)
    else:
        merged_pdfname = 'spectra_' + run.fnames[ifile]
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

    return fig
