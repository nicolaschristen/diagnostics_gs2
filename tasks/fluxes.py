from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
import pickle
import copy as cp

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
    
        kx = mygrids.kx
        ky = mygrids.ky

        time = mytime.time
        time_steady = mytime.time_steady
        it_min = mytime.it_min
        it_max = mytime.it_max
        
        phi2_avg = myfields.phi2_avg
        if myout['phi2_by_ky_present']:
            phi2_by_ky = myout['phi2_by_ky']
        
        if myout['phi2_by_mode_present']:
            phi2_kxky = np.concatenate((myout['phi2_by_mode'][:,:,nxmid:],myout['phi2_by_mode'][:,:,:nxmid]),axis=2)
            mydim = (mygrids.ny,mygrids.nx)
            phi2_kxky_tavg = np.zeros(mydim, dtype=float)
            for ik in range(ny):
                for it in range(nx):
                    phi2_kxky_tavg[ik,it] = mytime.timeavg(phi2_kxky[:,ik,it])
        else:
            mydim = (mytime.ntime,mygrids.ny,mygrids.nx)
            phi2_kxky = np.zeros(mydim, dtype=float)
            mydim = (mygrids.ny,mygrids.nx)
            phi2_kxky_tavg = np.zeros(mydim, dtype=float)

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
        mydict = {'pflx':pflx,'qflx':qflx,'vflx':vflx,'xchange':xchange,'pflx_kxky_tavg':pflx_kxky_tavg,
                'qflx_kxky_tavg':qflx_kxky_tavg,'vflx_kxky_tavg':vflx_kxky_tavg,'pflx_vpth_tavg':pflx_vpth_tavg,
                'qflx_vpth_tavg':qflx_vpth_tavg,'vflx_vpth_tavg':vflx_vpth_tavg,'pioq':pioq,'nvpa':nvpa,
                'ntheta':ntheta,'nx':nx,'ny':ny,'nxmid':nxmid,'islin':islin,'nspec':nspec,'spec_names':spec_names,
                'naky':naky,'kx':kx,'ky':ky,'time':time,'time_steady':time_steady,'it_min':it_min,'it_max':it_max,
                'phi2_avg':phi2_avg,'phi2_by_ky':phi2_by_ky,'has_flowshear':has_flowshear,
                'phi2_kxky_tavg':phi2_kxky_tavg,'phi2_kxky':phi2_kxky}
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
    
    # A lot of stuff is the same for all runs
    islin = full_fluxes[0]['islin']
    has_flowshear = full_fluxes[0]['has_flowshear']
    twin = full_time[0].twin
    nspec = full_fluxes[0]['nspec']
    spec_names = full_fluxes[0]['spec_names']
    nx = full_fluxes[0]['nx']
    ny = full_fluxes[0]['ny']
    naky = full_fluxes[0]['naky']
    kx = full_fluxes[0]['kx']
    ky = full_fluxes[0]['ky']
    datfile_name = run.out_dir + run.fnames[0] + '.grids.dat'
    with open(datfile_name,'rb') as datfile:
        my_grids = pickle.load(datfile)

    # Stitching the arrays together
    stitch_my_time = cp.deepcopy(full_time[0])
    stitch_my_time.ntime = Nt_tot
    stitch_my_time.time = np.zeros(Nt_tot)

    stitch_pflx = np.zeros((Nt_tot,nspec))
    stitch_qflx = np.zeros((Nt_tot,nspec))
    stitch_vflx = np.zeros((Nt_tot,nspec))
    stitch_pioq = np.zeros((Nt_tot,nspec))
    
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

    stitch_my_time.it_min = int(np.ceil((1.0-twin)*stitch_my_time.ntime))
    stitch_my_time.it_max = stitch_my_time.ntime-1
    stitch_my_time.time_steady = stitch_my_time.time[stitch_my_time.it_min:stitch_my_time.it_max]
    stitch_my_time.ntime_steady = stitch_my_time.time_steady.size

    # Computing time averaged versions of stitched fluxes vs (kx,ky)
    for ik in range(ny):
        for it in range(nx):
            stitch_phi2_kxky_tavg[ik,it] = stitch_my_time.timeavg(stitch_phi2_kxky[:,ik,it])
            for ispec in range(myout['nspec']):
                stitch_pflx_kxky_tavg[ispec,ik,it] = stitch_my_time.timeavg(stitch_pflx_kxky[:,ispec,ik,it])
                stitch_qflx_kxky_tavg[ispec,ik,it] = stitch_my_time.timeavg(stitch_qflx_kxky[:,ispec,ik,it])
                stitch_vflx_kxky_tavg[ispec,ik,it] = stitch_my_time.timeavg(stitch_vflx_kxky[:,ispec,ik,it])

    # Plotting the stitched fluxes
    ifile = None
    stitch_dict = {'pflx':stitch_pflx,'qflx':stitch_qflx,'vflx':stitch_vflx,'pioq':stitch_pioq,
            'nx':nx,'ny':ny,'islin':islin,'has_flowshear':has_flowshear,'nspec':nspec,'spec_names':spec_names,
            'naky':naky,'kx':kx,'ky':ky,'phi2_avg':stitch_phi2_avg,'phi2_by_ky':stitch_phi2_by_ky,
            'pflx_kxky_tavg':stitch_pflx_kxky_tavg,'qflx_kxky_tavg':stitch_qflx_kxky_tavg,
            'vflx_kxky_tavg':stitch_vflx_kxky_tavg,'phi2_kxky_tavg':stitch_phi2_kxky_tavg}
    plot_fluxes(ifile,run,stitch_my_time,stitch_dict)

def plot_fluxes(ifile,run,mytime,mydict):

    islin = mydict['islin']
    has_flowshear = mydict['has_flowshear']
    
    # t grid
    time = mytime.time
    time_steady = mytime.time_steady
    it_min = mytime.it_min
    it_max = mytime.it_max

    # k grids
    nx = mydict['nx']
    ny = mydict['ny']
    naky = mydict['naky']
    kx = mydict['kx']
    ky = mydict['ky']

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
    phi2_avg = mydict['phi2_avg']
    phi2_by_ky = mydict['phi2_by_ky']
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
            gplot.plot_1d(time,phi2_avg,'$t (a/v_{t})$',title)
        plt.grid(True)
        write_fluxes_vs_t = True
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
    print("-- plotting particle flux")
    if pflx is not None:
        title = '$\Gamma_{GS2}$'
        plot_flux_vs_t(islin,nspec,spec_names,mytime,pflx,title)
        write_fluxes_vs_t = True
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
    print("-- plotting heat flux")
    if qflx is not None:
        title = '$Q_{GS2}$'
        plot_flux_vs_t(islin,nspec,spec_names,mytime,qflx,title)
        write_fluxes_vs_t = True
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
    print("-- plotting momentum flux")
    if vflx is not None:
        title = '$\Pi_{GS2}$'
        plot_flux_vs_t(islin,nspec,spec_names,mytime,vflx,title,)
        write_fluxes_vs_t = True
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
        title = '$\Pi_{GS2}/Q_{GS2}$'
        for idx in range(nspec):
            plt.plot(mytime.time_steady,pioq[it_min:it_max,idx],label=spec_names[idx])
        plt.title(title)
        plt.xlabel('$t (a/v_t)$')
        plt.legend()
        plt.grid(True)
        write_fluxes_vs_t = True
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
    print("-- plotting phi2 by ky")
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
        plt.legend(prop={'size': 11}, ncol=6)

        plt.grid(True)
        write_fluxes_vs_t = True
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)

        if naky>4:
            
            tmp_pdf_id = tmp_pdf_id+1
            title = '$\\langle|\phi^{2}|\\rangle_{\\theta,k_x}$ for low $k_y$'
            #plt.figure(figsize=(8,8)) # NDCDEL
            if islin:
                title = '$\\ln$' + title
                plt.semilogy(time, np.log(phi2_by_ky[:,0]),label='ky = '+'{:5.3f}'.format(ky[0]),linestyle='dashed',color='black')
                for iky in range(1,5) :
                    plt.semilogy(time, np.log(phi2_by_ky[:,iky]),label='ky = '+'{:5.3f}'.format(ky[iky]),color=my_colors[iky-1])
            else:
                plt.plot(time[:], phi2_by_ky[:,0],label='ky = '+'{:5.3f}'.format(ky[0]),linestyle='dashed',color='black')
                #for iky in range(1,4) :# NDCDEL
                for iky in range(1,5) :
                    plt.semilogy(time[:], phi2_by_ky[:,iky],label='ky = '+'{:5.3f}'.format(ky[iky]),color=my_colors[iky-1])
            #plt.xlabel('$t$') # NDCDEL
            plt.xlabel('$t (a/v_t)$')
            #plt.ylabel('$\\langle|\phi^{2}|\\rangle_{\\theta,k_x}$') # NDCDEL
            plt.title(title)
            plt.legend()
            plt.grid(True)
            # NDCDEL
            #axes = plt.gca()
            #axes.set_xlim([0,500])
            #plt.savefig('terrific.pdf')
            # endNDCDEL
            write_fluxes_vs_t = True
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1

            title = '$\\langle|\phi^{2}|\\rangle_{\\theta,k_x}$ for high $k_y$'
            if islin:
                title = '$\\ln$' + title
                for iky in range(naky-5,naky) :
                    plt.semilogy(time, np.log(phi2_by_ky[:,iky]),label='ky = '+'{:5.3f}'.format(ky[iky]),color=my_colors[iky-1])
            else:
                for iky in range(naky-5,naky) :
                    plt.semilogy(time, phi2_by_ky[:,iky],label='ky = '+'{:5.3f}'.format(ky[iky]),color=my_colors[iky-1])
            plt.xlabel('$t (a/v_t)$')
            plt.title(title)
            plt.legend()
            plt.grid(True)
            write_fluxes_vs_t = True
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)

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

    # Plot phi2 averaged over t and theta, vs (kx,ky)
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
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

    print('complete')

    #print()
    #print('producing plots of fluxes vs (vpa,theta)...',end='')

    #write_vpathetasym = False
    #tmp_pdf_id = 1
    #pdflist = []
    #if pflx_vpth_tavg is not None and mygrids.vpa is not None:
    #    title = '$\Gamma_{GS2}$'
    #    plot_flux_vs_vpth(mygrids,pflx_vpth_tavg,title)
    #    write_vpathetasym = True
    #    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    #    gplot.save_plot(tmp_pdfname, run, ifile)
    #    pdflist.append(tmp_pdfname)
    #    tmp_pdf_id = tmp_pdf_id+1
    #if qflx_vpth_tavg is not None and mygrids.vpa is not None:
    #    title = '$Q_{GS2}$'
    #    plot_flux_vs_vpth(mygrids,qflx_vpth_tavg,title)
    #    write_vpathetasym = True
    #    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    #    gplot.save_plot(tmp_pdfname, run, ifile)
    #    pdflist.append(tmp_pdfname)
    #    tmp_pdf_id = tmp_pdf_id+1
    #if vflx_vpth_tavg is not None and mygrids.vpa is not None:
    #    title = '$\Pi_{GS2}$'
    #    plot_flux_vs_vpth(mygrids,vflx_vpth_tavg,title)
    #    write_vpathetasym = True
    #    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    #    gplot.save_plot(tmp_pdfname, run, ifile)
    #    pdflist.append(tmp_pdfname)
    #    tmp_pdf_id = tmp_pdf_id+1

    #if write_vpathetasym:  
    #    merged_pdfname = 'fluxes_vs_vpa_theta'
    #    gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

    #print('complete')

def plot_flux_vs_t(islin,nspec,spec_names,mytime,flx,title,):

    fig=plt.figure(figsize=(12,8))
    dum = np.empty(mytime.ntime_steady)
    if islin:
        title = '$\\ln($' + title + '$)$'
    # generate a curve for each species
    # on the same plot
    
    # plot time-traces for each species
    for idx in range(nspec):
        # get the time-averaged flux
        if islin:
            plt.plot(mytime.time,np.log(flx[:,idx]),label=spec_names[idx])
        else:
            plt.plot(mytime.time,flx[:,idx],label=spec_names[idx])
    
    # plot time-averages
    for idx in range(nspec):
        if islin:
            flxavg = mytime.timeavg(np.log(np.absolute(flx[:,idx])))
            dum.fill(flxavg)
        else:
            flxavg = mytime.timeavg(flx[:,idx])
            dum.fill(flxavg)
        plt.plot(mytime.time_steady,dum,'--')
        print('flux avg for '+spec_names[idx]+': '+str(flxavg))

    plt.xlabel('$t (a/v_t)$')
    plt.xlim([mytime.time[0],mytime.time[-1]])
    plt.title(title)
    plt.legend()
    plt.grid(True)

    return fig

def plot_flux_vs_kxky(ispec,spec_names,kx,ky,flx,title,has_flowshear):

    from gs2_plotting import plot_2d

    if has_flowshear:
        xlab = '$\\bar{k}_{x}\\rho_i$'
    else:
        xlab = '$k_{x}\\rho_i$'
    ylab = '$k_{y}\\rho_i$'

    cmap = 'Blues' # 'Reds','Blues'
    z = np.abs(flx[ispec,:,:]) # take absolute value of contribution to fluxes
    z_min, z_max = 0.0, z.max()
    
    title = 'Contributions to ' + title
    if ispec > 1:
        title += ' (impurity ' + str(ispec-1) + ')'
    else:
        title += ' (' + spec_names[ispec] + 's)'
    fig = plot_2d(z,kx,ky,z_min,z_max,xlab,ylab,title,cmap)

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

    cmap = 'RdBu'# 'RdBu_r','Blues'
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
