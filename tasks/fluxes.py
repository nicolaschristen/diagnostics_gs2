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
    
        theta = mygrids.theta
        theta0 = mygrids.theta0
        kx = mygrids.kx
        ky = mygrids.ky
        jtwist = mygrids.jtwist

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
        mydict = {'pflx':pflx,'qflx':qflx,'vflx':vflx,'xchange':xchange,
                'pflx_kxky':pflx_kxky,'qflx_kxky':qflx_kxky,'vflx_kxky':vflx_kxky,
                'pflx_kxky_tavg':pflx_kxky_tavg,
                'qflx_kxky_tavg':qflx_kxky_tavg,'vflx_kxky_tavg':vflx_kxky_tavg,'pflx_vpth_tavg':pflx_vpth_tavg,
                'qflx_vpth_tavg':qflx_vpth_tavg,'vflx_vpth_tavg':vflx_vpth_tavg,'pioq':pioq,'nvpa':nvpa,
                'ntheta':ntheta,'nx':nx,'ny':ny,'nxmid':nxmid,'islin':islin,'nspec':nspec,'spec_names':spec_names,
                'naky':naky,'kx':kx,'ky':ky,'theta':theta,'theta0':theta0,'jtwist':jtwist,
                'time':time,'time_steady':time_steady,'it_min':it_min,'it_max':it_max,
                'phi_bytheta_tfinal':phi_bytheta_tfinal, 'phi2_avg':phi2_avg,'phi2_by_ky':phi2_by_ky,'has_flowshear':has_flowshear,
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

    tmin = stitch_my_time.time[-1] * (1.0-twin)
    stitch_my_time.it_min = 0
    while stitch_my_time.time[stitch_my_time.it_min] < tmin:
        stitch_my_time.it_min += 1
    stitch_my_time.it_max = stitch_my_time.ntime-1
    stitch_my_time.time_steady = stitch_my_time.time[stitch_my_time.it_min:stitch_my_time.it_max]
    stitch_my_time.ntime_steady = stitch_my_time.time_steady.size

    # Computing time averaged versions of stitched fluxes vs t
    for ispec in range(nspec):
        stitch_pflx_tavg[ispec] = stitch_my_time.timeavg(stitch_pflx[:,ispec])
        stitch_qflx_tavg[ispec] = stitch_my_time.timeavg(stitch_qflx[:,ispec])
        stitch_vflx_tavg[ispec] = stitch_my_time.timeavg(stitch_vflx[:,ispec])
    # Computing time averaged versions of stitched fluxes vs (kx,ky)
    for ik in range(ny):
        for it in range(nx):
            stitch_phi2_kxky_tavg[ik,it] = stitch_my_time.timeavg(stitch_phi2_kxky[:,ik,it])
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
            'vflx_kxky_tavg':stitch_vflx_kxky_tavg,'phi2_kxky_tavg':stitch_phi2_kxky_tavg,
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
        plot_flux_vs_t(islin,nspec,spec_names,mytime,pflx,title,ylims,label_ypos,avg_in_title)
        write_fluxes_vs_t = True
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
        plot_flux_vs_t(islin,nspec,spec_names,mytime,qflx,title,ylims,label_ypos,avg_in_title)
        write_fluxes_vs_t = True
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
        plot_flux_vs_t(islin,nspec,spec_names,mytime,vflx,title,ylims,label_ypos,avg_in_title)
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
            # indicating area of saturation
            plt.xlim((time[0], time[-1]))
            plt.axvline(x=time_steady[0], color='grey', linestyle='-')
            ax = plt.gca()
            ax.axvspan(time_steady[0], time_steady[-1], alpha=0.1, color='grey')
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
    energy_dens_x = np.squeeze(kx*np.sum(phi2_kxky_tavg[1:,:],axis=0))
    plt.loglog(kx, energy_dens_x, color=gplot.myblue, linewidth=3.0)
    [xmin,xmax] = plt.gca().get_xlim()
    xvec = np.linspace(xmin,xmax)
    [ymin,ymax] = plt.gca().get_ylim()
    yvec = np.linspace(ymin,ymax)
    fit = (xvec/kx[-1])**(-7.0/3.0) * energy_dens_x[-1]
    plt.loglog(xvec, fit, color='k', linewidth=1.5, label='$(\\bar{k}_x\\rho_i)^{-7/3}$')
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))
    plt.grid(True)
    plt.xlabel('$\\bar{k}_{x}\\rho_i$')
    plt.ylabel('$\\sum_{k_y\\neq 0} \\bar{k}_x\\rho_i\\langle \\vert\\varphi\\vert ^2\\rangle_{t,\\theta}$')
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
    plt.loglog(ky, energy_dens_y, color=gplot.myblue, linewidth=2.0)
    [xmin,xmax] = plt.gca().get_xlim()
    xvec = np.linspace(xmin,xmax)
    [ymin,ymax] = plt.gca().get_ylim()
    yvec = np.linspace(ymin,ymax)
    fit = (xvec/ky[-1])**(-7.0/3.0) * energy_dens_y[-1]
    plt.loglog(xvec, fit, color='k', linewidth=1.5, label='$(k_y\\rho_i)^{-7/3}$')
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
    # First zonal mode
    plt.semilogy(kx,phi2_kxky_tavg[0,:], marker='o', color=gplot.myblue, \
            markersize=8, markerfacecolor=gplot.myblue, markeredgecolor=gplot.myblue, linewidth=2.0)
    plt.grid(True)
    plt.xlabel('$\\bar{k}_{x}\\rho_i$')
    plt.ylabel('$\\langle\\vert\\varphi\\vert ^2\\rangle_{t,\\theta}$')
    plt.title('$k_y\\rho_i = 0$')
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run, ifile)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1
    # Then non-zonal modes
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

        for dmid in range(jtwist*iky):

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

    print('complete')


def plot_flux_vs_t(islin,nspec,spec_names,mytime,flx,ylabel,ylims=None,my_label_ypos=None,avg_in_title=None):

    fig=plt.figure(figsize=(12,8))
    if islin:
        ylabel = '$\\ln($' + ylabel + '$)$'

    #my_colorlist = plt.cm.YlGnBu(np.linspace(0.5,1,nspec)) # for old algo
    my_colorlist = plt.cm.YlOrBr(np.linspace(0.5,1,nspec)) # for new algo

    my_curves = []
    my_labels = ['$^2H$','$e^-$','$^{12}C$']
    #my_linestyle_list = ['-', '--', ':']
    my_linestyle_list = ['-', '-', '-']

    # indicating area of saturation
    plt.axvline(x=mytime.time_steady[0], color='grey', linestyle='-')
    ax = plt.gca()
    ax.axvspan(mytime.time_steady[0], mytime.time_steady[-1], alpha=0.1, color='grey')
    
    # plot time-traces for each species
    for idx in range(nspec):
        # get the time-averaged flux
        if islin:
            crv, = plt.plot(mytime.time,np.log(flx[:,idx]),color=my_colorlist[idx],linewidth=3.0, \
                    linestyle=my_linestyle_list[idx])
        else:
            crv, = plt.plot(mytime.time,flx[:,idx],color=my_colorlist[idx],linewidth=3.0, \
                    linestyle=my_linestyle_list[idx])
        my_curves.append(crv)
    
    # plot time-averages
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

                plt.title('avg = {:.2f}'.format(flxavg))

            print('flux avg for '+spec_names[idx]+': '+str(flxavg))

    plt.xlabel('$t [L/v_{th,i}]$')
    plt.ylabel(ylabel)
    plt.xlim([mytime.time[0],mytime.time[-1]])
    if ylims is not None:
        plt.ylim(ylims)
    plt.grid(True)

    my_legend = plt.legend(my_curves,my_labels,frameon=True,fancybox=False,framealpha=1.0,loc='upper left')
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
