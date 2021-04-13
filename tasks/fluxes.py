from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import numpy as np
from numpy import pi as pi
import pickle
import copy as cp
from scipy.integrate import simps
import os

import gs2_plotting as gplot
from gs2_plotting import plot_2d
from plot_phi2_vs_time import plot_phi2_ky_vs_t

f_labels = {'es':r'$\vert\phi\vert^2$', 'apar':r'$\vert A_\parallel\vert^2$', 'bpar':r'$\vert B_\parallel\vert^2$'}
fk_labels = {'es':r'$\vert\hat{\phi}\vert^2_k$', 'apar':r'$\vert \hat{A_\parallel}\vert^2_k$', 'bpar':r'$\vert \hat{B_\parallel}_k\vert^2_k$'}
f_colors = {'es':'blue', 'apar':'red', 'bpar':'green'}

flux_labels = [r'$\Gamma_{GS2}$',r'$\Pi_{GS2}$',r'$Q_{GS2}$']
it0 = 20
itf = -5 #-1 #-1 for most things as the last index is a time average.

def my_single_task(ifile,run,myin,myout,mygrids,mytime,myfields,stitching=False):

    # Compute and save to dat file
    if not run.only_plot:
        # OB 140918 ~ mygrids.ny = naky. Before, ny was defined as mygrids.ny, but I have changed it to ny specified in the input file, so we have both.
        #             (Same for x, and I replaced all further ny,nx that should have been naky and nakx, respectively)
        kx_gs2 = myout['kx']
        ky = myout['ky']
        
        #gexb = myin['dist_fn_knobs']['g_exb']
        shat = myin['theta_grid_parameters']['shat']
        jtwist = int(myin['kt_grids_box_parameters']['jtwist'])

        dky = 1./myin['kt_grids_box_parameters']['y0']
        dkx = 2.*pi*abs(shat)*dky/jtwist
        
        nakx = kx_gs2.size
        naky = ky.size
        
        theta = myout['theta']
        ntheta = len(theta)
        
        theta0 = myout['theta0']
        ntheta0 = len(kx_gs2)

        gds2 = myout['gds2']
        gds21 = myout['gds21']
        gds22 = myout['gds22']
        gbdrift = myout['gbdrift']
        gbdrift0 = myout['gbdrift0']
        cvdrift = myout['cvdrift']
        cvdrift0 = myout['cvdrift0']

        full_kperp = np.zeros((naky, ntheta0, ntheta))
        full_drift = np.zeros((naky, ntheta0, ntheta))
        kperp2_avg = np.zeros((naky, ntheta0))
        for iky in range(naky):
            for ith0 in range(ntheta0):
                for ith in range(ntheta):
                    # n.b. don't sqrt kperp2 just yet - we want to theta average it first.
                    full_kperp[iky,ith0,ith] = np.power(ky[iky],2) * ( gds2[ith] + 2*theta0[iky,ith0]*gds21[ith] + np.power(theta0[iky,ith0],2) * gds22[ith] )
                    full_drift[iky,ith0,ith] = ky[iky] * (gbdrift[ith] + cvdrift[ith] + theta0[iky,ith0]*(gbdrift0[ith] + cvdrift0[ith]))
        
        kperp2_thavg = simps(full_kperp,x=theta)/(2.0*np.pi)
        full_kperp = np.sqrt(full_kperp)

        # kx grid has not been sorted to be monotonic yet, so maximum occurs ~halfway.
        ikx_max = int(round((nakx-1)/2))
        ikx_min = ikx_max+1

        # sorting kx_gs2 to get monotonic kx
        kx = np.concatenate((kx_gs2[ikx_min:],kx_gs2[:ikx_min]))
       
        fields_present = []
        fields2 = []
        fields2_by_mode = []
        fields2_by_ky = []
        fields2_by_kx = []
        
        if myout['phi2_present']:
            fields2.append(myout['phi2'])
            fields2_by_mode.append(myout['phi2_by_mode'])
            fields2_by_ky.append(myout['phi2_by_ky'])
            fields2_by_kx.append(myout['phi2_by_kx'])
            fields_present.append('es')
        if myout['apar2_by_mode_present']:
            fields2.append(myout['apar2'])
            fields2_by_mode.append(myout['apar2_by_mode'])
            fields2_by_ky.append(myout['apar2_by_ky'])
            fields2_by_kx.append(myout['apar2_by_kx'])
            fields_present.append('apar')
        if myout['bpar2_by_mode_present']:
            fields2.append(myout['bpar2'])
            fields2_by_mode.append(myout['bpar2_by_mode'])
            fields2_by_ky.append(myout['bpar2_by_ky'])
            fields2_by_kx.append(myout['bpar2_by_kx'])
            fields_present.append('bpar')
        
        fields2 = np.asarray(fields2)            # field, t
        fields2 = append_tavg(fields2, 1, mytime)

        fields2_by_mode = np.asarray(fields2_by_mode) # field, t, ky, kx
        fields2_by_mode = append_tavg(fields2_by_mode, 1, mytime)

        fields2_by_ky = np.asarray(fields2_by_ky)    # field, t, ky
        fields2_by_ky = append_tavg(fields2_by_ky, 1, mytime)

        fields2_by_kx = np.asarray(fields2_by_kx)    # field, t, kx
        fields2_by_kx = append_tavg(fields2_by_kx, 1, mytime)
      
        # sort kx-dependent quantities to monotonic kx.
        fields2_by_mode = np.concatenate(( fields2_by_mode[:,:,:,ikx_min:], fields2_by_mode[:,:,:,:ikx_min] ), axis=3 )
        fields2_by_kx = np.concatenate(( fields2_by_kx[:,:,ikx_min:], fields2_by_kx[:,:,:ikx_min] ), axis=2 )
        
        nl_term_by_mode = fields2_by_mode * kperp2_thavg[np.newaxis, np.newaxis, :, :]
        nl_term_by_mode = append_tavg(nl_term_by_mode, 1, mytime)

        fields_t_present = []
        fields_t = []
        
        if myout['phi_t_present']:
            fields_t.append(myout['phi_t'])
            fields_t_present.append('es')
        if myout['apar_t_present']:
            fields_t.append(myout['apar_t'])
            fields_t_present.append('apar')
        if myout['bpar_t_present']:
            fields_t.append(myout['bpar_t'])
            fields_t_present.append('bpar')
        fields_t = np.asarray(fields_t)   # [field, t, ky, kx, theta, ri]

        if len(fields_t_present)>0:
            fields_t_abs = np.sqrt(np.sum(np.power(fields_t, 2), axis=5, keepdims=True))       # absolute value of fields squared, with theta included [field, t, ky, kx, theta]
            fields_t = np.concatenate(( fields_t, fields_t_abs ), axis = 5)
            fields_t = np.concatenate(( fields_t[:,:,:,ikx_min:,:,:], fields_t[:,:,:,:ikx_min,:,:] ), axis=3)   # [field, t, ky, kx, theta, ria])
        
        fluxes_present = myout['es_heat_flux_present'] or myout['apar_heat_flux_present'] or myout['bpar_heat_flux_present']
        fluxes_by_mode_present = myout['es_heat_flux_by_mode_present'] or myout['apar_heat_flux_by_mode_present'] or myout['bpar_heat_flux_by_mode_present']
        
        fluxes = []
        
        if fluxes_present or fluxes_by_mode_present:
            fluxes = []                 # [field, flux, t, spec]
            fluxes_by_mode = []         # [field, flux, t, spec, ky, kx]
            for field in fields_present:
                tmp_fluxes = []
                tmp_fluxes_by_mode = []
                for flux_type in ['part_flux', 'mom_flux', 'heat_flux']:
                    if myout['{}_{}_present'.format(field, flux_type)]:
                        tmp_fluxes.append(myout['{}_{}'.format(field,flux_type)])
                    if myout['{}_{}_by_mode_present'.format(field, flux_type)]:
                        tmp_fluxes_by_mode.append(myout['{}_{}_by_mode'.format(field,flux_type)])
                fluxes.append(tmp_fluxes)
                fluxes_by_mode.append(tmp_fluxes_by_mode)
 
            if len(fluxes) == 0: 
                fluxes = None
                fluxes_by_mode = np.asarray(fluxes_by_mode)
            elif len(fluxes_by_mode) == 0:
                fluxes_by_mode = None
                fluxes = np.asarray(fluxes)
            else:
                fluxes = np.asarray(fluxes)
                fluxes_by_mode = np.asarray(fluxes_by_mode)
        else:
            fluxes = None
            fluxes_by_mode = None
            
        if fluxes is not None:
            pioq = np.divide(fluxes[:,1,:,:],fluxes[:,2,:,:])
            pioq = append_tavg(pioq, 1, mytime)
            fluxes = append_tavg(fluxes, 2, mytime)
        else:
            pioq = None

        if fluxes_by_mode is not None:
            fluxes_by_mode = np.concatenate(( fluxes_by_mode[:,:,:,:,:,ikx_min:], fluxes_by_mode[:,:,:,:,:,:ikx_min] ), axis=5 )
            fluxes_by_mode = append_tavg(fluxes_by_mode, 2, mytime)

        import time

        fluxes_by_mode_tavg = mytime.timeavg(fluxes_by_mode, axis=2)[:,:,np.newaxis,:,:,:]
        t0= time.time()
        fluxes_by_mode = np.concatenate((fluxes_by_mode, fluxes_by_mode_tavg), axis=2)
        t1 = time.time() - t0     

        print("reassigning took {:1.10f} seconds".format(t1))
        # need to multiply this by rhoc/(g_exb*rmaj**2)
        #prandtl = np.copy(es_vflx[:,0]*tprim[0]/(es_qflx[:,0]*q)
         
       
        # Save computed quantities      OB 140918 ~ added tri,kap to saved dat.
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.fluxes.dat'

        mydict = {
                    'fields_present':fields_present,
                    'fluxes':fluxes, 
                    'fields_t_present':fields_t_present,
                    'fields_t':fields_t,
                    'fluxes_by_mode':fluxes_by_mode, 
                    'fields2':fields2,
                    'fields2_by_mode':fields2_by_mode,
                    'fields2_by_ky':fields2_by_ky,
                    'fields2_by_kx':fields2_by_kx,
                    'pioq':pioq,
                    'nl_term_by_mode':nl_term_by_mode,
                    'kx':kx,
                    'nakx':nakx,
                    'ky':ky,
                    'naky':naky,
                    'theta':theta,
                    'ntheta':ntheta,
                    'time':mytime,
                    'full_kperp':full_kperp,
                    'full_drift':full_drift,
                    'input_file':myin
        }

        with open(datfile_name,'wb') as datfile:
            pickle.dump(mydict,datfile, protocol=4)
            

    # Read from dat files
    else:
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.fluxes.dat'
        with open(datfile_name,'rb') as datfile:
            mydict = pickle.load(datfile)
        
    if not run.no_plot and not stitching: # plot fluxes for this single file
            
        plot_fluxes(ifile,run,mydict)

def append_tavg(data, t_axis, mytime):
    return np.concatenate( ( data, np.expand_dims(mytime.timeavg(data, axis = t_axis), t_axis) ), axis = t_axis )

def stitching_fluxes(run):

    # Only executed if we want to plot the data
    Nfile = len(run.fnames)
    full_fluxes = [dict() for ifile in range(Nfile)]
    full_grids = [dict() for ifile in range(Nfile)]

    # Reading .dat file for each run
    # and calc how long the stitched array will be
    nt_tot = 0
    
    for ifile in range(Nfile):
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.fluxes.dat'
        with open(datfile_name,'rb') as datfile:
            full_fluxes[ifile] = pickle.load(datfile)
        nt_tot += len(full_fluxes[ifile]['time'].time)
        if ifile > 0:
            nt_tot -= 1     

    print(nt_tot)

    # Stitching the arrays together
    stitch_mytime = cp.deepcopy(full_fluxes[0]['time'])
    stitch_mytime.ntime = nt_tot
    stitch_mytime.time = np.zeros(nt_tot)
    
    fields_dims = full_fluxes[0]['fields2_by_mode'].shape
    nfields = fields_dims[0]
    nspec = full_fluxes[0]['input_file']['species_knobs']['nspec']
    ky = full_fluxes[0]['ky']
    naky = len(ky)
    kx = full_fluxes[0]['kx']
    nakx = len(kx)
    theta = full_fluxes[0]['theta']
    ntheta = len(theta)

    if full_fluxes[0]['fluxes'] is not None:
        nfluxes = full_fluxes[0]['fluxes'].shape[0]
        stitch_fluxes = np.zeros((nfluxes, nfields, nt_tot, nspec))
        stitch_pioq = np.zeros((nfields, nt_tot, nspec))
       
    if full_fluxes[0]['fluxes_by_mode'] is not None:
        nfluxes = full_fluxes[0]['fluxes_by_mode'].shape[0]
        stitch_fluxes_by_mode = np.zeros((nfluxes, nfields, nt_tot, nspec, naky, nakx))
    
    stitch_fields2 = np.zeros(( nfields, nt_tot ))
    stitch_fields2_by_mode = np.zeros(( nfields, nt_tot, naky, nakx ))
    stitch_nl_term_by_mode = np.zeros(( nfields, nt_tot, naky, nakx ))
    stitch_fields2_by_ky = np.zeros(( nfields, nt_tot, naky ))
    stitch_fields2_by_kx = np.zeros(( nfields, nt_tot, nakx ))
    
    if full_fluxes[0]['fields_t'] is not None:
        stitch_fields_t = np.zeros(( nfields, nt_tot, naky, nakx, ntheta, 3 ))
    else:
        stitch_fields_t = None
    
    it_tot = 0
    for ifile in range(Nfile):
        for it in range(min(1,ifile),full_fluxes[ifile]['time'].ntime):
            stitch_mytime.time[it_tot] = full_fluxes[ifile]['time'].time[it]
            
            stitch_fields2[:,it_tot] = full_fluxes[ifile]['fields2'][:,it]
            stitch_fields2_by_mode[:,it_tot,:,:] = full_fluxes[ifile]['fields2_by_mode'][:,it,:,:]
            stitch_fields2_by_ky[:,it_tot,:] = full_fluxes[ifile]['fields2_by_ky'][:,it,:]
            stitch_fields2_by_kx[:,it_tot,:] = full_fluxes[ifile]['fields2_by_kx'][:,it,:]
            stitch_nl_term_by_mode[:,it_tot,:,:] = full_fluxes[ifile]['nl_term_by_mode'][:,it,:,:]
            
            if stitch_fluxes is not None:
                stitch_fluxes[:,:,it_tot,:] = full_fluxes[ifile]['fluxes'][:,:,it,:]
                stitch_pioq[:,it_tot,:] = full_fluxes[ifile]['pioq'][:,it,:]
            
            if stitch_fluxes_by_mode is not None:
                stitch_fluxes_by_mode[:,:,it_tot,:,:,:] = full_fluxes[ifile]['fluxes_by_mode'][:,:,it,:,:,:]
            
            if stitch_fields_t is not None:
                stitch_fields_t[:,it_tot,:,:,:,:] = full_fluxes[ifile]['fields_t'][:,it,:,:,:,:]

            it_tot += 1

    stitch_mytime.it_min = int(np.ceil((1.0-run.twin)*stitch_mytime.ntime))
    stitch_mytime.it_max = stitch_mytime.ntime-1
    stitch_mytime.time_steady = stitch_mytime.time[stitch_mytime.it_min:stitch_mytime.it_max]
    stitch_mytime.ntime_steady = stitch_mytime.time_steady.size

    # Computing time averaged versions of stitched fluxes vs (kx,ky) OB 140918 ~ New timeavg doesn't need us to explicitly loop over.

    stitch_fields2 = append_tavg(stitch_fields2, 1, stitch_mytime)
    stitch_fields2_by_mode = append_tavg(stitch_fields2_by_mode, 1, stitch_mytime)
    stitch_fields2_by_kx = append_tavg(stitch_fields2_by_kx, 1, stitch_mytime)
    stitch_fields2_by_ky = append_tavg(stitch_fields2_by_ky, 1, stitch_mytime)

    if stitch_fluxes is not None:
        stitch_fluxes = append_tavg(stitch_fluxes, 2, stitch_mytime)
        stitch_pioq = append_tavg(stitch_pioq, 1, stitch_mytime)
    if stitch_fluxes_by_mode is not None:
        stitch_fluxes_by_mode = append_tavg(stitch_fluxes_by_mode, 2, stitch_mytime)

    # Plotting the stitched fluxes
    ifile = None
    mydict = {
                'fields_present':full_fluxes[0]['fields_present'],
                'fields_t_present':full_fluxes[0]['fields_t_present'],
                'fluxes':stitch_fluxes, 
                'fluxes_by_mode':stitch_fluxes_by_mode, 
                'fields_t':stitch_fields_t,
                'fields2':stitch_fields2,
                'fields2_by_mode':stitch_fields2_by_mode,
                'fields2_by_ky':stitch_fields2_by_ky,
                'fields2_by_kx':stitch_fields2_by_kx,
                'pioq':stitch_pioq,
                'nl_term_by_mode':stitch_nl_term_by_mode,
                'kx':kx,
                'nakx':nakx,
                'ky':ky,
                'naky':naky,
                'ntheta':ntheta,
                'time':stitch_mytime,
                'full_kperp':full_fluxes[0]['full_kperp'],
                'full_drift':full_fluxes[0]['full_drift'],
                'input_file':full_fluxes[-1]['input_file']
    }

    # Save computed quantities      OB 140918 ~ added tri,kap to saved dat.
    datfile_name = run.work_dir + 'fluxes_stitch.dat'
    with open(datfile_name,'wb') as datfile:
        pickle.dump(mydict,datfile,protocol=4)

    
    if not run.no_plot:
        plot_fluxes(ifile,run,mydict)

def plot_fluxes(ifile,run,mydict):

    # t grid
    mytime = mydict['time']
    time_steady = mytime.time_steady
    it_min = mytime.it_min
    it_max = mytime.it_max
    time = mytime.time
    # k grids
    naky = mydict['naky']
    nakx = mydict['nakx']
    kx = mydict['kx']
    ky = mydict['ky']
    ikxmid = np.where(kx==0)[0][0]
    
    # species
    nspec = mydict['input_file']['species_knobs']['nspec']
    nions = 0
    spec_names = []
    for ispec in range(nspec):
        if mydict['input_file']['species_parameters_{}'.format(ispec+1)]['type'] == 'ion':
            nions += 1
            spec_names.append(r'Ions \#{}'.format(nions))
        else:
            spec_names.append(r'Electrons')
            
    # flow shear?
    has_flowshear = False#mydict['input_file']['dist_fn_knobs']['g_exb'] != 0

    # fluxes vs t
    fluxes = mydict['fluxes']
    pioq = mydict['pioq']

    # fluxes vs (kx,ky)
    fluxes_by_mode = mydict['fluxes_by_mode']
    if fluxes is not None:
        nfluxes = fluxes.shape[1]
    elif fluxes_by_mode is not None:
        nfluxes = fluxes_by_mode.shape[1]

    # potential
    fields_present = mydict['fields_present']
    fields2 = mydict['fields2']
    fields2_by_mode = mydict['fields2_by_mode']
    fields2_by_ky = mydict['fields2_by_ky']
    fields2_by_kx = mydict['fields2_by_kx']
    nl_term_by_mode = mydict['nl_term_by_mode']
    
    nfields = len(fields_present)

    fields_t_present = mydict['fields_present']
    fields_t = mydict['fields_t']    

    twin = mytime.twin
    print()
    print(">>> producing plots of fluxes vs time...")

    print("-- plotting avg(phi2)")
    write_fluxes_vs_t = False
    tmp_pdf_id = 1
    pdflist = []
    
    title = r'$\langle \rm{fields} \rangle_{\theta,k_x,k_y}$'
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca()
    for ifield in range(nfields):
        label = f_labels[fields_present[ifield]]
        color = f_colors[fields_present[ifield]]
        gplot.plot_1d(time[it0:itf+1],fields2[ifield,it0:itf],'$t (a/v_{t})$', axes=ax, title=title, label=label, color=color)
    plt.grid(True)
    ax.legend(loc='best')
    write_fluxes_vs_t = True
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run, ifile)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id = tmp_pdf_id+1
    
    
    try_plot_vs_t = True

    if try_plot_vs_t:
        print("-- plotting flux vs t")
        if fluxes is not None:
            for iflux in range(nfluxes):
                title = flux_labels[iflux]
                plot_flux_vs_t(spec_names,mytime,fluxes[:,iflux,:,:],title, fields_present)
                write_fluxes_vs_t = True
                tmp_pdfname = 'tmp'+str(tmp_pdf_id)
                gplot.save_plot(tmp_pdfname, run, ifile)
                pdflist.append(tmp_pdfname)
                tmp_pdf_id = tmp_pdf_id+1
            
            print("-- plotting momentum/heat flux ratio")
            title = '$\Pi_{GS2}/Q_{GS2}$'
            plt.figure(figsize=(12,8))
            plot_flux_vs_t(spec_names, mytime, pioq, title, fields_present)
            axes = plt.gca()
            plt.grid(True)
            write_fluxes_vs_t = True
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
        
        print("-- plotting fields by ky")
        for ifield in range(nfields):
            fig = plt.figure(figsize=(16,8))
            axes = fig.gca()
            field_label = f_labels[fields_present[ifield]]
            title = r'$\langle$ {} $\rangle_{{\theta,k_x}}$'.format(field_label)
            gplot.plot_1d(time[:], fields2_by_ky[ifield,:it_max+1,0], '$t$', axes=axes, label='$k_y=0$', linestyle='dashed', color='black', log='y')
            gplot.plot_multi_1d(time[0:], [fields2_by_ky[ifield,0:it_max+1,iky] for iky in range(1,len(ky))], '$t(a/v_t)$', axes=axes, title=title, labels=['$k_y={:1.3f}$'.format(aky) for aky in ky[1:]])
            plt.grid(True)
            write_fluxes_vs_t = True
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
            
            if naky>4:
                tmp_pdf_id = tmp_pdf_id+1
                title = r'$\langle$ {} $\rangle_{{\theta,k_x}}$ for low $k_y$'.format(field_label)
                plt.figure(figsize=(16,8))
                axes = plt.gca()
                gplot.plot_1d(time, fields2_by_ky[ifield, :it_max+1, 0], '$t$', axes=axes, label='$k_y=0$', linestyle='dashed', color='black', log='y')
                gplot.plot_multi_1d(time, [fields2_by_ky[ifield,:it_max+1,iky] for iky in range(1,5)], '$t(a/v_t)$', axes=axes, title=title, labels=['$k_y={:5.3f}$'.format(aky) for aky in ky[1:5]], log='y')
                plt.grid(True)
                
                tmp_pdfname = 'tmp'+str(tmp_pdf_id)
                gplot.save_plot(tmp_pdfname, run, ifile)
                pdflist.append(tmp_pdfname)
                tmp_pdf_id = tmp_pdf_id+1

                title = r'$\langle$ {} $\rangle_{{\theta,k_x}}$ for high $k_y$'.format(field_label)
                plt.figure(figsize=(16,8))
                axes = plt.gca()
                gplot.plot_multi_1d(time, [fields2_by_ky[ifield,:it_max+1,iky] for iky in range(-4,0)], '$t(a/v_t)$', axes=axes, title=title, labels=['$k_y={:5.3f}$'.format(aky) for aky in ky[naky-5:naky-1]], log='y')
                plt.grid(True)
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
    
    # Plot phi2 averaged over t and theta, vs (kx,ky)
    for ifield in range(nfields):
        title = r'$\langle$ {} $\rangle_{{t,\theta}}$'.format(fk_labels[fields_present[ifield]])
        plot_field_vs_kxky(kx, ky, fields2_by_mode[ifield,-1], False, title, jtwist = mydict['input_file']['kt_grids_box_parameters']['jtwist'])
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        write_fluxes_vs_kxky = True
        tmp_pdf_id += 1
   
    # Plot energy spectra for the central set of kxs
    kx_spread = min(8,len(kx)//2 - 1)
    ikx_step = 1
    start_index = 1
    nzky = ky[start_index:]
    energy = fields2_by_mode[:,-1,start_index:,ikxmid-kx_spread:ikxmid+kx_spread+1:1]*nzky[np.newaxis, :, np.newaxis]

    energy_kxav = np.mean(energy, axis=2)
    ikx_range = range(ikxmid-kx_spread, ikxmid+kx_spread, ikx_step)
    kx_spread_labels = [r"{:1.2f}".format(kx[i]) for i in ikx_range]
    slope_start_index = len(energy_kxav[0,:])//2
    slope = np.power(nzky/nzky[-1], -7/3)
    #slope = 5 * slope * energy_kxav[:, slope_start_index]  / slope[0]
    for ifield in range(nfields):
        fig = plt.figure(figsize=(12,8))
        axes = plt.gca()
        field_label = fk_labels[fields_present[ifield]]
        title = r'$k_y\langle$ {} $\rangle_{{\theta}}$, slope$=k_y^{{-7/3}}$'.format(field_label)
        print(energy.shape)
        print(ky.shape)
        print(ikx_range)
        gplot.plot_multi_1d(nzky, [energy[ifield, :, ikx] for ikx in range(len(ikx_range))],r'$k_y \rho_i$', axes=axes, labels = kx_spread_labels)
        # On the same figure, plot the average of the above set of kxs.
        gplot.plot_1d(nzky, np.mean(energy[ifield,:,:], axis=1), r'$k_y \rho_i$', axes=axes, grid='both',log='both', linewidth=5)
        gplot.plot_1d(nzky[slope_start_index:], 5*energy_kxav[ifield,-1]*slope[slope_start_index:], r'$k_y \rho_i$', axes=axes, title=title, grid='both',log='both', linewidth=2)
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        write_fluxes_vs_kxky = True
        tmp_pdf_id += 1
    

    if False: 
        # Plot the contribution to the nonlinear term vs kx,ky
        plot_field_vs_kxky(kx, ky, nl_phi_term, has_flowshear, r'NL contribution from $\phi$')
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id += 1

        # Plot the contribution to the nonlinear term vs kx,ky
        plot_field_vs_kxky(kx, ky, nl_apar_term, has_flowshear, r'NL contribution from $v_\parallel A_\parallel$')
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id += 1

    if fluxes_by_mode is not None:
        cmap = 'Blues' # 'Reds','Blues'
        for iflux in range(nfluxes):
            flux_name = flux_labels[iflux]
            if has_flowshear:
                xlabel = r'$\bar{k}_{x}\rho_{\rm{ref}}$'
            else:
                xlabel = r'$k_{x}\rho_{\rm{ref}}$'
            ylabel = r'$k_y\rho_{\rm{ref}}$' 
            fig, axes = plt.subplots(ncols=nspec, nrows=nfields, figsize=((12*nspec, 8*nfields)), sharex=True, sharey=True)
            if type(axes).__name__ not in ['ndarray', 'list']:
                axes = np.array([axes])[np.newaxis,np.newaxis]
            elif nspec == 1:
                axes = axes[:,np.newaxis]
            elif nfields == 1:
                axes = axes[np.newaxis,:]
            for ispec in range(nspec):
                if ispec == 0: 
                    ylab = ylabel
                else:
                    ylab = ""
                z_max = np.amax(fluxes_by_mode[:,iflux,-1,ispec,:,:])
                z_min = np.amin(fluxes_by_mode[:,iflux,-1,ispec,:,:])
                for ifield in range(nfields):
                    if ifield == nfields-1:
                        xlab = xlabel
                    else:
                        xlab = ""
                    field_name = fk_labels[fields_present[ifield]]

                    ax = axes[ifield, ispec]
                    z = fluxes_by_mode[ifield, iflux, -1, ispec, :,:]
                    title = r'{} {} {}'.format(flux_name, spec_names[ispec], field_name)
                    if ifield == nfields-1:
                        corners = ax.get_position().corners()
                        cbar_pos = [corners[0,0], -0.05, corners[2,0] - corners[0,0], 0.05]
                    else:
                        cbar_pos = None
                    plot_2d(np.transpose(z),kx,ky,z_min,z_max,ax,xlab,ylab,title,cmap,cbar_pos = cbar_pos)
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

    #print()
    #print('producing plots of fluxes vs (vpa,theta)...',end='')

    #write_vpathetasym = False
    #tmp_pdf_id = 1
    #pdflist = []
    #if es_pflx_vpth_tavg is not None and mygrids.vpa is not None:
    #    title = '$\Gamma_{GS2}$'
    #    plot_flux_vs_vpth(mygrids,es_pflx_vpth_tavg,title)
    #    write_vpathetasym = True
    #    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    #    gplot.save_plot(tmp_pdfname, run, ifile)
    #    pdflist.append(tmp_pdfname)
    #    tmp_pdf_id = tmp_pdf_id+1
    #if es_qflx_vpth_tavg is not None and mygrids.vpa is not None:
    #    title = '$Q_{GS2}$'
    #    plot_flux_vs_vpth(mygrids,es_qflx_vpth_tavg,title)
    #    write_vpathetasym = True
    #    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    #    gplot.save_plot(tmp_pdfname, run, ifile)
    #    pdflist.append(tmp_pdfname)
    #    tmp_pdf_id = tmp_pdf_id+1
    #if es_vflx_vpth_tavg is not None and mygrids.vpa is not None:
    #    title = '$\Pi_{GS2}$'
    #    plot_flux_vs_vpth(mygrids,es_vflx_vpth_tavg,title)
    #    write_vpathetasym = True
    #    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    #    gplot.save_plot(tmp_pdfname, run, ifile)
    #    pdflist.append(tmp_pdfname)
    #    tmp_pdf_id = tmp_pdf_id+1

    #if write_vpathetasym:  
    #    merged_pdfname = 'fluxes_vs_vpa_theta'
    #    gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

    #print('complete')
    #print(len(phi2_avg))



import math
def str_with_err(value, error):
    power = int
    digits = -int(math.floor(math.log10(error)))
    return "{0:.{2}f}({1:.0f})".format(value, error*10**digits, digits)

def str_with_err(x, dx):
    dx = abs(dx)
    
    # Fill sign string
    sign = round(x/abs(x))
    if sign == -1:
        x = abs(x)
        sign = "-"
    else:
        sign = ""
    
    xpow = math.floor(math.log10(x))

    # Only use scientific notation if we pass some threshold.
    if xpow > 2 or xpow < -2:
        post_add = r" $\times$ $10^{{{}}}$".format(xpow)
        x = x/(10**xpow)
        dx = dx/(10**xpow)
    else:
        post_add = ""

    trail_digits = -int(math.floor(math.log10(dx)))
    lead_digits = 0
    if trail_digits<0:
        lead_digits = -trail_digits-1
        trail_digits = 0

    return sign + "{0:{2}.{3}f}({1:.0f})".format(x, dx*10**max(lead_digits,trail_digits), lead_digits, trail_digits) + post_add


def plot_flux_vs_t(spec_labels,mytime,flx,title,fields_present):
    fig=plt.figure(figsize=(12,8))
    ax = fig.gca()
    
    # plot time-traces for each species
    nfields = len(fields_present)
    nspec = len(spec_labels)
    colors = gplot.truncate_colormap('nipy_spectral', 0.0, 0.9, nfields*nspec)
    
    for ifield in range(nfields):
        for ispec in range(nspec):
            color=colors(ifield*nspec + ispec)
            flxav = flx[ifield,-1,ispec]
            stddev = np.sqrt(np.sum(np.power(flx[ifield,mytime.it_min:,ispec] - flxav,2))/(mytime.ntime_steady-1))
            label = r'{} ({}): {}'.format(spec_labels[ispec], f_labels[fields_present[ifield]], str_with_err(flxav, stddev) )
            gplot.plot_1d(mytime.time, flx[ifield,:-1,ispec], label=label, color=color, axes=ax)
            gplot.plot_1d([mytime.time[mytime.it_min],mytime.time[mytime.it_max]], [flxav,flxav], linestyle=':',axes=ax, color=color, linewidth=2)
            ax.fill_between(mytime.time[mytime.it_min:mytime.it_max], flxav+stddev, flxav-stddev,color=color, alpha=0.25)
    plt.xlabel('$t (a/v_t)$')
    plt.xlim([mytime.time[0],mytime.time[mytime.it_max-1]])
    plt.ylim( bottom = 1.1*np.amin(flx[:,1:-1,:]), top = 1.1*np.amax(flx[:,1:-1,:])   )  
    plt.title(title)
    ax.legend()
    plt.grid(True)

    return fig

def plot_flux_vs_kxky(ispec,spec_names,kx,ky,flx,title,has_flowshear, axes=None):

    from gs2_plotting import plot_2d
    if has_flowshear:
        xlab = '$\\bar{k}_{x}\\rho_i$'
    else:
        xlab = '$k_{x}\\rho_i$'
    ylab = '$k_{y}\\rho_i$'

    cmap = 'Blues' # 'Reds','Blues'
    z = flx[ispec,:,:] # OB 140918 ~ Don't take absolute value of fluxes. 
    z_min, z_max = 0.0, z.max()
    
    if ispec > 1:
        title += ' (impurity ' + str(ispec-1) + ')'
    else:
        title += ' (' + spec_names[ispec] + 's)'
    fig = plot_2d(np.transpose(z),kx,ky,z_min,z_max,axes,xlab,ylab,title,cmap)
    return fig

def plot_field_vs_kxky(kx,ky,field,has_flowshear, title, z_min = None, z_max = None, jtwist = None):
    from gs2_plotting import plot_2d
    from gs2_plotting import plot_1d
    cmap_title = title + r' $\forall$ $k_y > 0$'
    zonal_title = title + r' for $k_y=0$'
    if has_flowshear:
        xlab = r'$\bar{k}_{x}\rho_i$'
    else:
        xlab = r'$k_{x}\rho_i$'
    ylab = r'$k_{y}\rho_i$'

    cmap = 'RdBu'# 'RdBu_r','Blues'
    
    z = field[1:,:] # taking out zonal modes because they are much larger
    if z_min == None:
        z_min = z.min()
    if z_max == None:
        z_max = z.max()
    use_logcolor = True

    zonal = field[0,:]

    fig, axes = plt.subplots(nrows = 2, sharex = True, gridspec_kw={ 'height_ratios': [2, 1], 'hspace':0.1 }, figsize = (16,16))
    plot_2d(np.transpose(z),kx,ky[1:],z_min,z_max, axes[0], "",ylab,cmap_title,cmap,use_logcolor)
    plot_1d(kx,zonal,xlab,axes[1],'', linewidth=2, log='y', label ='zonal field', linestyle=':', color='green')
    plot_1d(kx,np.power(kx,2)*zonal,xlab,axes[1],'',linewidth=2,label = 'zonal flow', color='blue')
    plot_1d(kx,np.power(kx,4)*zonal,xlab,axes[1],'',ylab=zonal_title, linewidth=2, label ='zonal shear', linestyle='--', color='red')
    axes[1].legend(loc='best')
    if jtwist is not None:
        ikxmid = np.where(kx==0)[0][0]
        i = 0
        while i < len(kx) - ikxmid:
            axes[1].axvline(kx[ikxmid + i], linestyle=':', color = '#999999')
            axes[1].axvline(kx[ikxmid - i], linestyle=':', color = '#999999')
            i += jtwist

    return fig

def plot_flux_vs_vpth(mygrids,flx,title):

    from gs2_plotting import plot_2d

    xlab = '$\\theta$'
    ylab = '$v_{\parallel}$'
    cmap = 'RdBu'
    for idx in range(flx.shape[0]):
        z = flx[idx,:,:]
        z_min, z_max = z.min(), z.max()
        fig = plot_2d(z,mygrids.theta,mygrids.vpa,z_min,z_max,None, xlab,ylab,title+' (is= '+str(idx+1)+')',cmap)

    return fig

def compare_time_traces(run):
    import gs2_plotting as gplot
    Nfile = len(run.fnames)
    all_fluxes = [dict() for ifile in range(Nfile)]
    all_time = [dict() for ifile in range(Nfile)]
    all_grids = [dict() for ifile in range(Nfile)]

    for ifile in range(Nfile):
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.fluxes.dat'
        with open(datfile_name,'rb') as datfile:
            all_fluxes[ifile] = pickle.load(datfile)
    
    tmp_pdf_id = 1
    pdflist = []
    
    plt.figure(figsize=(12,8))
    ax = plt.gca()

    for ifile in range(Nfile):
        fname = run.fnames[ifile]
        phit = all_fluxes[ifile]['phi2_avg']
        t = all_time[ifile].time
        gplot.plot_1d(t,phit,xlab='t', ylab='phi2', axes = ax)
    
    ifile=None 
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    #gplot.save_plot(tmp_pdfname, run, ifile)
    gplot.save_plot('compare_phi2_t', run, ifile)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id = tmp_pdf_id+1
     

# Method to compare the y-value of a range of linear scans. y is an array of arrays of different possible parameters, such as real frequency, growth rate, etc.
# Each array of parameters is plotted on the same figure. 
# E.g. if jobs = [['phitheta','gds2'],['gamma'],['omega']], we will have 3 figures:
#   - one containing phi and gds2 as functions of theta
#   - one containing growth rates as functions of ky
#   - one containing real frequency as functions of ky
def compare_fluxes(run):
    # Only execute if plotting
    if run.no_plot:
        return
    
    max_nspec = 1
    fluxes_present = False
    nfluxes = 3

    # Create list of colors
    Nfile = len(run.fnames)
    full_fluxes = [dict() for ifile in range(Nfile)]
    for ifile in range(Nfile):
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.fluxes.dat'
        with open(datfile_name,'rb') as datfile:
            full_fluxes[ifile] = pickle.load(datfile)
            nspec = full_fluxes[ifile]['input_file']['species_knobs']['nspec']
            max_nspec = max(nspec, max_nspec)
            fluxes_present = fluxes_present or full_fluxes[ifile]['fluxes'] is not None
    cmap = plt.get_cmap('nipy_spectral')
    colors = [cmap(i) for i in np.linspace(0,1,Nfile+1)]

    # 3 subplots per flux figure: Ion flux time trace, electron flux time trace, ion-electron flux time trace.
    
    tmp_pdf_id = 1
    pdflist = []
    merged_pdfname = ''
    """ 
    mydict = {
    'fields_present':fields_present,
    'fluxes':fluxes, 
    'fields_t_present':fields_t_present,
    'fields_t':fields_t,
    'fluxes_by_mode':fluxes_by_mode, 
    'fields2':fields2,
    'fields2_by_mode':fields2_by_mode,
    'fields2_by_ky':fields2_by_ky,
    'fields2_by_kx':fields2_by_kx,
    'pioq':pioq,
    'nl_term_by_mode':nl_term_by_mode,
    'kx':kx,
    'nakx':nakx,
    'ky':ky,
    'naky':naky,
    'theta':theta,
    'ntheta':ntheta,
    'time':mytime,
    'full_kperp':full_kperp,
    'full_drift':full_drift,
    'input_file':myin
    }  
    """

    if not fluxes_present:
        print("No fluxes to plot. Quitting.")
        quit()
    xlab = r'$t$ $(a/v_{th,\rm{ref}})$'
    
    for iflux in range(nfluxes):
        fig, axes = plt.subplots( ncols = min(3, max_nspec), figsize=(20,7) )
        if type(axes).__name__ not in ['ndarray', 'list']:
            axes = [axes]
        for ifile in range(Nfile):
            flx_i = None
            flx_e = None
            flx_ratio = None
            myin = full_fluxes[ifile]['input_file']
            mytime = full_fluxes[ifile]['time']
            itmin = mytime.it_min
            nspec = myin['species_knobs']['nspec']
            for ispec in range(nspec):
                flx = np.sum(full_fluxes[ifile]['fluxes'][:, iflux, :, ispec], axis=0)
                if myin['species_parameters_{}'.format(ispec+1)]['type'][0] == 'e':
                    if flx_e is None:
                        flx_e = flx
                    else:
                        flx_e = np.add(flx_e, flx)
                else:
                    if flx_i is None:
                        flx_i = flx
                    else:
                        flx_i = np.add(flx_i, flx)
            if flx_i is None and flx_e is not None:
                title = 'Ion flux'
            elif flx_i is not None and flx_e is None:
                title = 'Electron flux'
            elif flx_i is None and flx_e is None:
                print('Something went wrong! #001')
                quit()
            else:
                print('multiue spec')
                flx_ratio = np.divide(flx_i, flx_e)
            t = mytime.time[itmin:]
            flab = flux_labels[iflux]
            llabel = run.fnames[ifile].replace("_", "\_")

            #TODO Does not handle cases where 1 species and some parts of scan are ions and some are electrons ONLY (max_nspec=1)
            gplot.plot_1d(t,flx_i[itmin:-1], xlab, axes=axes[0], title='', ylab = r'{}$_{{,i}}$'.format(flab), label=llabel, color=colors[ifile], grid='both')
            gplot.plot_1d(t,flx_e[itmin:-1], xlab, axes=axes[1], title='', ylab = r'{}$_{{,e}}$'.format(flab), color=colors[ifile], grid='both')
            gplot.plot_1d(t,flx_ratio[itmin:-1], xlab, axes=axes[2], title='', ylab = r'{}$_{{,i}}$/{}$_{{,e}}$'.format(flab,flab), color=colors[ifile], grid='both')
                
            if run.files[0] == "outer" and iflux==2:
                fp = FontProperties(size='small')
                for i in range(3):
                    q = [0.03923, 0.02170, 0.03923/0.02170][i]
                    axes[i].axhline(q, ls='--', color='gray')
                    axes[i].text(0.1, q*1.01, "Exp", transform=axes[i].get_yaxis_transform(), color='gray', fontproperties=fp)

        handles,labels = axes[0].get_legend_handles_labels()
        for ax in axes:
            ax.ticklabel_format(axis='y', style='sci', scilimits = (-3,3))
        fig.legend(handles,labels, ncol = min(Nfile,5), loc='upper left', bbox_to_anchor=(0.0, 0.15))

            
        ifile = None
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

    gplot.merge_pdfs(pdflist, 'fluxes_compare', run, ifile)



def compare_flux_tavg(run): 
    import gs2_plotting as gplot
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    # Only executed if we want to plot the data
    Nfile = len(run.fnames)
    all_fluxes = [dict() for ifile in range(Nfile)]
    all_balloon = [dict() for ifile in range(Nfile)]
    all_time = [dict() for ifile in range(Nfile)]
    all_grids = [dict() for ifile in range(Nfile)]

    for ifile in range(Nfile):
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.fluxes.dat'
        with open(datfile_name,'rb') as datfile:
            all_fluxes[ifile] = pickle.load(datfile)
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.time.dat'
        with open(datfile_name,'rb') as datfile:
            all_time[ifile]  =  pickle.load(datfile)
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.grids.dat'
        with open(datfile_name,'rb') as datfile:
            all_grids[ifile] = pickle.load(datfile)
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.boxballoon.dat'
        with open(datfile_name,'rb') as datfile:
            all_balloon[ifile] = pickle.load(datfile)

    # Connectivity at ky=1.0 will be x axis, y0 will be shown by marker type and colour, y-axis will be Qflx-tavg.
    # Finite number of markers. Could change to colormap if necessary.
    tmp_pdf_id = 1
    pdflist = []
    
   # Get resolution parameters
    data = []
    connects = []
    plt.figure(figsize=(12,8))
    for ifile in range(Nfile):
        fname = run.fnames[ifile]
        rundir = run.dirs[ifile]
        kymax = float(rundir[rundir.find('kymax')+5:rundir.find('kymax')+8])
        data.append([ all_fluxes[ifile]['y0'], kymax*all_fluxes[ifile]['nx']/( all_fluxes[ifile]['ny'] * all_fluxes[ifile]['jtwist']), all_fluxes[ifile]['es_qflx_tavg'][0], 'hv_x' in fname, 'hv_y' in fname, 'hv_n' in fname, 'col' in fname, kymax])
    data = np.array(data)
    data = data[(-data[:,0]).argsort()] 
    sety0 = sorted(list(set(data[:,0])))
    colors = gplot.truncate_colormap('nipy_spectral', 0.0, 0.9, len(sety0))
    colorsused = np.zeros(len(sety0))
    legend_elements = []
    
    for i in range(Nfile):
        y0 = data[i,0]
        C = data[i,1]
        Q = data[i,2]
        hvx = data[i,3]
        hvy = data[i,4]
        hv = data[i,5]
        col = data[i,6]
        kymax = data[i,7]
        marker_style = dict(markeredgecolor=colors(sety0.index(y0)), color = colors(sety0.index(y0)), marker='s', fillstyle='full', markerfacecoloralt='none', markeredgewidth = kymax)
        
        if hvx:
            marker_style['fillstyle'] = 'left'
        if hvy:
            marker_style['fillstyle'] = 'top'
        if hv:
            marker_style['fillstyle'] = 'none'
        if col:
            marker_style['marker'] = 'v'
        
        if colorsused[sety0.index(y0)] == 0 and not hv and not col: 
            legend_elements.append( Patch(facecolor=colors(sety0.index(y0)), label='y0 = {:1.1f}'.format(y0)) )
            colorsused[sety0.index(y0)] = 1
        plt.plot(C,Q, linestyle='none', **marker_style)
    plt.grid()
    plt.xlabel('Connectivity at $k_y=1.0$')
    plt.ylabel('$Q/Q_{GB}$')
    legend_elements.append(  Line2D([0], [0], color='gray', label='$k_{y,\\rm{max}}=1.00$', linewidth=1.00) )  
    legend_elements.append(  Line2D([0], [0], color='gray', label='$k_{y,\\rm{max}}=2.00$',  linewidth=2.00) )  
    legend_elements.append(  Line2D([0], [0], marker='s', color='gray', label='No HV or COL', markerfacecolor='gray', markersize=10, linestyle='none') )  
    legend_elements.append(  Line2D([0], [0], marker='s', color='gray', label='HV in x and y, no COL', markerfacecolor='none', markersize=10, linestyle='none') )  
    legend_elements.append(  Line2D([0], [0], marker='s', color='gray', label='HV in x, no COL', markerfacecoloralt='none', markerfacecolor='gray', fillstyle='left' , markersize=10, linestyle='none') )  
    legend_elements.append(  Line2D([0], [0], marker='s', color='gray', label='HV in y, no COL', markerfacecoloralt='none', markerfacecolor='gray', fillstyle='top' , markersize=10, linestyle='none') )  
    legend_elements.append(  Line2D([0], [0], marker='v', color='gray', label='COL, no HV', markerfacecolor='gray', markersize=10, linestyle='none') )  
    legend_elements.append(  Line2D([0], [0], marker='v', color='gray', label='COL + HV', markerfacecolor='none', markersize=10, linestyle='none') )  
    plt.legend(ncol=2, handles = legend_elements, handlelength=1)
    ifile = None
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run, ifile)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id = tmp_pdf_id+1
     
    # Now check whether potential has dropped at the edge of the ballooning chain.
         
    plt.figure(figsize=(12,8))  
    
    data = []
        

    for ifile in range(Nfile):
        ky = all_balloon[ifile]['ky']
        bloonphi2_thavg_tavg = all_balloon[ifile]['bloonphi2_thavg_tavg']
        bloonkxs = all_balloon[ifile]['bloonkxs']
        ratios = np.zeros(len(ky)-1)
        for iky in range(1,len(ky)):
            phi2chains = np.array(bloonphi2_thavg_tavg[iky-1])        # [itheta0, iikx]
            chainphi2ratios = np.zeros(len(phi2chains))
            for itheta0 in range(len(phi2chains)):
                midavphi2 = phi2chains[itheta0][(len(phi2chains[itheta0])-1)//2]
                Lavphi2 = phi2chains[itheta0][0]
                Ravphi2 = phi2chains[itheta0][-1]
                if Lavphi2 == midavphi2 and Ravphi2 == midavphi2:   # Chain length 1
                    endavphi2 = midavphi2
                elif Lavphi2 == midavphi2:  # Chain starts at far left
                    endavphi2 = Ravphi2
                elif Ravphi2 == midavphi2:  # Chain starts at far right
                    endavphi2 = Lavphi2
                else:                       # Chain continues to left and right.
                    endavphi2 = (Lavphi2+Ravphi2)/2.0
                chainphi2ratios[itheta0] = endavphi2/midavphi2
            ratios[iky-1] = np.mean(chainphi2ratios)
        fname = run.fnames[ifile]
        rundir = run.dirs[ifile]

        data.append([ all_fluxes[ifile]['y0'], np.mean(ratios), all_fluxes[ifile]['es_qflx_tavg'][0], 'hv' in fname, 'col' in fname, float(rundir[rundir.find('kymax')+5:rundir.find('kymax')+8])  ])
    
    # Data sort for formatting purposes.
    data = np.array(data)
    data = data[data[:,0].argsort()] 
    sety0 = sorted(list(set(data[:,0])))
    colors = gplot.truncate_colormap('nipy_spectral', 0.0, 0.9, len(sety0))
    colorsused = np.zeros(len(sety0))
   
    for i in range(Nfile):
        y0 = data[i,0]
        R = data[i,1]
        Q = data[i,2]
        hv = data[i,3]
        col = data[i,4]
        kymax = data[i,5]
        marker_style = dict(markeredgecolor=colors(sety0.index(y0)), color = colors(sety0.index(y0)), marker='s', fillstyle='full', markeredgewidth = kymax)
        
        if hv:
            marker_style['fillstyle'] = 'none'
        if col:
            marker_style['marker'] = 'v'
        
        if colorsused[sety0.index(y0)] == 0 and not hv and not col: 
            plt.plot(R,Q, linestyle='none', label = 'y0 = {:1.1f}'.format(y0), **marker_style)
            colorsused[sety0.index(y0)] = 1
        else:
            plt.plot(R,Q, linestyle='none', **marker_style)
       
    plt.grid()
    plt.xlabel('$\\langle\\langle|\\delta\\phi|^2_{\\rm{end}}\\rangle_{t,\\theta} / \\langle|\\delta\\phi|^2_{\\rm{mid}}\\rangle_{t,\\theta}\\rangle_{k_y, \\theta_0}$')
    plt.ylabel('$Q/Q_{GB}$')
    plt.legend()
    ifile = None

    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run, ifile)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id = tmp_pdf_id+1
     
    merged_pdfname = 'res_scan'
    gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

# Method to compare the y-value of a range of nonlinear simulations. y is an array of arrays of different possible parameters, such as phi2kykx, qflx, etc.
# Each array of parameters is plotted on the same figure. 
# E.g. if jobs = [['phitheta','gds2'],['gamma'],['omega']], we will have 3 figures:
#   - one containing phi and gds2 as functions of theta
#   - one containing growth rates as functions of ky
#   - one containing real frequency as functions of ky
def compare(run, y):
    # Only execute if plotting
    from gs2_plotting import plot_2d
    if run.no_plot:
        return
    
    # Linestyles for multiple plots (limit to 4).
    style = ['-', '--', '-.', ':', 'LIMITED TO 4']

    # Add new y-parameter options here, and what they correspond to in x-axis.
    ykeytodat = {
        'phi2kxky':'phi2_kxky_tavg', 
        'apar2kxky':'apar2_kxky_tavg' ,
        'bpar2kxky':'bpar2_kxky_tavg' 
                }
    ytitle = {
        'phi2kxky':r'$\langle\vert\hat{\varphi}_k\vert ^2\rangle_{t,\theta}$',
        'apar2kxky':r'$\langle\vert\hat{A_\parallel}_k\vert ^2\rangle_{t,\theta}$',
        'bpar2kxky':r'$\langle\vert\hat{B_\parallel}_k\vert ^2\rangle_{t,\theta}$'
            }
    yx = {
        'phi2kxky':'2d_kx_ky',
        'apar2kxky':'2d_kx_ky',
        'bpar2kxky':'2d_kx_ky'
        }
    
    # Avoid duplicates in commonly used x-titles (ky, theta, kx)
    xtitle = {'theta':r'$\theta$',
        'ky':r'$k_y\rho_{\rm{ref}}$',
        'kx':r'$k_x\rho_{\rm{ref}}$'}
    # Create list of colors
    Nfile = len(run.fnames)
    full_fluxes = [dict() for ifile in range(Nfile)]
    for ifile in range(Nfile):
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.fluxes.dat'
        with open(datfile_name,'rb') as datfile:
            full_fluxes[ifile] = pickle.load(datfile)
    cmap = plt.get_cmap('nipy_spectral')
    colors = [cmap(i) for i in np.linspace(0,1,Nfile+1)]
    if Nfile > 3:
        gridcols = int(np.sqrt(Nfile)) + 1
        gridrows = gridcols
    else:
        gridcols = Nfile
        gridrows = 1
    gridscale = 8
    tmp_pdf_id = 1
    pdflist = []
    merged_pdfname = ''
    for iset in range(len(y)):
        plt.figure(figsize=(12,8))
        plt.grid(True)
        yset = y[iset]
        for yitem in yset:
            if yitem[:2] == '2d' and len(set) > 1:
                print('Cannot plot multiple quantities on a single colourmap. Quitting.')
                quit()
        if len(set([yx[yitem] for yitem in yset])) > 1:
            print('Cannot plot multiple quantities with different x-values on the same axis. Quitting.')
            quit()
        
        # Now either 1d with one or more quantities plotted against the same axis, or 2d with multiple colormaps plotting the same quantity.
        if yx[yset[0]][:2] == '2d':
            # Plotting one colormap for each run. 
            axeslabels = yx[yset[0]][3:]
            xaxis = axeslabels[:axeslabels.rfind('_')]
            yaxis = axeslabels[(axeslabels.rfind('_')+1):]
            fig, axes = plt.subplots(ncols = gridcols, nrows = gridrows, figsize = (gridscale*gridcols, gridscale*gridrows))
            zdatkey = ykeytodat[yset[0]]
            zkey = yset[0]
            zmin = 1e10
            zmax = -1e10
            for ifile in range(Nfile):
                zdat = full_fluxes[ifile][zdatkey][1:,:]
                if np.amin(zdat) < zmin:
                    zmin = np.amin(zdat)
                if np.amax(zdat) > zmax:
                    zmax = np.amax(zdat)
                print(zmin)
            for ifile in range(Nfile):
                zdat = full_fluxes[ifile][zdatkey][1:,:]
                xdat = full_fluxes[ifile][xaxis]
                ydat = full_fluxes[ifile][yaxis][1:]
                if gridrows == 1:
                    ax = axes[ifile]
                else:
                    row = ifile // gridcols
                    col = ifile % gridcols
                    ax = axes[row, col]
                im = plot_2d(np.transpose(zdat),xdat,ydat,zmin,zmax, axes = ax, xlab=xtitle[xaxis], ylab=xtitle[yaxis], title=run.fnames[ifile].replace('_', '\_'), use_logcolor=True, add_cbar = False)
            cbar_ax = fig.add_axes([0.08, -0.05, 0.92, 0.05])
            cbar_ax.set_title(ytitle[zkey], pad=-75)
            fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

        else:
            quit()
        
        # Plotting one or more variables for each run
        """
        plt.xlabel(xtitle[yx[yset[0]]])
        style_legend_lines = []
        for iy in range(len(yset)):    
            thisy = yset[iy]
            maxthisy = -1
            for ifile in range(Nfile):
                maxthisy = np.max([np.max(np.abs(full_fluxes[ifile][thisy])),maxthisy])
            for ifile in range(Nfile):
                yvals = full_fluxes[ifile][thisy]
                
                if thisy in ['gds2','gds21','gds22','gbdrift','cvdrift','drifts','kperp2']:   # Normalize to max value.
                    yvals = yvals/maxthisy
                elif thisy in ['phitheta','apartheta']:
                    yvals = yvals/np.max(np.abs(full_fluxes[ifile][thisy]))    
                xvals = full_fluxes[ifile][yx[thisy]]
                if iy == 0:
                    linelabel = label = run.fnames[ifile].replace("_", "\_")
                else:
                    linelabel = None
                if yx[thisy] in ['theta']:
                    plt.plot([val*radians for val in xvals], yvals, label = linelabel, color = colors[ifile], xunits = radians, linestyle = style[iy])
                else:
                    plt.plot(xvals, yvals, label = linelabel, color = colors[ifile], linestyle = style[iy])
            if iy == 0:
                file_legend = plt.legend( prop={'size': 11}, ncol = 4, loc=1)
                ax = plt.gca().add_artist(file_legend)
            yline, = plt.plot([], label=ytitle[thisy], linestyle=style[iy], color='gray') 
            style_legend_lines.append(yline)

            merged_pdfname = merged_pdfname + thisy + '-'
        merged_pdfname = merged_pdfname[:-1] + '_'
        # Add the legend manually to the current Axes.
                 
        # Create another legend for the second line.
        plt.legend(handles = style_legend_lines, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=len(yset), mode="expand", borderaxespad=0.) 
        """    
        ifile = None
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

    gplot.merge_pdfs(pdflist, merged_pdfname+'compare', run, ifile)


