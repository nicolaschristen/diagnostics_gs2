# Task to write/plot several quantities for linear simulations done in range mode.
import numpy as np
import scipy.optimize as opt
import matplotlib.animation as anim
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
import matplotlib.colors as mpl_colors
from scipy.special import j0
import gs2_plotting as gplot
import pickle
import gs2_plotting as gplot
from gs2_utils import radians
import imageio
import os
import sys

f_colors = {'es':'blue', 'apar':'red', 'bpar':'green', 'other':'black'}
f_labels = {'es':r'$|\phi|$', 'apar':r'$|A_\parallel|$', 'bpar':r'$|B_\parallel|$'}
flux_labels = [r'$|\Gamma_{{{}GS2}}|$',r'$|\Pi_{{{}GS2}}|$',r'$|Q_{{{}GS2}}|$']

def my_task_single(ifile, run, myin, myout):
    

    # User parameters
    dump_at_start = 0.3 # fraction of initial time to dump when fitting
    
    # Compute and save growthrate
    if not run.only_plot:

        outnperiod = 0
        
        grid_option = myin['kt_grids_knobs']['grid_option']
        
        t = myout['t']
        nt = t.size
        it_start = round(nt*dump_at_start)
        kx = myout['kx']
        theta0 = myout['theta0']
        nakx = kx.size          # Same as ntheta0
        ntheta0 = nakx
        ky = myout['ky']
        naky = ky.size
        
        fields_present = []
        fields = []
        fields2_by_mode = []
        
        phi = myout['phi']      # Potential [ky,kx,theta,ri]
        apar = myout['apar']    # Parallel magnetic potential [ky,kx,theta,ri]
        bpar = myout['bpar']    # Parallel magnetic fluctuation [ky,kx,theta,ri]

        phi2_by_mode = myout['phi2_by_mode']
        apar2_by_mode = myout['apar2_by_mode']
        bpar2_by_mode = myout['bpar2_by_mode']
        
        if phi is not None:
            fields.append(phi)
            fields2_by_mode.append(phi2_by_mode)
            fields_present.append('es')
        if apar is not None:
            fields.append(apar)
            fields2_by_mode.append(apar2_by_mode)
            fields_present.append('apar')
        if bpar is not None:
            fields.append(bpar)
            fields2_by_mode.append(bpar2_by_mode)
            fields_present.append('bpar')
        
        fields = np.asarray(fields)
        fields2_by_mode = np.asarray(fields2_by_mode)
 
        fields_t_present = []
        fields_t = []
        
        phi_t = myout['phi_t']      # Potential [ky,kx,theta,ri]
        apar_t = myout['apar_t']    # Parallel magnetic potential [ky,kx,theta,ri]
        bpar_t = myout['bpar_t']    # Parallel magnetic fluctuation [ky,kx,theta,ri]

        if phi_t is not None:
            fields_t.append(phi_t)
            fields_t_present.append('es')
        if apar_t is not None:
            fields_t.append(apar_t)
            fields_t_present.append('apar')
        if bpar_t is not None:
            fields_t.append(bpar_t)
            fields_t_present.append('bpar')
        fields_t = np.asarray(fields_t)
       
        fluxes_present = myout['es_heat_flux_present'] or myout['apar_heat_flux_present'] or myout['bpar_heat_flux_present']
        
        if fluxes_present:
            fluxes = []                 # [field, flux, t, spec]
            for field in fields_present:
                tmp_fluxes = []
                for flux_type in ['part_flux', 'mom_flux', 'heat_flux']:
                    if myout['{}_{}_present'.format(field, flux_type)]:
                        tmp_fluxes.append(myout['{}_{}'.format(field,flux_type)])
                fluxes.append(tmp_fluxes)
 
            fluxes = np.asarray(fluxes)
        else:
            fluxes = None
            
        

        if fluxes is not None:
            pioq = np.divide(fluxes[:,1,:,:],fluxes[:,2,:,:])
        else:
            pioq = None
        
        theta = myout['theta']
        it_end  = nt
        for it in range(nt):
            for ikx in range(nakx):
                for iky in range(naky):
                    if it < it_end and not np.isfinite(fields2_by_mode[0,it,iky,ikx]):
                        it_end = it - 10 
                        break
        # Fit phi2 to get growthrates
        gamma = np.zeros( (naky,nakx) )
        omega = np.zeros( (naky,nakx) )
        for iky in range(naky):
            for ith0 in range(nakx):
                gamma[iky,ith0] = get_growthrate(t,fields2_by_mode[0],it_start,it_end,ith0,iky)
                omega[iky,ith0] = myout['omega'][len(myout['omega'][:,0,0,0])-1, iky,ith0, 0]
        # Look only within Maximum ky within ITG range
        
        complexFreq = myout['omega']        # Real frequency [t, ky, kx, ri]

        gds2 = myout['gds2']
        gds21 = myout['gds21']
        gds22 = myout['gds22']
        gbdrift = myout['gbdrift']
        cvdrift = myout['cvdrift']
        gbdrift0 = myout['gbdrift0']      
        cvdrift0 = myout['cvdrift0']      
       
        ntheta = myin['theta_grid_parameters']['ntheta']
        nperiod = myin['theta_grid_parameters']['nperiod']

        fullkperp2 = np.zeros((naky,ntheta0,len(theta) ))
        fulldrifts = np.zeros((naky,ntheta0,len(theta) ))
        for iky in range(naky):
            for itheta0 in range(ntheta0):
                for itheta in range(len(theta)):
                    fullkperp2[iky,itheta0,itheta] = np.power(ky[iky],2) * ( gds2[itheta] + 2*theta0[iky,itheta0]*gds21[itheta] + np.power(theta0[iky,itheta0],2) * gds22[itheta] ) 
                    fulldrifts[iky,itheta0,itheta] = ky[iky] * (gbdrift[itheta] + cvdrift[itheta] + theta0[iky,itheta0]*(gbdrift0[itheta] + cvdrift0[itheta]))
        fullj0 = j0(np.sqrt(fullkperp2))

        if outnperiod in [nperiod,0]:
            outthetaslice = slice(None,None)    # Slice is whole array
        else:
            midthetaindex = len(theta)//2
            outthetaslice = slice(midthetaindex - (2*outnperiod-1)*ntheta//2, midthetaindex + (2*outnperiod-1)*ntheta//2)

        try:
            zeff = myin['parameters']['zeff']
        except:
            zeff = 1.0
        beta = myin['parameters']['beta']
        bprim = myin['theta_grid_eik_knobs']['beta_prime_input']
        
        specparams = []
        nspec = myin['species_knobs']['nspec']
        elecindex = -1
        refmass = -1
        refcharge = 0

        fluxes_e = None
        fluxes_i = None
        fluxes_ratio = None

        for ispec in range(nspec):
            specstring = 'species_parameters_{:d}'.format(ispec+1)
            specparams.append(myin[specstring])
            if myin[specstring]['type'] == 'electron': 
                if elecindex == -1:
                    elecindex = ispec
                    refmass = round(1.0/(myin[specstring]['mass']*1833.0))  # Reference mass in amu.
                    refcharge = 1                                           # Assume reference charge is +1. If not true then check z for electron species.
                    # Sum fluxes over species (ion vs electron), and take the ratio of ion-electron. 
                    if fluxes is not None:
                        fluxes_e = fluxes[:,:,:,ispec]
                        #if nspec > 1:
                        #    fluxes_i = np.sum(fluxes - fluxes_e[:,:,:,None],axis=3)        
                else:
                    quit('Too many electron species in species params')
            elif myin[specstring]['type']=='ion' and fluxes is not None:
                if fluxes_i is None:
                    fluxes_i = fluxes[:,:,:,ispec]
                else:
                    fluxes_i += fluxes[:,:,:,ispec]
        if fluxes is not None:
            if fluxes_e is not None and fluxes_i is not None:
                #fluxes_i = np.abs(fluxes_i)
                #fluxes_e = np.abs(fluxes_e)
                fluxes_ratio = fluxes_i/fluxes_e
            #elif fluxes_e is None:
                #fluxes_i = np.abs(np.sum(fluxes,axis=3))
            #fluxes = np.abs(fluxes)
        # Work out a species label based on electron mass, charge and z
        for ispec in range(nspec):
            if ispec == elecindex:
                specparams[ispec]['label'] = '$e^-$'
            else:
                specparams[ispec]['label'] = '{}{:d}+$'.format(round(specparams[ispec]['mass']), round(specparams[ispec]['z']*refcharge))

        fapar = myin['knobs']['fapar']
        try:
            fbpar = myin['knobs']['fbpar']
        except:
            fbpar = myin['knobs']['faperp']

        if bprim == 0 and beta == 0:
            pprim = 0
        else: 
            pprim = -bprim / beta
        mydict = {
            'myin':myin,
            'rhoc':myin['theta_grid_parameters']['rhoc'], 
            't':t, 
            'theta':theta[outthetaslice],  
            'theta0':theta0, 
            'kx':kx,
            'ky':ky,
            'gamma':gamma,
            'omega':omega,
            'complexFreq':complexFreq,
            'tri':myin['theta_grid_parameters']['tri'], 
            'kap':myin['theta_grid_parameters']['akappa'],
            'shat':myin['theta_grid_parameters']['shat'],
            'bprim':bprim, 
            'zeff':zeff,
            'beta':beta, 
            'pprim':pprim,
            'fields_present':fields_present,
            'fluxes':fluxes,
            'fluxes_i':fluxes_i,
            'fluxes_e':fluxes_e,
            'fluxes_ratio':fluxes_ratio,
            'pioq':pioq,
            'fields':fields,
            'fields2_by_mode':fields2_by_mode,
            'fields_t_present':fields_t_present,
            'fields_t':fields_t,
            'specparams':specparams, 
            'fullkperp2':fullkperp2, 
            'fulldrifts':fulldrifts, 
            'fullj0':fullj0,
            'fapar':fapar, 
            'fbpar':fbpar, 
            'ntheta':myin['theta_grid_parameters']['ntheta'], 
            'negrid':myin['le_grids_knobs']['negrid'], 
            'ngauss':myin['le_grids_knobs']['ngauss'], 
            'vcut':myin['le_grids_knobs']['vcut'],
            'input_file':myin
        }
        
        # Save to .dat file 
        # OB 170918 ~ Changed output to a dict and added kappa, tri
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.lingrowth.dat'
        with open(datfile_name, 'wb') as datfile:
            pickle.dump(mydict,datfile)
        
    # or read from .dat file
    else:
        
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.lingrowth.dat'
        with open(datfile_name, 'rb') as datfile:
            mydict = pickle.load(datfile)

    if not run.no_plot:
        plot(ifile, run, mydict)
        if run.make_movies and len(mydict['fields_t'])>0:
            movie_fields_theta(ifile, run, mydict)

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
    nfluxes = 3

    # Create list of colors
    Nfile = len(run.fnames)
    full_lingrowth = [dict() for ifile in range(Nfile)]
    for ifile in range(Nfile):
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.lingrowth.dat'
        with open(datfile_name,'rb') as datfile:
            full_lingrowth[ifile] = pickle.load(datfile)
            nspec = full_lingrowth[ifile]['input_file']['species_knobs']['nspec']
            max_nspec = max(nspec, max_nspec)
            fluxes_present = full_lingrowth[ifile]['fluxes'] is not None
    cmap = plt.get_cmap('nipy_spectral')
    colors = [cmap(i) for i in np.linspace(0,1,Nfile+1)]

    # 3 subplots per flux figure: Ion flux time trace, electron flux time trace, ion-electron flux time trace.
    
    tmp_pdf_id = 1
    pdflist = []
    merged_pdfname = ''
    
    if not fluxes_present:
        print("No fluxes to plot. Quitting.")
        quit()
    xlab = r'$t$ $(a/v_{th,\rm{ref}})$'
    for iflux in range(nfluxes):
        fig, axes = plt.subplots( ncols = 3, figsize=(20,7) )
        if type(axes).__name__ not in ['ndarray', 'list']:
            axes = [axes]
        for ifile in range(Nfile):
            if full_lingrowth[ifile]['fluxes_ratio'] is None:
                quit("Doesn't support single species yet")
            nfields = len(full_lingrowth[ifile]['fields_present'])
            flx_i = None
            flx_e = None
            flx_ratio = None
            myin = full_lingrowth[ifile]['input_file']
            nspec = myin['species_knobs']['nspec']
            flxi = full_lingrowth[ifile]['fluxes_i'][:, iflux, :]
            flxe = full_lingrowth[ifile]['fluxes_e'][:, iflux, :]
            flxr = full_lingrowth[ifile]['fluxes_ratio'][:, iflux, :]
            time = full_lingrowth[ifile]['t']
            itmin = len(time)//5
            t = time[itmin:]
            flab = flux_labels[iflux]
            llabel = run.fnames[ifile].replace("_", "\_")
            for ifield in range(nfields):
                    
                #TODO Does not handle cases where 1 species and some parts of scan are ions and some are electrons ONLY (max_nspec=1)
                gplot.plot_1d(t,flxi[ifield, itmin:], xlab, axes=axes[0], title='', ylab = r'{}$_{{,i}}$'.format(flab), label=llabel, color=colors[ifile], grid='both', log='y')
                gplot.plot_1d(t,flxe[ifield, itmin:], xlab, axes=axes[1], title='', ylab = r'{}$_{{,e}}$'.format(flab), color=colors[ifile], grid='both', log='y')
                gplot.plot_1d(t,flxr[ifield, itmin:], xlab, axes=axes[2], title='', ylab = r'{}$_{{,i}}$/{}$_{{,e}}$'.format(flab,flab), color=colors[ifile], grid='both')

        handles,labels = axes[0].get_legend_handles_labels()
        #for ax in axes:
        #    ax.ticklabel_format(axis='y', style='sci', scilimits = (-3,3))
        fig.legend(handles,labels, ncol = min(Nfile,5), loc='upper left', bbox_to_anchor=(0.0, 0.15))

            
        ifile = None
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

    gplot.merge_pdfs(pdflist, 'fluxes_compare', run, ifile)


# Method to compare the y-value of a range of linear scans. y is an array of arrays of different possible parameters, such as real frequency, growth rate, etc.
# Each array of parameters is plotted on the same figure. 
# E.g. if jobs = [['phitheta','gds2'],['gamma'],['omega']], we will have 3 figures:
#   - one containing phi and gds2 as functions of theta
#   - one containing growth rates as functions of ky
#   - one containing real frequency as functions of ky

def compare(run, y):
    # Only execute if plotting
    if run.no_plot:
        return
    
    # Linestyles for multiple plots (limit to 4).
    style = ['-', '--', '-.', ':', 'LIMITED TO 4']

    # Add new y-parameter options here, and what they correspond to in x-axis.
    ytitle = {'phitheta':'$|\\delta\\phi| \\hspace{1mm}\\rm{ at } \\hspace{1mm}{k_y^{\\gamma_{max}}} $',
        'apartheta':'$|\\delta A_\\parallel| \hspace{1mm} \\rm{ at } \\hspace{1mm} {k_y^{\\gamma_{max}}} $',
        'phithetaoverapartheta':'$|\\delta\\phi|/|A_\\parallel| \\hspace{1mm}\\rm{ at }\\hspace{1mm} {k_y^{F\\gamma M}}$',
        'omega':'$\\omega a_{\\rm{ref}}/v_{\\rm{th,ref}}$',
        'gamma':'$\\gamma a_{\\rm{ref}}/v_{\\rm{th,ref}}$',
        'kperp2':'$k_\perp^2$',
        'drifts':'$k_\perp\cdot v_{D}$'}
    yx = {'phitheta':'theta',
        'apartheta':'theta',
        'phithetaoverapartheta':'theta',
        'omega':'ky',
        'gamma':'ky',
        'drifts':'theta',
        'kperp2':'theta'}
    
    # Avoid duplicates in commonly used x-titles (ky, theta, kx)
    xtitle = {'theta':'$\\theta$',
        'ky':'$k_y\\rho_{\\rm{ref}}$'}
    # Create list of colors
    Nfile = len(run.fnames)
    full_lingrowth = [dict() for ifile in range(Nfile)]
    for ifile in range(Nfile):
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.lingrowth.dat'
        with open(datfile_name,'rb') as datfile:
            full_lingrowth[ifile] = pickle.load(datfile)
    cmap = plt.get_cmap('nipy_spectral')
    colors = [cmap(i) for i in np.linspace(0,1,Nfile+1)]
    
    tmp_pdf_id = 1
    pdflist = []
    merged_pdfname = ''
    for iset in range(len(y)):
        plt.figure(figsize=(12,8))
        plt.grid(True)
        yset = y[iset]
        if len(set([yx[yitem] for yitem in yset])) > 1:
            print('Cannot plot multiple quantities with different x-values on the same axis. Quitting.')
            quit()
        plt.xlabel(xtitle[yx[yset[0]]])
        style_legend_lines = []
        for iy in range(len(yset)):    
            thisy = yset[iy]
            maxthisy = -1
            for ifile in range(Nfile):
                maxthisy = np.max([np.max(np.abs(full_lingrowth[ifile][thisy])),maxthisy])
            for ifile in range(Nfile):
                yvals = full_lingrowth[ifile][thisy]
                
                if thisy in ['drifts','kperp2']:   # Normalize to max value.
                    yvals = yvals/maxthisy
                elif thisy in ['phitheta','apartheta']:
                    yvals = yvals/np.max(np.abs(full_lingrowth[ifile][thisy]))    
                xvals = full_lingrowth[ifile][yx[thisy]]
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
        ifile = None
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

    gplot.merge_pdfs(pdflist, merged_pdfname+'compare', run, ifile)

def plot(ifile, run, mydict):

    t = mydict['t']
    nt = len(t)
    dump_t = 0.35 
    it_min = int(nt*dump_t)
    omega = mydict['omega']
    gamma = mydict['gamma']
    complexFreq = mydict['complexFreq']
    kx = mydict['kx']
    ikx0 = np.abs(kx - 0.0).argmin() 
    theta = mydict['theta']

    theta0 = mydict['theta0']
    ky = mydict['ky']
    shat = mydict['shat']

    ntheta0 = len(kx)
    naky = len(ky)
    fields = mydict['fields']
    fields_present = mydict['fields_present']
    fieldskykxtheta = np.sqrt(fields[:,:,:,:,0]**2 + fields[:,:,:,:,1]**2) 
    fields2_by_mode = mydict['fields2_by_mode']    

    fluxes = mydict['fluxes']
    fluxes_i = mydict['fluxes_i']
    fluxes_e = mydict['fluxes_e']
    fluxes_ratio = mydict['fluxes_ratio']

    fullkperp2 = mydict['fullkperp2']
    fullj0 = mydict['fullj0']
    fulldrifts = mydict['fulldrifts']

    print('Maximum linear growthrate: '+str(np.nanmax(gamma)))
    tmp_pdf_id = 1
    pdflist = []
    
    if complexFreq is not None and True:
        title = r'$\omega$ vs $t$ for different $k_y$'
        print('plotting omega vs t for all ky')
        gplot.plot_multi_1d(t[it_min:],np.transpose(complexFreq[it_min:,:,0,0]), '$t$', title=title, labels = ['{:1.2f}'.format(formatky) for formatky in ky])  
        plt.grid(True)
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1   
        title = r'$\gamma$ vs $t$ for different $k_y$'
        print('plotting gamma vs t for all ky')
        gplot.plot_multi_1d(t[it_min:],np.transpose(complexFreq[it_min:,:,0,1]), '$t$', title=title, labels = ['{:1.2f}'.format(formatky) for formatky in ky])  
        plt.grid(True)
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1   
    
    if fields2_by_mode is not None:
        title = r'Quantities vs $t$ for different $k_y$'
        fig, axes = plt.subplots(nrows = len(fields), figsize = (16, 6*len(fields)), gridspec_kw = {'wspace':0, 'hspace':0} )
        if type(axes).__name__ not in ['ndarray', 'list']:
            axes = np.array([axes])
        axes[0].set_title(title)
        for ifield in range(len(fields)):
            ax = axes[ifield]
            gplot.plot_multi_1d(t[it_min:],np.transpose(fields2_by_mode[ifield, it_min:,:,0]), '$t$', axes=ax, ylab=f_labels[fields_present[ifield]], labels = ['{:1.2f}'.format(fky) for fky in ky] , log='y')
        plt.grid(True)
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1   
    
    if gamma is not None:
        title=r'$\gamma$' 
        if len(kx) > 1:
            gplot.plot_2d(np.transpose(gamma),theta0[0],ky,np.amin(gamma),np.amax(gamma),None, r'$\theta_{0}$',r'$k_{y}$',title,cmp='RdBu_r')
        else:
            gplot.plot_1d(ky,gamma[:,ikx0],'$k_y\\rho_i$',title=title)
            plt.grid(True)
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
 
    if omega is not None:
        title = '$\\omega \\ [v_{thr}/r_r]$'
        if len(kx) > 1:
            gplot.plot_2d(np.transpose(omega),theta0[0],ky,np.amin(omega),np.amax(omega),None, r'$\theta_{0}$',r'$k_{y}$',title)
        else:
            gplot.plot_1d(ky,omega,'$k_y\\rho_i$',title=title)
            plt.grid(True)
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
    
    if fluxes is not None:
        xlab = r'$t a/v_{th,ref}$'
        figsize = (12,8)
        if fluxes_ratio is not None:
            print(fluxes.shape)
            fig, axes = plt.subplots(figsize=(24,24), ncols = 3, nrows = 3)
            for iflux in range(3):   
                for ifield in range(len(fields_present)):
                    label = f_labels[fields_present[ifield]]
                    color = f_colors[fields_present[ifield]]
                    title = flux_labels[iflux]
                    gplot.plot_1d(t, fluxes_i[ifield,iflux,:], xlab, axes=axes[iflux,0], log='y',ylab = title.format("i,"), color=color, label = label)
                    gplot.plot_1d(t, fluxes_e[ifield,iflux,:], xlab, axes=axes[iflux,1], log='y', ylab = title.format("e,"), color=color)
                    gplot.plot_1d(t, fluxes_ratio[ifield,iflux,:], xlab, axes=axes[iflux,2], title = r"$i/e$", color=color)
                avg = np.amax(np.mean(fluxes_ratio[:,iflux,len(t)//2:],axis=1))
                axes[iflux][2].set_ylim(0.8*avg,1.2*avg)
                axes[iflux][0].legend(loc='best')
        else:
            fig = plt.figure(figsize=(12,8))
            axes = [fig.gca()]
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

    if fieldskykxtheta is not None:
        maxfieldskykx = np.amax(fieldskykxtheta, axis = 3, keepdims = True)
        normfieldskykxtheta = fieldskykxtheta / maxfieldskykx
        normfulldrifts = fulldrifts/np.amax(fulldrifts, axis=2, keepdims = True)

        Ntheta0 = len(theta0[0,:])
        dtheta0 = theta0[0,1]-theta0[0,0]
        colors = gplot.truncate_colormap('nipy_spectral', 0.0, 0.9, Ntheta0)
        many_th0 = Ntheta0 > 1
        show_ri = False
        norm = mpl_colors.BoundaryNorm(theta0[0,:]-dtheta0/2.0,Ntheta0)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=colors)
        sm.set_array([])
        print('plotting fields vs theta for all ky')
        for iky in range(len(ky)):
            fig, axes = plt.subplots(nrows = len(fields)+1, figsize = (16, 6*len(fields)), gridspec_kw = {'wspace':0, 'hspace':0} )
            xlab = r'$\theta-\theta_0$'
            #if len(theta0[0,:]) == 1:
            for ith0 in range(len(theta0[0,:])):
                #ith0 = 0
                th0 = theta0[0,ith0]
                x = theta-th0
                lines = []
                labels = []
                title = r'Quantities vs $\theta-\theta_0$ for different $\theta_0$ at $k_y={}$'.format(ky[iky])
                j0lab = r'$J_0(|k_\perp|)$'
                driftlab = r'$\omega_{D}/\omega_{D,\mathrm{max}}$'
                if many_th0:
                    th0_colors = [colors(ith0)]*(len(fields)+1)
                else:
                    th0_colors = [f_colors[c] for c in fields_present]
                    th0_colors.append(f_colors['other'])
                for ifield in range(len(fields)):
                    ax = axes[ifield+1]
                    ylab = f_labels[fields_present[ifield]]
                    gplot.plot_1d(x, fieldskykxtheta[ifield,iky,ith0,:], xlab, axes=ax, ylab=ylab, rads=True, linewidth=2.0, color = th0_colors[ifield], label = str(th0))
                    if show_ri:
                        gplot.plot_1d(x, fields[ifield,iky,ith0,:,0], axes=ax, color = f_colors[fields_present[ifield]], linestyle='--', label='Real')
                        gplot.plot_1d(x, fields[ifield,iky,ith0,:,1], axes=ax, color = f_colors[fields_present[ifield]], linestyle=':', label='Im')
                    #ax.legend()

                gplot.plot_1d(x, normfulldrifts[iky,ith0,:], xlab, axes=axes[0], rads = True, linestyle = '--', color = th0_colors[-1])
                gplot.plot_1d(x, fullj0[iky,ith0,:], xlab, title=title, axes=axes[0], rads = True, linestyle=':', color = th0_colors[-1])
            
            cbar_ax = fig.add_axes([0.115, 0.01, 0.825, 0.03])
            cbar_ax.set_title(r'$\theta_0$', pad=-100) 
            fig.colorbar(sm, cax = cbar_ax, ticks=theta0[0,:], format='%.1f', orientation='horizontal')
            axes[0].plot([0],[0],label=driftlab, color='black', linestyle='--')
            axes[0].plot([0],[0],label=j0lab, color='black', linestyle=':')

            axes[0].legend()
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1   
            #else:
            #    print("Not plotting full mode structures when ntheta0 > 1, as it would become too convoluted.")

   
    merged_pdfname = 'range_plots'
    gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

def get_growthrate(t,phi2,it_start,it_end,ikx,iky):
    # OB 031018 ~ Growth rate is half of the gradient of log(phi2)  
    popt, pcov = opt.curve_fit(lin_func, t[it_start:it_end], np.log(phi2[it_start:it_end,iky,ikx]))
    return popt[0]/2

def lin_func(x,a,b):
    return a*x+b

def movie_fields_theta(ifile, run, mydict):
    if run.no_plot:
        return
    t = mydict['t']
    nt = len(t)
    fields = mydict['fields_t']    # field,t,ky,kx,theta,ri 
    fields_present = mydict['fields_t_present']    # field,t,ky,kx,theta,ri 
    ky = mydict['ky']

    absfields = np.sqrt(fields[:,:,:,:,:,0]**2 + fields[:,:,:,:,:,1]**2)
    absfieldsthetat = np.mean(np.mean(absfields,2),2)
    theta = mydict['theta']
    fullj0 = mydict['fullj0']
    fulldrifts = mydict['fulldrifts']
    
    for iky in range(len(ky)):
        ## Save snapshots
        tmp_pdf_id = 1
        pdflist = []
        fig, axes = plt.subplots(nrows = len(fields), figsize = (16, 6*len(fields)), gridspec_kw = {'wspace':0, 'hspace':0} )
        if type(axes).__name__ not in ['ndarray', 'list']:
            axes = np.array([axes])
        # Set up formatting for the movie files
        Writer = anim.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        # intialize three line objects (one in each axes)
        line = []
        for ifield in range(len(fields)):
            ax = axes[ifield]
            field_type = fields_present[ifield]
            ax.set_ylabel(f_labels[field_type])
            ax.set_xlabel(r'$\theta$')
            ax.grid()
            ax.set_xlim(np.amin(theta), np.amax(theta))
            field_abs_line, = ax.plot([], [], lw=2, color=f_colors[fields_present[ifield]])
            field_r_line,   = ax.plot([], [], lw=1, color=f_colors[fields_present[ifield]], linestyle = '--')
            field_i_line,   = ax.plot([], [], lw=1, color=f_colors[fields_present[ifield]], linestyle = ':' )
            line = line + [ field_abs_line, field_r_line, field_i_line ]
        #j0_line, = axes[0].plot([], [], lw=1, color='green', linestyle='--', label = r'$J_0(k_\perp)$')
        #drift_line, = axes[0].plot([], [], lw=1, color='red', linestyle='--', label = r'$\omega_D/\omega_{D,\rm{max}}$')
        #line = line + [j0_line, drift_line, phi2_line]
        
        # the same axes initalizations as before (just now we do it for both of them)
        #max_phi2bloon_this_t = max_phi2bloon[iiky][ it_start_for_snap:max_it_for_snap ]
        #ax2.legend(loc = 'upper right')
        
        def data_gen(theta_dat, abs_dat, ri_dat, t_range):
            frame = 0
            for it in t_range:
                print("Frame {}/{}...".format(frame,n_frames), end="\r")
                theta = theta_dat[:]
                #drift = drift_dat/np.amax(drift_dat)
                #kperp = kperp_dat
                abs_fields = abs_dat[:,it,:]
                ri_fields = ri_dat[:,it,:,:]
                frame = frame + 1
                yield theta,abs_fields,ri_fields

        def runplot(data):
            # update the data
            theta, abs_f, ri_f = data


            # update the data of both line objects
            for ifield in range(len(abs_f)):
                # axis limits checking. Same as before, just for both axes
                ylim = np.amax( abs_f[ifield,:] ) * 1.1 
                axes[ifield].set_ylim( bottom = -ylim, top = ylim )
                axes[ifield].figure.canvas.draw()
                line[ifield*3].set_data(theta, abs_f[ifield, :])
                line[1 + ifield*3].set_data(theta, ri_f[ifield, :,0])
                line[2 + ifield*3].set_data(theta, ri_f[ifield, :,1])
            return line

        # initialize the data arrays 
        it_start = int(len(t)*0.0)
        it_end = int(len(t))
        it_step = 1
        n_frames = (it_end - it_start) // it_step + min((it_end - it_start)%it_step,1) 
        #(max_it_for_snap - it_start_for_snap)//it_step_for_snap + min((max_it_for_snap - it_start_for_snap)%it_step_for_snap,1)
        ani = anim.FuncAnimation(fig, runplot, 
            data_gen(   theta, 
                        absfields[:, :, iky, 0, :], 
                        fields[:, :, iky, 0, :, :], 
                        range( it_start, it_end, it_step )   ),
            blit=False, interval=15,
            repeat=False, save_count=n_frames)
        #ani.save('phibloon_t.mp4', writer=writer)
        gplot.save_anim(ani, writer, 'phibloon_t_ky_{:1.2f}'.format(ky[iky]), run, ifile = ifile)


    plt.close()





scanlabels = {'bprim':'$\\beta\'$', 'kap':'$\\kappa$', 'tri':'$\\delta$', 'beta':'$\\beta$', 'shat':'$\hat{s}$', 'vnewke':'$vnewk_e$','zeff':r'$Z_{\rm{eff}}$',
'tprimi':r'tprim (primary ions)','tprime':r'tprim (electrons)','tprimx':r'tprim (impurity)','vnewki':r'vnewk (primary ions)','vnewke':r'vnewk (electrons)','vnewkx':r'vnewk (impurity)','fprim':r'fprim',
'ntheta':'ntheta','negrid':'negrid', 'ngauss':'ngauss', 'vcut':'vcut','rhoc':r'$r/a_{\rm{min}}$'
}

# Scans in linear growth rates, changing in two parameters, x and y.
def scan2d(run,x,y):
    # Only execute if plotting
    if run.no_plot:
        return

    setpilabels = ['bumppos']
    print("Plotting various quantities for scan of ("+str(x)+" vs "+str(y)+")...")
    Nfile = len(run.fnames)

    # Init arrays of data used in scan.
    full_lingrowth = [dict() for ifile in range(Nfile)]
    
    # Get lingrowth data from .dat file.
    for ifile in range(Nfile):
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.lingrowth.dat'
        with open(datfile_name,'rb') as datfile:
            full_lingrowth[ifile] = pickle.load(datfile)
    xdat,ydat = np.zeros(Nfile),np.zeros(Nfile)
    for ifile in range(Nfile):
        
        lastcharx = x[-1]
        xstem = x[:len(x)-1]
        if xstem in ['dens','temp','fprim','tprim','vnewk']:
            if lastcharx == 'i':
                specindex = 1
            if lastcharx == 'e':
                specindex = 2
            if lastcharx == 'x':
                specindex = 3
            xdat[ifile] = full_lingrowth[ifile]['specparams'][specindex-1][xstem]
        elif x in ['dens','temp','fprim','tprim','vnewk']:
            xdat[ifile] = full_lingrowth[ifile]['specparams'][0][x]
        else:
            xdat[ifile] = full_lingrowth[ifile][x]

        lastchary = y[-1]
        ystem = y[:len(y)-1]
        if ystem in ['dens','temp','fprim','tprim','vnewk']:
            if lastchary == 'i':
                specindex = 1
            if lastchary == 'e':
                specindex = 2
            if lastchary == 'y':
                specindex = 3
            ydat[ifile] = full_lingrowth[ifile]['specparams'][specindex-1][ystem]
        elif y in ['dens','temp','fprim','tprim','vnewk']:
            ydat[ifile] = full_lingrowth[ifile]['specparams'][0][y]
        else:
            ydat[ifile] = full_lingrowth[ifile][y]

        ydat[ifile] = round(ydat[ifile], 3)
        ydat[ifile] = round(ydat[ifile], 3)

    xvals = sorted(list(set(xdat)))
    yvals = sorted(list(set(ydat)))
    print(str(x) + " values: " + str(xvals))
    print(str(y) + " values: " + str(yvals))
    if len(xvals) * len(yvals) != Nfile:
        quit("Incorrect number of files added to populate the scan - exiting")
    gammas = np.zeros((len(xvals), len(yvals)))
    kymax = np.zeros((len(xvals), len(yvals)))
    phis = np.zeros((len(xvals), len(yvals)))
    apars = np.zeros((len(xvals), len(yvals)))
    markersize = np.zeros((len(xvals), len(yvals)))
    for ifile in range(Nfile):
        gamma = full_lingrowth[ifile]['gamma']
        phikykx = full_lingrowth[ifile]['phikykx']
        aparkykx = full_lingrowth[ifile]['aparkykx']
        ky = full_lingrowth[ifile]['ky']
        kx = full_lingrowth[ifile]['kx']
        # Limits search to ITG.
        ikymax = int((np.abs(ky-1.0)).argmin())
        for ix in range(len(xvals)):
            for iy in range(len(yvals)):
                if xdat[ifile] == xvals[ix] and ydat[ifile] == yvals[iy]:
                    maxindex = np.nanargmax(gamma[:ikymax])
                    if maxindex < ikymax-1 and gamma[maxindex] > gamma[maxindex+1]:
                        gammas[ix,iy] = gamma[maxindex]
                        kymax[ix,iy] = ky[maxindex]
                        markersize[ix,iy] = 0
                    else:
                        gammas[ix,iy] = 0
                        kymax[ix,iy,] = 0
                        markersize[ix,iy] = 30
                    phis[ix,iy] = np.amax(phikykx)
                    apars[ix,iy] = np.amax(aparkykx)
    pdflist = [] 
    tmp_pdf_id=0
    gplot.plot_2d(gammas,xvals,yvals,np.min(gammas[:,:]),np.max(gammas[:,:]),cmp='Reds',xlab=scanlabels[x],ylab=scanlabels[y],title='$\gamma a/v_t$', markersize=np.reshape(markersize[:,:],np.size(markersize[:,:])))
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1

    gplot.plot_2d(kymax,xvals,yvals,np.min(kymax[:,:]),np.max(kymax[:,:]),cmp='Reds_r',xlab=scanlabels[x],ylab=scanlabels[y],title='$k_y\\rho_i$') 
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1

    gplot.plot_2d(phis/apars,xvals,yvals,np.min(phis/apars),np.max(phis/apars),cmp='Reds_r',xlab=scanlabels[x],ylab=scanlabels[y],title=r'$|\delta\phi|/|\delta{A}_\parallel|_{\rm{max}}$') 
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1

    merged_pdfname = str(x) + "_" + str(y) + "_scan"
    gplot.merge_pdfs(pdflist, merged_pdfname, run)
 

# Scans in linear growth rates, changing in x.
def scan1d(run,x):
    # Only execute if plotting
    if run.no_plot:
        return

    
    print("Plotting scan of growth rates with " + str(x) + "...")
    Nfile = len(run.fnames)

    # Init arrays of data used in scan.
    full_lingrowth = [dict() for ifile in range(Nfile)]
    
    # Get lingrowth data from .dat file.
    for ifile in range(Nfile):
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.lingrowth.dat'
        with open(datfile_name,'rb') as datfile:
            full_lingrowth[ifile] = pickle.load(datfile)
    
    xdat = np.zeros(Nfile)

    

    for ifile in range(Nfile):
        lastcharx = x[-1]
        xstem = x[:len(x)-1]
        if xstem in ['dens','temp','fprim','tprim','vnewk']:
            if lastcharx == 'i':
                specindex = 1
            if lastcharx == 'e':
                specindex = 2
            if lastcharx == 'x':
                specindex = 3
            xdat[ifile] = full_lingrowth[ifile]['specparams'][specindex-1][xstem]
        elif x in ['dens','temp','fprim','tprim','vnewk']:
            xdat[ifile] = full_lingrowth[ifile]['specparams'][0][x]
        else:
            xdat[ifile] = full_lingrowth[ifile][x]
    xvals = sorted(list(set(xdat)))
    print(str(x) + " values: " + str(xvals))
    gammamaxs = np.zeros(len(xvals))
    kymax = np.zeros(len(xvals))
    phimax = np.zeros(len(xvals))
    aparmax = np.zeros(len(xvals))
    phi_o_apar_max = np.zeros(len(xvals))
    markersize = np.zeros(len(xvals))
    ky = full_lingrowth[0]['ky']
    gammas = np.zeros(( len(xvals),len(ky) ))
    omegas = np.zeros(( len(xvals),len(ky) ))
    for ifile in range(Nfile):
        ikmax = full_lingrowth[ifile]['ikmax']
        gammas[ifile] = full_lingrowth[ifile]['gamma'][:,0]
        omegas[ifile] = full_lingrowth[ifile]['omega'][:,0]
        gammamaxs[ifile] = full_lingrowth[ifile]['gamma'][ikmax[0],ikmax[1]]
        print('ky at max is {}'.format(ky[ikmax[0]]))
        phimax[ifile] = np.amax(full_lingrowth[ifile]['phikykxtheta'][ikmax[0],ikmax[1],:])
        if 'aparkykxtheta' in full_lingrowth[ifile]:
            aparmax[ifile] = np.amax(full_lingrowth[ifile]['aparkykxtheta'][ikmax[0],ikmax[1],:])
            phi_o_apar_max[ifile] = phimax[ifile]/aparmax[ifile]
        kymax[ifile] = full_lingrowth[ifile]['ky'][ikmax[0]]
        kxmax = full_lingrowth[ifile]['kx'][ikmax[1]]
    pdflist = [] 
    tmp_pdf_id=0
    
    gplot.plot_1d(xvals,gammamaxs, xlab=scanlabels[x],title='$\gamma_{max} a_{GS2}/v_{th,ref}$', grid = "both")
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1
    
    gplot.plot_2d(np.transpose(gammas),ky,xvals,np.amin(gammas), np.amax(gammas), xlab=r'$k_y$', ylab=r'$r$', title = r'$\gamma$', cmp='RdBu_r' )
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1
    #def plot_2d(z,xin,yin,zmin,zmax,xlab='',ylab='',title='',cmp='RdBu',use_logcolor=False, interpolation='nearest', markersize=[], anim=False):

    gplot.plot_multi_1d(ky,gammas,xlab=r'$k_y$',title=r'$\gamma$', grid='both',labels=['{:1.2e}'.format(xval) for xval in xvals],legendtitle=scanlabels[x])
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1

    gplot.plot_multi_1d(ky,omegas,xlab=r'$k_y$',title=r'$\omega$', grid='both',labels=['{:1.2e}'.format(xval) for xval in xvals],legendtitle=scanlabels[x])
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1

    gplot.plot_1d(xvals,kymax, xlab=scanlabels[x],title=r'$k_y^{\gamma_{max}}\rho_i$', grid = 'both')
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1

    gplot.plot_1d(xvals,phimax, xlab = scanlabels[x],title=r'log$_{10}|\delta\phi|_{\rm{max}}$', grid='both',log='both')
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1

    gplot.plot_1d(xvals,aparmax, xlab = scanlabels[x],title=r'log$_{10}|\delta{A}_\parallel|_{\rm{max}}$', grid='both',log='both')
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1

    gplot.plot_1d(xvals,phi_o_apar_max, xlab = scanlabels[x],title=r'$|\delta\phi|/|\delta{A}_\parallel|_{\rm{max}}$', grid='both')
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1


    merged_pdfname = str(x) + "_scan"
    gplot.merge_pdfs(pdflist, merged_pdfname, run)
 




 
