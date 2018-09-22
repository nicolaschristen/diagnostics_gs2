import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
import gs2_plotting as gplot
import pickle

def my_task_single(ifile, run, myin, myout):

    # User parameters
    dump_at_start = 0.3 # fraction of initial time to dump when fitting
    ikx_list = [-1] # choose which kx to plot, negative means plot all

    # Compute and save growthrate
    if not run.only_plot:
        
        grid_option = myin['kt_grids_knobs']['grid_option']

        t = myout['t']
        nt = t.size
        it_start = round(nt*dump_at_start)
        kx = myout['kx']
        nakx = kx.size
        ky = myout['ky']
        naky = ky.size
        phi2 = myout['phi2_by_mode'] # modulus squared, avged over theta (indices: [t,ky,kx])
        it_end  = nt
        for it in range(nt):
            for ikx in range(nakx):
                for iky in range(naky):
                    if it < it_end and not np.isfinite(phi2[it,iky,ikx]):
                        it_end = it - 10 
                        break
        if grid_option=='range':
            # In range, plot the only kx
            ikx_list = [0]
        elif grid_option=='box':
            if ikx_list[0]==-1:
                ikx_list = [i in range(nakx)]

        
        # Fit phi to get growthrates
        gamma = np.zeros([naky,len(ikx_list)])
        gamma[0,:]=float('nan') # skip zonal mode
        for iky in range(1,naky):
            for ikx in ikx_list:
                gamma[iky,ikx] = get_growthrate(t,phi2,it_start,it_end,ikx,iky)
        mydict = {'ikx_list':ikx_list,'kx':kx,'ky':ky,'gamma':gamma,'tri':myin['theta_grid_parameters']['tri'],
                        'kap':myin['theta_grid_parameters']['akappa']}
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


    # Plotting
    if not run.no_plot:
    
        plt.figure(figsize=(12,8))
        plt.xlabel('$k_y\\rho_i$')
        plt.ylabel('$\\gamma \\ [v_{thr}/r_r]$')
        plt.title('Linear growthrate')
        plt.grid(True)
        my_legend = []
        for ikx in mydict['ikx_list']:
            plt.plot(mydict['ky'],mydict['gamma'][:,ikx])
            my_legend.append('$\\rho_i k_x='+str(mydict['kx'][ikx])+'$')
        plt.legend(my_legend)
        pdfname = 'lingrowth'
        gplot.save_plot(pdfname, run, ifile)
        print('Maximum linear growthrate: '+str(np.nanmax(mydict['gamma'])))

def get_growthrate(t,phi2,it_start,it_end,ikx,iky):
   
    popt, pcov = opt.curve_fit(lin_func, t[it_start:it_end], np.log(phi2[it_start:it_end,iky,ikx]))
    return popt[0]

def lin_func(x,a,b):
    return a*x+b
 
# OB ~ function to plot a heatmap of linear growth rates vs triangularity and elongation.
def trikap(run):
    # Only execute if plotting
    if run.no_plot:
        return
    print("Plotting scan of triangularity vs elongation...")
    Nfile = len(run.fnames)

    # Init arrays of data used in scan.
    full_lingrowth = [dict() for ifile in range(Nfile)]
    
    # Get lingrowth data from .dat file.
    for ifile in range(Nfile):
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.lingrowth.dat'
        with open(datfile_name,'rb') as datfile:
            full_lingrowth[ifile] = pickle.load(datfile)
    
    nakx = len(full_lingrowth[0]['kx']) 
    for ifile in range(Nfile):
        if len(full_lingrowth[ifile]['kx']) != nakx:
            quit("Error - number of kx values should be constant across all items in the scan - Exiting...")

    tri,kap = np.zeros(Nfile),np.zeros(Nfile)
    for ifile in range(Nfile):
        tri[ifile] = full_lingrowth[ifile]['tri']
        kap[ifile] = full_lingrowth[ifile]['kap']
    tris = sorted(list(set(tri)))
    kaps = sorted(list(set(kap)))
    print("Triangularity values: " + str(tris))
    print("Elongation values: " + str(kaps))
    if len(tris) * len(kaps) != Nfile:
        quit("Incorrect number of files added to populate the scan - exiting")
    gammas = np.zeros((len(tris), len(kaps), nakx))
    kymax = np.zeros((len(tris), len(kaps), nakx))
    
    for ifile in range(Nfile):
        gamma = full_lingrowth[ifile]['gamma']
        ky = full_lingrowth[ifile]['ky']
        kx = full_lingrowth[ifile]['kx']
        # Limits search to ITG.
        ikymax = int((np.abs(ky-1.0)).argmin())
        for itri in range(len(tris)):
            for ikap in range(len(kaps)):
                if tri[ifile] == tris[itri] and kap[ifile] == kaps[ikap]:
                    for ikx in range(len(kx)): 
                        gammas[itri,ikap,ikx] = np.nanmax(gamma[:ikymax,ikx])
                        kymax[itri,ikap,ikx] = ky[np.nanargmax(gamma[:ikymax,ikx])]

    pdflist = [] 
    tmp_pdf_id=0
    for ikx in range(nakx):    
        
        gplot.plot_2d(gammas,tris,kaps,np.min(gammas[:,:,ikx]),np.max(gammas[:,:,ikx]),cmp='Reds',xlab='$\delta$',ylab='$\kappa$',title='$\gamma a/v_t: k_x$ = ' + str(full_lingrowth[0]['kx'][ikx])) 
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id += 1

    merged_pdfname = 'tri_kap_scan'
    gplot.merge_pdfs(pdflist, merged_pdfname, run)


