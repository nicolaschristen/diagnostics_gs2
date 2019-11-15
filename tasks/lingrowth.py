import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
import gs2_plotting as gplot
import pickle
import math

def my_task_single(ifile, run, myin, myout):

    # User parameters
    dump_at_start = 0.3 # fraction of initial time to dump when fitting
    ikx_list = [-1] # choose which kx to plot, negative means plot all
    skip_first_ky = False
    kymax = 1.5 # max of gamma is taken over range [0, kymax]
    fix_ylim = True
    yl = [0.00, 0.40]

    iky_first = 0
    if skip_first_ky:
        iky_first = 1

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
        shat = myin['theta_grid_parameters']['shat']
        g_exb = 0.052557 ### USER ###

        # Store index of first NaN or +/- inf in it_stop
        it = 0
        it_stop = nt
        no_nan_inf = True
        while it < nt and no_nan_inf:
            for ikx in range(nakx):
                for iky in range(1,naky):
                    if not is_number(phi2[it,iky,ikx]):
                        no_nan_inf = False
                        it_stop = it
            it = it + 1

        if grid_option=='range':
            # In range, plot the only kx
            ikx_list = [0]
        elif grid_option=='box':
            if ikx_list[0]==-1:
                ikx_list = [i in range(nakx)]

        # Fit phi to get growthrates
        gamma = np.zeros([naky,len(ikx_list)])
        gamma[0,:]=float('nan') # skip zonal mode
        for iky in range(iky_first,naky):
            for ikx in ikx_list:
                gamma[iky,ikx] = 0.5*get_growthrate(t,phi2,it_start,it_stop,ikx,iky)

        # Read real frequency
        omega = np.zeros([naky,len(ikx_list)])
        omega[0,:] = float('nan') # skip zonal mode
        for iky in range(iky_first,naky):
            for ikx in ikx_list:
                omega[iky,ikx] = myout['omega_average'][-1,iky,ikx,0] # last index is for real part
        
        # Save to .dat file
        datfile_name = run.out_dir + run.fnames[ifile] + '.lingrowth.dat'
        with open(datfile_name, 'wb') as datfile:
            pickle.dump([ikx_list,kx,ky,gamma,omega],datfile)
   
    # or read from .dat file
    else:
        
        datfile_name = run.out_dir + run.fnames[ifile] + '.lingrowth.dat'
        with open(datfile_name, 'rb') as datfile:
            [ikx_list,kx,ky,gamma,omega] = pickle.load(datfile)

    # Plotting
    if not run.no_plot:

        # Plot growthrate
        plt.figure(figsize=(12,8))
        my_legend = []
        plt.grid(True)
        plt.xlabel('$k_y\\rho_i$')
        plt.ylabel('$\\gamma \\ [v_{thr}/r_r]$')
        # Find maximum gamma, within range [0, kymax]
        ikymax = 0
        gamma_max = -1000
        for ikx in ikx_list:
            iky = 0
            while ky[iky] <= 1.5 and iky < naky:
                if not math.isnan(gamma[iky,ikx]) and gamma_max < gamma[iky,ikx]:
                    gamma_max = gamma[iky,ikx]
                    ikymax = iky
                iky = iky + 1
            plt.plot(ky,gamma[:,ikx])
            my_legend.append('$\\rho_i k_x='+str(kx[ikx])+'$')
        plt.legend(my_legend)
        my_title = '$\\max(\\gamma)=' + str(round(gamma_max,3)) + '$'
        my_title = my_title + ' at $k_y = ' + str(round(ky[ikymax],2)) + '$'
        plt.title(my_title)
        if fix_ylim:
            plt.ylim(yl)
        pdfname = 'lingrowth'
        gplot.save_plot(pdfname, run, ifile)
        print('Maximum linear growthrate: '+str(gamma_max))
    
        # Plot Floquet vs growthrate
        if g_exb != 0.0:

            Tf = 2*math.pi*shat/g_exb

            plt.figure(figsize=(12,8))
            my_legend = []
            plt.grid(True)
            plt.xlabel('$k_y\\rho_i$')
            plt.ylabel('$T_f\\gamma$')
            for ikx in ikx_list:
                plt.plot(ky,Tf*gamma[:,ikx])
                my_legend.append('$\\rho_i k_x='+str(kx[ikx])+'$')
            plt.legend(my_legend)
            my_title = '$T_F=' + str(round(Tf,3)) + '\ [r_r/v_{thr}]$'
            plt.title(my_title)
            pdfname = 'floq_vs_growth'
            gplot.save_plot(pdfname, run, ifile)
    
        # Plot real frequency
        plt.figure(figsize=(12,8))
        plt.xlabel('$k_y\\rho_i$')
        plt.ylabel('$\\omega \\ [v_{thr}/r_r]$')
        plt.title('Real freq. at last time-step')
        plt.grid(True)

        my_legend = []
        for ikx in ikx_list:
            plt.plot(ky,omega[:,ikx])
            my_legend.append('$\\rho_i k_x='+str(kx[ikx])+'$')
        plt.legend(my_legend)
        pdfname = 'realfreq'
        gplot.save_plot(pdfname, run, ifile)

        # Plot potential
        plt.figure(figsize=(12,8))
        plt.xlabel('$t\\ [L/v_{th}]$')
        plt.ylabel('$\\vert\\varphi\\vert^2$')
        plt.title('Potential')
        plt.grid(True)

        my_legend = []
        cmap = plt.get_cmap('nipy_spectral')
        my_colors = [cmap(i) for i in np.linspace(0,1,naky*len(ikx_list))]
        for iky in range(naky):
            for ikx in ikx_list:
                plt.semilogy(t[0:it_stop],phi2[0:it_stop,iky,ikx],color=my_colors[ikx*naky+iky])
                my_legend.append('$(k_x,k_y)=({:.1f},{:.1f})$'.format(kx[ikx],ky[iky]))
        plt.legend(my_legend, ncol=1, prop={'size': 10},loc='upper left')
        plt.ylim([np.amin(phi2[0:it_stop,:,:]), np.amax(phi2[0:it_stop,:,:])])
        pdfname = 'linpotential'
        gplot.save_plot(pdfname, run, ifile)

def get_growthrate(t,phi2,it_start,it_stop,ikx,iky):
   
    popt, pcov = opt.curve_fit(lin_func, t[it_start:it_stop], np.log(phi2[it_start:it_stop,iky,ikx]))
    return popt[0]

def lin_func(x,a,b):
    return a*x+b

def is_number(x):
    try:
        x_str = str(float(x))
        if x_str=='nan' or x_str=='inf' or x_str=='-inf':
            return False
    except ValueError:
        try:
            complex(x_str)
        except ValueError:
            return False
    return True
