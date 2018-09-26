import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
import gs2_plotting as gplot
import pickle
import gs2_plotting as gplot

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
                gamma[iky,ikx] = get_growthrate(t,phi2,it_start,ikx,iky)
        
        # Save to .dat file
        datfile_name = run.out_dir + run.fnames[ifile] + '.lingrowth.dat'
        with open(datfile_name, 'wb') as datfile:
            pickle.dump([ikx_list,kx,ky,gamma],datfile)
   
    # or read from .dat file
    else:
        
        datfile_name = run.out_dir + run.fnames[ifile] + '.lingrowth.dat'
        with open(datfile_name, 'rb') as datfile:
            [ikx_list,kx,ky,gamma] = pickle.load(datfile)

    # Plotting
    if not run.no_plot:
    
        # If we ran for many kx ky, then plot colormap
        if len(ikx_list) > 3 and ky.size > 3:

            title = '$\\gamma \\ [v_{thr}/r_r]$'
            
            xlab = '$k_{x}\\rho_i$'
            ylab = '$k_{y}\\rho_i$'

            cmap = 'RdBu'# 'RdBu_r','Blues'
            z = gamma[:,:] # taking out zonal modes because they are much larger
            z_min, z_max = z.min(), z.max()

            fig = gplot.plot_2d(z,kx,ky,z_min,z_max,xlab,ylab,title,cmap)

        # Otherwise plot curves vs ky for each kx
        else:
            plt.figure(figsize=(12,8))
            plt.xlabel('$k_y\\rho_i$')
            plt.ylabel('$\\gamma \\ [v_{thr}/r_r]$')
            plt.title('Linear growthrate')
            plt.grid(True)

            my_legend = []
            for ikx in ikx_list:
                plt.plot(ky,gamma[:,ikx])
                my_legend.append('$\\rho_i k_x='+str(kx[ikx])+'$')
            plt.legend(my_legend)

        pdfname = 'lingrowth'
        gplot.save_plot(pdfname, run, ifile)

        print('Maximum linear growthrate: '+str(np.nanmax(gamma)))

def get_growthrate(t,phi2,it_start,ikx,iky):
   
    popt, pcov = opt.curve_fit(lin_func, t[it_start:], np.log(phi2[it_start:,iky,ikx]))
    return popt[0]

def lin_func(x,a,b):
    return a*x+b
