import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import gs2_plotting as gplot
import matplotlib.colors as mcolors
import pickle
import math

def my_task_single(ifile, run, myin, myout, mytime):





    # vvv User parameters vvv

    # Fraction of time to be discarded when fitting
    tRatio_init_dump = 0.8
    tRatio_end_dump = 0.0

    # Limit y-axis in growthrate plots ?
    fix_ylim = False
    ylim = (0.00, 0.11)

    # Limit colorbar on gamma vs (theta0,ky) plot ?
    fix_cbarlim = False
    cbarmin = -0.05
    cbarmax = 0.4

    # Specify flow shear rate to compare Tfloq vs gamma
    g_exb = 0.0

    # ^^^ User parameters ^^^






    # Process data


    t = myout['t']
    nt = t.size
    it_start = int(nt*tRatio_init_dump)
    tt0 = myout['theta0'][0]
    ntt0 = tt0.size
    ky = myout['ky']
    naky = ky.size
    theta = myout['theta']
    # modulus squared, avged over theta (indices: [t,ky,tt0])
    phi2 = myout['phi2_by_mode']
    # with theta dependence: phi2_bytheta[t,ky,tt0,theta]
    phi_t_present = myout['phi_t_present']
    if phi_t_present:
        phi2_bytheta = np.sum(np.power(myout['phi_t'],2), axis=4)
    else:
        print('\nCannot plot phi vs theta: write_phi_over_time was probably set to false.\n')
    shat = myin['theta_grid_parameters']['shat']
    # modulus squared of non-adiabatic part of the density at last time step [spec,ky,kx,theta]
    if myout['density_present']:
        dens2 =  np.sum(np.power(myout['density'],2), axis=4)
    else:
        print("\nCannot plot dens vs theta, 'density' was not found in output.\n")

    # Store index of first NaN or +/- inf in it_stop
    it = 0
    it_stop = int(nt*(1-tRatio_end_dump))
    no_nan_inf = True
    while it < nt and no_nan_inf:
        for itt0 in range(ntt0):
            for iky in range(1,naky):
                if not is_number(phi2[it,iky,itt0]):
                    no_nan_inf = False
                    it_stop = it
        it = it + 1

    # Fit phi^2 to get growthrates
    gamma = np.zeros([naky,ntt0])
    for iky in range(naky):
        for itt0 in range(ntt0):
            if ky[iky] == 0.0:
                # Skip zonal modes
                gamma[iky,itt0] = float('nan')
            else:
                # Factor of 0.5 because we fit phi^2
                gamma[iky,itt0] = 0.5*get_growthrate(t,phi2,it_start,it_stop,itt0,iky)

    # Read real frequency
    omega = np.zeros([naky,ntt0])
    for iky in range(naky):
        for itt0 in range(ntt0):
            # First idx = last time step, last idx = real part
            omega[iky,itt0] = myout['omega_average'][-1,iky,itt0,0]

    # Compute <Qe/Qi>_t
    try:
        Qe = myout['es_heat_flux'][:,1]
        Qi = myout['es_heat_flux'][:,0]
        Qratio = Qe/Qi
        Qratio_avg = mytime.timeavg(Qratio)
    except:
        Qratio_avg = float('nan')
    
    # Save to .dat file
    datfile_name = run.out_dir + run.fnames[ifile] + '.linrange.dat'
    with open(datfile_name, 'wb') as datfile:
        pickle.dump([tt0,ky,gamma,omega,Qratio_avg],datfile)







    # Plotting


    # Plot growthrate vs ky, one plot per theta0

    if naky > 1:

        plt.figure(figsize=(12,8))

        tmp_pdf_id = 1
        pdflist = []
        for itt0 in range(ntt0):
            l, = plt.plot(ky,gamma[:,itt0], color=gplot.myblue, linewidth=3.0)
            l.set_label('$\\theta_0=$'+gplot.str_tt0(tt0[itt0]))
            gplot.legend_matlab()
            if fix_ylim:
                plt.ylim(ylim)
            plt.grid(True)
            plt.xlabel('$k_y\\rho_i$')
            plt.ylabel('$\\gamma \\ [v_{th}/a]$')
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1

        merged_pdfname = 'gamma_vs_ky'
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
        plt.clf()
        plt.cla()


    # Plot growthrate vs theta0, one plot per ky

    if ntt0 > 1:

        plt.figure(figsize=(12,8))

        tmp_pdf_id = 1
        pdflist = []
        for iky in range(naky):
            l, = plt.plot(tt0,gamma[iky,:], color=gplot.myblue, linewidth=3.0)
            l.set_label('$k_y=$'+gplot.str_ky(ky[iky]))
            gplot.legend_matlab()
            if fix_ylim:
                plt.ylim(ylim)
            plt.grid(True)
            plt.xlabel('$\\theta_0$')
            plt.ylabel('$\\gamma \\ [v_{th}/a]$')
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1

        merged_pdfname = 'gamma_vs_theta0'
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
        plt.clf()
        plt.cla()


    # Contour gamma vs (theta0, ky)

    if naky > 1 and ntt0 > 1:

        xl = '$\\theta_0$'
        yl = '$\\rho k_y$'
        g = gamma
        if not fix_cbarlim:
            gmax = np.amax(g)
            gmin = np.amin(g)
        else:
            gmin, gmax = [cbarmin, cbarmax]
        gplot.plot_2d(g,tt0,ky,gmin,gmax,xlab=xl,ylab=yl,cmp='RdBu_c',title='$\\gamma\\ [v_{th}/a]$')
        pdfname = 'gamma_vs_theta0_ky'
        gplot.save_plot(pdfname, run, ifile)
    

    # Plot real frequency vs ky, one plot per theta0

    if naky > 1:

        plt.figure(figsize=(12,8))

        tmp_pdf_id = 1
        pdflist = []
        for itt0 in range(ntt0):
            l, = plt.plot(ky,omega[:,itt0], color=gplot.myblue, linewidth=3.0)
            l.set_label('$\\theta_0=$'+gplot.str_tt0(tt0[itt0]))
            gplot.legend_matlab()
            plt.grid(True)
            plt.xlabel('$k_y\\rho_i$')
            plt.ylabel('$\\omega \\ [v_{th}/a]$')
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1

        merged_pdfname = 'omega_vs_ky'
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
        plt.clf()
        plt.cla()


    # Plot real frequency vs theta0, one plot per ky

    if ntt0 > 1:

        plt.figure(figsize=(12,8))

        tmp_pdf_id = 1
        pdflist = []
        for iky in range(naky):
            l, = plt.plot(tt0,omega[iky,:], color=gplot.myblue, linewidth=3.0)
            l.set_label('$k_y=$'+gplot.str_ky(ky[iky]))
            gplot.legend_matlab()
            plt.grid(True)
            plt.xlabel('$\\theta_0$')
            plt.ylabel('$\\omega \\ [v_{th}/a]$')
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1

        merged_pdfname = 'omega_vs_theta0'
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
        plt.clf()
        plt.cla()


    # Contour real frequency vs (theta0, ky)

    if naky > 1 and ntt0 > 1:

        xl = '$\\theta_0$'
        yl = '$\\rho k_y$'
        z = omega
        zmax = np.amax(z)
        zmin = np.amin(z)
        gplot.plot_2d(z,tt0,ky,zmin,zmax,xlab=xl,ylab=yl,cmp='RdBu_c',title='$\\omega\\ [v_{th}/a]$')
        pdfname = 'omega_vs_theta0_ky'
        gplot.save_plot(pdfname, run, ifile)


    # Plot potential vs t for each ky, one plot per theta0

    plt.figure(figsize=(12,8))

    tmp_pdf_id = 1
    pdflist = []
    if naky > 1:
        cmap = plt.get_cmap('nipy_spectral')
        my_colors = [cmap(i) for i in np.linspace(0,1,naky)]
    else:
        my_colors = [gplot.myblue]
    for itt0 in range(ntt0):
        lgd = []
        for iky in range(naky):
            plt.semilogy(t[0:it_stop],phi2[0:it_stop,iky,itt0],color=my_colors[iky])
            lgd.append('$k_y=$'+gplot.str_ky(ky[iky]))
        leg = plt.legend(lgd,prop={'size': 12}, ncol=3,frameon=True,fancybox=False,framealpha=1.0)
        leg.get_frame().set_facecolor('w')
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(1.0)
        plt.grid(True)
        plt.xlabel('$t\\ [a/v_{th}]$')
        plt.ylabel('$\\vert\\varphi\\vert^2$')
        plt.title('$\\theta_0=$'+gplot.str_tt0(tt0[itt0]))
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

    merged_pdfname = 'phi_vs_t'
    gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
    plt.clf()
    plt.cla()


    # Plot potential at t[it_stop] vs theta for every ky, one plot per theta0
    # Plot vs lin-lin, lin-log and log-log scales.

    # lin-lin

    plt.figure(figsize=(12,8))

    tmp_pdf_id = 1
    pdflist = []
    if naky > 1:
        cmap = plt.get_cmap('nipy_spectral')
        my_colors = [cmap(i) for i in np.linspace(0,1,naky)]
    else:
        my_colors = [gplot.myblue]
    for itt0 in range(ntt0):
        lgd = []
        for iky in range(naky):
            plt.plot(theta,phi2_bytheta[it_stop-1,iky,itt0,:],color=my_colors[iky])
            lgd.append('$k_y=$'+gplot.str_ky(ky[iky]))
        leg = plt.legend(lgd,prop={'size': 12}, ncol=3,frameon=True,fancybox=False,framealpha=1.0)
        leg.get_frame().set_facecolor('w')
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(1.0)
        plt.grid(True)
        plt.xlabel('$\\theta$')
        plt.ylabel('$\\vert\\varphi\\vert^2$')
        plt.title('$\\theta_0=$'+gplot.str_tt0(tt0[itt0]))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2E'))
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

    merged_pdfname = 'phi_vs_theta'
    gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
    plt.clf()
    plt.cla()

    # lin-log

    plt.figure(figsize=(12,8))

    tmp_pdf_id = 1
    pdflist = []
    if naky > 1:
        cmap = plt.get_cmap('nipy_spectral')
        my_colors = [cmap(i) for i in np.linspace(0,1,naky)]
    else:
        my_colors = [gplot.myblue]
    for itt0 in range(ntt0):
        lgd = []
        for iky in range(naky):
            plt.semilogy(theta,phi2_bytheta[it_stop-1,iky,itt0,:],color=my_colors[iky])
            lgd.append('$k_y=$'+gplot.str_ky(ky[iky]))
        leg = plt.legend(lgd,prop={'size': 12}, ncol=3,frameon=True,fancybox=False,framealpha=1.0)
        leg.get_frame().set_facecolor('w')
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(1.0)
        plt.grid(True)
        plt.xlabel('$\\theta$')
        plt.ylabel('$\\vert\\varphi\\vert^2$')
        plt.title('$\\theta_0=$'+gplot.str_tt0(tt0[itt0]))
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

    merged_pdfname = 'phi_vs_theta_linlog'
    gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
    plt.clf()
    plt.cla()

    # log-log

    plt.figure(figsize=(12,8))

    tmp_pdf_id = 1
    pdflist = []
    if naky > 1:
        cmap = plt.get_cmap('nipy_spectral')
        my_colors = [cmap(i) for i in np.linspace(0,1,naky)]
    else:
        my_colors = [gplot.myblue]
    for itt0 in range(ntt0):
        lgd = []
        for iky in range(naky):
            plt.loglog(theta,phi2_bytheta[it_stop-1,iky,itt0,:],color=my_colors[iky])
            lgd.append('$k_y=$'+gplot.str_ky(ky[iky]))
        leg = plt.legend(lgd,prop={'size': 12}, ncol=3,frameon=True,fancybox=False,framealpha=1.0)
        leg.get_frame().set_facecolor('w')
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(1.0)
        plt.grid(True)
        plt.xlabel('$\\theta$')
        plt.ylabel('$\\vert\\varphi\\vert^2$')
        plt.title('$\\theta_0=$'+gplot.str_tt0(tt0[itt0]))
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

    merged_pdfname = 'phi_vs_theta_loglog'
    gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
    plt.clf()
    plt.cla()


    # Plot dens^2 at tfinal vs theta for every ky, one plot per theta0

    # First ions

    plt.figure(figsize=(12,8))

    tmp_pdf_id = 1
    pdflist = []
    if naky > 1:
        cmap = plt.get_cmap('nipy_spectral')
        my_colors = [cmap(i) for i in np.linspace(0,1,naky)]
    else:
        my_colors = [gplot.myblue]
    for itt0 in range(ntt0):
        lgd = []
        for iky in range(naky):
            plt.semilogy(theta,dens2[0,iky,itt0,:],color=my_colors[iky])
            lgd.append('$k_y=$'+gplot.str_ky(ky[iky]))
        leg = plt.legend(lgd,prop={'size': 12}, ncol=3,frameon=True,fancybox=False,framealpha=1.0)
        leg.get_frame().set_facecolor('w')
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(1.0)
        plt.grid(True)
        plt.xlabel('$\\theta$')
        plt.ylabel('$\\vert\\delta n_{h}\\vert^2$')
        plt.title('$\\theta_0=$'+gplot.str_tt0(tt0[itt0]))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2E'))
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1

    merged_pdfname = 'dens_vs_theta_ion'
    gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
    plt.clf()
    plt.cla()

    # Then electrons

    if myin['species_knobs']['nspec'] > 1:

        plt.figure(figsize=(12,8))

        tmp_pdf_id = 1
        pdflist = []
        if naky > 1:
            cmap = plt.get_cmap('nipy_spectral')
            my_colors = [cmap(i) for i in np.linspace(0,1,naky)]
        else:
            my_colors = [gplot.myblue]
        for itt0 in range(ntt0):
            lgd = []
            for iky in range(naky):
                plt.semilogy(theta,dens2[1,iky,itt0,:],color=my_colors[iky])
                lgd.append('$k_y=$'+gplot.str_ky(ky[iky]))
            leg = plt.legend(lgd,prop={'size': 12}, ncol=3,frameon=True,fancybox=False,framealpha=1.0)
            leg.get_frame().set_facecolor('w')
            leg.get_frame().set_edgecolor('k')
            leg.get_frame().set_linewidth(1.0)
            plt.grid(True)
            plt.xlabel('$\\theta$')
            plt.ylabel('$\\vert\\delta n_{h}\\vert^2$')
            plt.title('$\\theta_0=$'+gplot.str_tt0(tt0[itt0]))
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2E'))
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1

        merged_pdfname = 'dens_vs_theta_electrons'
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
        plt.clf()
        plt.cla()


    # Plot TFloq*gamma vs ky, one plot per theta0

    if g_exb != 0.0 and naky > 1:

        # Floquet period
        Tf = 2*math.pi*shat/g_exb

        plt.figure(figsize=(12,8))

        tmp_pdf_id = 1
        pdflist = []
        for itt0 in range(ntt0):
            l, = plt.plot(ky,Tf*gamma[:,itt0], color=gplot.myblue, linewidth=3.0)
            l.set_label('$\\theta_0=$'+gplot.str_tt0(tt0[itt0]))
            gplot.legend_matlab()
            plt.grid(True)
            plt.xlabel('$k_y\\rho_i$')
            plt.ylabel('$\\gamma T_F$')
            my_title = '$T_F=' + str(round(Tf,3)) + '\ [r_r/v_{thr}]$'
            plt.title(my_title)
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1

        merged_pdfname = 'floq_vs_growth'
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
        plt.clf()
        plt.cla()






def get_growthrate(t,phi2,it_start,it_stop,itt0,iky):
   
    popt, pcov = opt.curve_fit(lin_func, t[it_start:it_stop], np.log(phi2[it_start:it_stop,iky,itt0]))
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
