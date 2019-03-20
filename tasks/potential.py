import sys
import numpy as np
import gs2_plotting as gplot
import matplotlib.pyplot as plt
import pickle

def my_task_single(ifile, run, myin, myout, mygrids, mytime):

    if not run.only_plot:
    
        #######################
        ### User parameters ###
        #######################
        
        # Choose which Fourier components to plot
        it_start_list = [mytime.it_min]
        ikx_start_list = [40] # kx=3 <=> ikx= 400 for old, 4 for new 
        iky_list = [1]

        ###########################
        ### End user parameters ###
        ###########################

        # Check parameters make sense
        size_check = (len(it_start_list)==len(ikx_start_list)==len(iky_list))
        if not size_check:
            print('potential.py: size mismatch in user parameters')
            sys.exit()

        # Number of Fourier components to plot
        Nplot = len(it_start_list)

        # Copy relevant quantities and make monotonous in kx
        t = myout['t']
        delt = myin['knobs']['delt']
        nwrite = myin['gs2_diagnostics_knobs']['nwrite']
        kxgrid = mygrids.kx # this version is monotonous
        dkx = kxgrid[1]-kxgrid[0]
        kygrid = mygrids.ky
        g_exb = myin['dist_fn_knobs']['g_exb']
        sgn_g = int(np.sign(g_exb))
        phi2_by_mode = myout['phi2_by_mode']
        phi2_by_mode = np.concatenate((phi2_by_mode[:,:,mygrids.nxmid:],phi2_by_mode[:,:,:mygrids.nxmid]),axis=2)

        # Arrays to store values for each (it,ikx,iky) chosen by the user
        kx_full = []
        ky_full = []
        t_plot_full = []
        t_zero_full = []
        t_outgrid_full = []
        it_drop_full = []
        phi2_kxky_full = []

        # Loop over (it,ikx,iky) chosen by the user
        for iplot in range(Nplot):

            # Time against which we plot
            t_plot = []
            t_plot.append(t[it_start_list[iplot]])

            # My ky
            iky = iky_list[iplot]
            ky = kygrid[iky]

            # Period of ExB remapping
            Tshift = abs(dkx/(g_exb*ky))

            # Time dependent radial wavenumber in lab frame
            kxstar = []
            # Grid point closest to kxstar
            kxbar = []

            # Starting at ...
            kxbar.append(kxgrid[ikx_start_list[iplot]])
            # ... which corresponds to the following wavenumber in shearing frame:
            kx = kxbar[0] + int(round(g_exb*ky*t[it_start_list[iplot]]/dkx))*dkx
            # ... and to the following time dependent wavenumber in lab frame:
            kxstar.append(kx-g_exb*ky*t[it_start_list[iplot]])

            # Mod. sq. of Fourier component phi(kx,ky):
            phi2_kxky = []
            phi2_kxky.append(phi2_by_mode[it_start_list[iplot],iky,ikx_start_list[iplot]])

            # At what time was this Fourier coefficient included in the sim ?
            if kx >= kxgrid.min() and kx <= kxgrid.max(): # Already in at the start
                t_ingrid = 0.
            else:
                if g_exb > 0.: # Enters from high kx end
                    t_ingrid = (kx-(kxgrid.max()+0.5*dkx))/(g_exb*ky)
                if g_exb < 0.: # Enters from low ky end
                    t_ingrid = (kx-(kxgrid.min()-0.5*dkx))/(g_exb*ky)

            # At what time was this Fourier coefficient dropped from the sim ?
            if g_exb > 0.: # Leaves through low kx end
                t_outgrid = (kx-(kxgrid.min()-0.5*dkx))/(g_exb*ky)
            if g_exb < 0.: # Leaves through high ky end
                t_outgrid = (kx-(kxgrid.max()+0.5*dkx))/(g_exb*ky)

            # At what time is kxstar=0 ?
            if kx*g_exb < 0:
                t_zero = np.nan # will never cross zero
            else:
                t_zero = kx/(g_exb*ky)

            # Compute corresponding it_drop (might be larger than t.size if it does not drop within simulation time)
            if t_outgrid < t.max():
                it_drop = int(np.ceil((t_outgrid+0.5*delt)/(nwrite*delt))) # add delt/2 because first step in GS2 uses dt/2
            else:
                it_drop = -1

            # Continue filling phi2_kxky by following it as it moves across kxgrid
            ikxgrid = ikx_start_list[iplot]
            if it_drop > 0 :
                it_max = it_drop
            else:
                it_max = t.size
            for it in range(it_start_list[iplot]+1,it_max):

                t_plot.append(t[it])

                # Compute new kxstar
                kxstar.append(kx-g_exb*ky*t[it])
                
                # Check if we now have a new nearest neighbour
                if abs(kxstar[-1]-kxbar[-1]) > 0.5*dkx:
                    kxbar.append(kxbar[-1] - sgn_g*dkx)
                    # In GS2, ExB remapping has shifted our Fourier coefficient -> update ikxgrid
                    ikxgrid = int(ikxgrid - sgn_g)
                else:
                    kxbar.append(kxbar[-1])

                phi2_kxky.append(phi2_by_mode[it,iky,ikxgrid])

            # End of t loop
        
            # Append computed quantities to _full arrays
            kx_full.append(kx)
            ky_full.append(ky)
            t_plot_full.append(t_plot)
            t_zero_full.append(t_zero)
            t_outgrid_full.append(t_outgrid)
            it_drop_full.append(it_drop)
            phi2_kxky_full.append(phi2_kxky)

        # End of iplot loop

        # Save quantities to a dat-file
        datfile_name = run.out_dir + run.fnames[ifile] + '.potential.dat'
        mydict = {'g_exb':g_exb,
                'kx_full':kx_full,'ky_full':ky_full,
                't_plot_full':t_plot_full,'t_zero_full':t_zero_full,'t_outgrid_full':t_outgrid_full,
                'it_drop_full':it_drop_full,'phi2_kxky_full':phi2_kxky_full}
        with open(datfile_name,'wb') as datfile:
            pickle.dump(mydict,datfile)

    # End if not only_plot

    ################
    ### Plotting ###
    ################

    if not run.no_plot:

        # If only plot, read quantities from dat-file
        if run.only_plot:

            datfile_name = run.out_dir + 'ollie_badshear_old_id_1.potential.dat'
            with open(datfile_name,'rb') as datfile:
                mydict = pickle.load(datfile)

            g_exb = mydict['g_exb']
            kx_full = mydict['kx_full']
            ky_full = mydict['ky_full']
            Nplot = len(kx_full)

            t_plot_full_old = mydict['t_plot_full']
            t_zero_full_old = mydict['t_zero_full']
            t_outgrid_full_old = mydict['t_outgrid_full']
            it_drop_full_old = mydict['it_drop_full']
            phi2_kxky_full_old = mydict['phi2_kxky_full']

            datfile_name = run.out_dir + 'ollie_badshear_fexp_id_1.potential.dat'
            with open(datfile_name,'rb') as datfile:
                mydict = pickle.load(datfile)

            t_plot_full_new = mydict['t_plot_full']
            phi2_kxky_full_new = mydict['phi2_kxky_full']

        # Start pdf list to merge at the end
        tmp_pdf_id = 1
        pdflist = []

        for iplot in range(Nplot):
            
            # Pick iplot elements from the _full arrays
            kx = kx_full[iplot]
            ky = ky_full[iplot]

            t_plot_old = t_plot_full_old[iplot]
            t_zero_old = t_zero_full_old[iplot]
            t_outgrid_old = t_outgrid_full_old[iplot]
            it_drop_old = it_drop_full_old[iplot]
            phi2_kxky_old = phi2_kxky_full_old[iplot]

            t_plot_new = t_plot_full_new[iplot]
            phi2_kxky_new = phi2_kxky_full_new[iplot]

            fig = plt.figure(figsize=(12,8))
            
            # Draw vertical line where kxstar=0
            props = dict(boxstyle='square', facecolor='white',edgecolor='white')
            if t_zero_old >= min(t_plot_old) and t_zero_old <= max(t_plot_old):
                plt.axvline(x=t_zero_old,color='grey',linestyle='-',linewidth=2)
            
            # Draw vertical line where kx is dropped from the sim
            if it_drop_old > 0 :
                plt.axvline(x=t_outgrid,color='k',linestyle='--',linewidth=2)

            # Plot phi_kxky vs t
            xlab = '$t [L/v_{th,i}]$'
            ylab = '$\\langle\\vert\\hat{\\varphi}_{k}\\vert ^2\\rangle_{\\theta}$'
            my_title = '$k_x='+'{:4.2f}'.format(kx)+',k_y='+'{:4.2f}'.format(ky)+'$'
            
            my_legend_old = 'nearest grid point'
            my_color_old = 'b'
            my_curve_old, = plt.semilogy(t_plot_old,phi2_kxky_old,linewidth=3.0, \
                    color=my_color_old)
            
            my_legend_new = 'continuous'
            my_color_new = 'r'
            my_curve_new, = plt.semilogy(t_plot_new,phi2_kxky_new,linewidth=3.0, \
                    color=my_color_new)

            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.grid(True)
            plt.ylim([0.01,1])
            plt.title(my_title,fontsize=24)

            my_legend = plt.legend([my_curve_old,my_curve_new],[my_legend_old,my_legend_new],\
                    frameon=True,fancybox=False,framealpha=1.0,loc='upper left')
            my_legend.get_frame().set_facecolor('w')
            my_legend.get_frame().set_edgecolor('k')
            my_legend.get_frame().set_linewidth(1.0)

            # Add textbox
            ax = plt.gca()
            xmin, xmax = ax.get_xlim()
            txt_ypos = 0.05 # Place text boxes at bottom
            txt_xpos = 0.5*(1.-(max(t_plot_old)-min(t_plot_old))/(xmax-xmin)) + (t_zero_old-min(t_plot_old))/(xmax-xmin)
            txt_str = '$k_x^*=0$'
            ax.text(txt_xpos, txt_ypos, txt_str, transform=ax.transAxes, fontsize=28, bbox=props,
                    horizontalalignment='center')
            # Add textbox
            txt_xpos = 0.5*(1.-(max(t_plot_old)-min(t_plot_old))/(xmax-xmin)) + (t_outgrid_old-min(t_plot_old))/(xmax-xmin)
            if g_exb > 0.:
                txt_str = '$k_x^*=\\min(k_{x,GS2})$'
            if g_exb < 0.:
                txt_str = '$k_x^*=\\max(k_{x,GS2})$'
            ax.text(txt_xpos, txt_ypos, txt_str, transform=ax.transAxes, fontsize=20, bbox=props,
                    horizontalalignment='right')
            
            # Save tmp plot and append to list for merge
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1

        # End of iplot loop
        
        # Merge pdfs
        merged_pdfname = 'potential'
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

    # End if not no_plot

    ####################
    ### End plotting ###
    ####################

# End of my_task_single
