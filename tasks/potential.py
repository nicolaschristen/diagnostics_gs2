import sys
import numpy as np
import gs2_plotting as gplot
import matplotlib.pyplot as plt
import pickle

def my_task_single(ifile, run, myin, myout, mygrids):

    if not run.only_plot:
    
        #######################
        ### User parameters ###
        #######################
        
        # Choose which Fourier components to plot
        it_start_list = [0,0]
        ikx_start_list = [3,4] # ikxmax=126 for nx=192
        iky_list = [1,1]

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

            datfile_name = run.out_dir + run.fnames[ifile] + '.potential.dat'
            with open(datfile_name,'rb') as datfile:
                mydict = pickle.load(datfile)

            g_exb = mydict['g_exb']
            kx_full = mydict['kx_full']
            ky_full = mydict['ky_full']
            t_plot_full = mydict['t_plot_full']
            t_zero_full = mydict['t_zero_full']
            t_outgrid_full = mydict['t_outgrid_full']
            it_drop_full = mydict['it_drop_full']
            phi2_kxky_full = mydict['phi2_kxky_full']

            Nplot = len(kx_full)

        # Start pdf list to merge at the end
        tmp_pdf_id = 1
        pdflist = []

        for iplot in range(Nplot):
            
            # Pick iplot elements from the _full arrays
            kx = kx_full[iplot]
            ky = ky_full[iplot]
            t_plot = t_plot_full[iplot]
            t_zero = t_zero_full[iplot]
            t_outgrid = t_outgrid_full[iplot]
            it_drop = it_drop_full[iplot]
            phi2_kxky = phi2_kxky_full[iplot]

            fig = plt.figure(figsize=(12,8))

            # Plot phi_kxky vs t
            title = '$(k_x='+'{:4.2f}'.format(kx)+',k_y='+'{:4.2f}'.format(ky)+')$ from $t='+'{:4.2f}'.format(t_plot[0])+'$'
            xlab = '$t (a/v_{t})$'
            ylab = '$\\langle\\vert\\hat{\\varphi}_{k}\\vert ^2\\rangle_{\\theta}$'
            plt.semilogy(t_plot,phi2_kxky,linewidth=2)
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.title(title)
            plt.grid(True)
            #ax = plt.gca() # NDCDEL
            #ax.set_ylim([0.01, 1.5]) # NDCDEL
            
            # Draw vertical line where kxstar=0
            props = dict(boxstyle='square', facecolor='white',edgecolor='white')
            if t_zero >= min(t_plot) and t_zero <= max(t_plot):
                plt.axvline(x=t_zero,color='k',linestyle='--',linewidth=2)
            # Add textbox
            ax = plt.gca()
            xmin, xmax = ax.get_xlim()
            txt_ypos = 0.05 # Place text boxes at bottom
            txt_xpos = 0.5*(1.-(max(t_plot)-min(t_plot))/(xmax-xmin)) + (t_zero-min(t_plot))/(xmax-xmin)
            txt_str = '$k_x^*=0$'
            ax.text(txt_xpos, txt_ypos, txt_str, transform=ax.transAxes, fontsize=20, bbox=props,
                    horizontalalignment='center')
            
            # Draw vertical line where kx is dropped from the sim
            if it_drop > 0 :
                plt.axvline(x=t_outgrid,color='k',linestyle='--',linewidth=2)
            # Add textbox
            txt_xpos = 0.5*(1.-(max(t_plot)-min(t_plot))/(xmax-xmin)) + (t_outgrid-min(t_plot))/(xmax-xmin)
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
