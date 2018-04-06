from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np

import gs2_plotting as gplot
from plot_phi2_vs_time import plot_phi2_ky_vs_t

class fluxobj:
    
    def __init__(self, myout, mygrids, mytime):

        ( self.pflx, self.qflx, self.vflx, self.xchange,
                self.pflx_kxky_tavg, self.qflx_kxky_tavg, self.vflx_kxky_tavg,
                self.pflx_vpth_tavg, self.qflx_vpth_tavg, self.vflx_vpth_tavg,
                self.pioq ) = self.get_attr(myout, mygrids, mytime)


    def get_attr(self, myout, mygrids, mytime):

        nvpa = mygrids.nvpa
        ntheta = mygrids.ntheta
        nx = mygrids.nx
        ny = mygrids.ny
        nxmid = mygrids.nxmid

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

        if myout['es_part_by_k_present']:
            pflx_kxky = np.concatenate((myout['es_part_by_k'][:,:,:,nxmid:],myout['es_part_by_k'][:,:,:,:nxmid]),axis=3)
            pflx_kxky_tavg = np.arange(myout['nspec']*nx*ny,dtype=float).reshape(myout['nspec'],ny,nx)
            for ispec in range(myout['nspec']):
                for ik in range(ny):
                    for it in range(nx):
                        pflx_kxky_tavg[ispec,ik,it] = timeavg(pflx_kxky[:,ispec,ik,it])
        else:
            mydim = (mytime.ntime,myout['nspec'],mygrids.ny,mygrids.nx)
            pflx_kxky = np.zeros(mydim, dtype=float)
            mydim = (myout['nspec'],mygrids.ny,mygrids.nx)
            pflx_kxky_tavg = np.zeros(mydim, dtype=float)

        if myout['es_heat_by_k_present']:
            qflx_kxky = np.concatenate((myout['es_heat_by_k'][:,:,:,nxmid:],myout['es_heat_by_k'][:,:,:,:nxmid]),axis=3)
            qflx_kxky_tavg = np.copy(pflx_kxky_tavg)
            for ispec in range(myout['nspec']):
                for ik in range(ny):
                    for it in range(nx):
                        qflx_kxky_tavg[ispec,ik,it] = timeavg(qflx_kxky[:,ispec,ik,it])
        else:
            mydim = (mytime.ntime,myout['nspec'],mygrids.ny,mygrids.nx)
            qflx_kxky = np.zeros(mydim, dtype=float)
            mydim = (myout['nspec'],mygrids.ny,mygrids.nx)
            qflx_kxky_tavg = np.zeros(mydim, dtype=float)

        if myout['es_mom_by_k_present']:
            vflx_kxky = np.concatenate((myout['es_mom_by_k'][:,:,:,nxmid:],myout['es_mom_by_k'][:,:,:,:nxmid]),axis=3)
            vflx_kxky_tavg = np.copy(pflx_kxky_tavg)
            for ispec in range(myout['nspec']):
                for ik in range(ny):
                    for it in range(nx):
                        vflx_kxky_tavg[ispec,ik,it] = timeavg(vflx_kxky[:,ispec,ik,it])
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
                        pflx_vpth_tavg[ispec,iv,ig] = timeavg(pflx_vpth[:,ispec,iv,ig])
        
        qflx_vpth = myout['es_heat_sym']
        qflx_vpth_tavg = np.copy(pflx_vpth_tavg)
        if myout['es_heat_sym_present']:
            for ispec in range(myout['nspec']):
                for iv in range(nvpa):
                    for ig in range(ntheta):
                        qflx_vpth_tavg[ispec,iv,ig] = timeavg(qflx_vpth[:,ispec,iv,ig])
        
        vflx_vpth = myout['es_mom_sym'] 
        vflx_vpth_tavg = np.copy(pflx_vpth_tavg)
        if myout['es_mom_sym_present']:
            for ispec in range(myout['nspec']):
                for iv in range(nvpa):
                    for ig in range(ntheta):
                        vflx_vpth_tavg[ispec,iv,ig] = timeavg(vflx_vpth[:,ispec,iv,ig])

        return ( pflx, qflx, vflx, xchange,
                pflx_kxky_tavg, qflx_kxky_tavg, vflx_kxky_tavg,
                pflx_vpth_tavg, qflx_vpth_tavg, vflx_vpth_tavg,
                pioq )


    def plot(self, ifile, run, myin, myout, mygrids, mytime, myfields, mytxt):

        islin = myin['nonlinear_terms_knobs']['nonlinear_mode']
        nspec = myin['species_knobs']['nspec']
        naky = (myin['kt_grids_box_parameters']['ny']-1)//3 + 1
       
        ky = mygrids.ky

        time = mytime.time
        time_steady = mytime.time_steady
        it_min = mytime.it_min
        it_max = mytime.it_max
        
        phi2_avg = myfields.phi2_avg

        print()
        print("producing plots of fluxes vs time...", end='')

        print('---GS2 FLUXES---',file=mytxt)
        write_fluxes_vs_t = False
        tmp_pdf_id = 1
        pdflist = []
        if myout['phi2_present']:
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
        if myout['es_part_flux_present']:
            title = '$\Gamma_{GS2}$'
            plot_flux_vs_t(myin,mytime,self.pflx,title,mytxt)
            write_fluxes_vs_t = True
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
        if myout['es_heat_flux_present']:
            title = '$Q_{GS2}$'
            plot_flux_vs_t(myin,mytime,self.qflx,title,mytxt)
            write_fluxes_vs_t = True
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
        if myout['es_mom_flux_present']:
            title = '$\Pi_{GS2}$'
            plot_flux_vs_t(myin,mytime,self.vflx,title,mytxt)
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
        if myout['es_part_flux_present'] and myout['es_heat_flux_present']:
            title = '$\Pi_{GS2}/Q_{GS2}$'
            for idx in range(nspec):
                plt.plot(mytime.time_steady,self.pioq[it_min:it_max,idx],label=myin['species_parameters_'+str(idx+1)]['type'])
            plt.title(title)
            plt.xlabel('$t (a/v_t)$')
            plt.legend()
            plt.grid(True)
            write_fluxes_vs_t = True
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
        if myout['phi2_by_ky_present']:
            phi2_by_ky = myout['phi2_by_ky']
            title = '$\\langle|\phi^{2}|\\rangle_{\\theta,k_x}$'
            if islin:
                title = '$\\ln$' + title
                for iky in range(naky) :
                    plt.semilogy(time, np.log(phi2_by_ky[:,iky]),label='ky = '+'{:5.3f}'.format(ky[iky]))
            else:
                for iky in range(naky) :
                    plt.plot(time, np.log(phi2_by_ky[:,iky]),label='ky = '+'{:5.3f}'.format(ky[iky]))
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
            gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

        print('complete')
        
        print()
        print('producing plots of fluxes vs (kx,ky)...', end='')

        write_fluxes_vs_kxky = False
        tmp_pdf_id = 1
        pdflist = []
        if myout['es_part_by_k_present']:
            title = '$\Gamma_{GS2}$'
            plot_flux_vs_kxky(self.pflx_kxky_tavg,title)
            write_fluxes_vs_kxky = True
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
        if myout['es_heat_by_k_present']:
            title = '$Q_{GS2}$'
            plot_flux_vs_kxky(self.qflx_kxky_tavg,title)
            write_fluxes_vs_kxky = True
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
        if myout['es_mom_by_k_present']:
            title = '$\Pi_{GS2}$'
            plot_flux_vs_kxky(self.vflx_kxky_tavg,title)
            write_fluxes_vs_kxky = True
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)

        if write_fluxes_vs_kxky:
            merged_pdfname = 'fluxes_vs_kxky'
            gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

        print('complete')

        print()
        print('producing plots of fluxes vs (vpa,theta)...',end='')

        write_vpathetasym = False
        tmp_pdf_id = 1
        pdflist = []
        if myout['es_part_sym_present']:
            title = '$\Gamma_{GS2}$'
            plot_flux_vs_vpth(self.pflx_vpth_tavg,title)
            write_vpathetasym = True
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
        if myout['es_heat_sym_present']:
            title = '$Q_{GS2}$'
            plot_flux_vs_vpth(self.qflx_vpth_tavg,title)
            write_vpathetasym = True
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
        if myout['es_mom_sym_present']:
            title = '$\Pi_{GS2}$'
            plot_flux_vs_vpth(self.vflx_vpth_tavg,title)
            write_vpathetasym = True
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1

        if write_vpathetasym:  
            merged_pdfname = 'fluxes_vs_vpa_theta'
            gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

        print('complete')


def plot_flux_vs_t(myin,mytime,flx,title,mytxt):

    islin = myin['nonlinear_terms_knobs']['nonlinear_mode']
    nspec = myin['species_knobs']['nspec']

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
            plt.plot(mytime.time,np.log(flx[:,idx]),label=myin['species_parameters_'+str(idx+1)]['type'])
        else:
            plt.plot(mytime.time,flx[:,idx],label=myin['species_parameters_'+str(idx+1)]['type'])
    
    # plot time-averages
    for idx in range(nspec):
        if islin:
            flxavg = mytime.timeavg(np.log(np.absolute(flx[:,idx])))
            dum.fill(flxavg)
        else:
            flxavg = mytime.timeavg(flx[:,idx])
            dum.fill(flxavg)
        plt.plot(mytime.time_steady,dum,'--')
        line = title + '(is= ' + str(idx+1) + '): ' + str(flxavg)
        print(line,file=mytxt)

    plt.xlabel('$t (a/v_t)$')
    plt.xlim([mytime.time[0],mytime.time[mytime.ntime-1]])
    plt.title(title)
    plt.legend()
    plt.grid(True)

    return fig

def plot_flux_vs_kxky(mygrids,flx,title):

    from gs2_plotting import plot_2d

    xlab = '$k_{x}\\rho$'
    ylab = '$k_{y}\\rho$'
    cmap = 'RdBu'
    for idx in range(flx.shape[0]):
        z = flx[idx,:,:]
        z_min, z_max = z.min(), z.max()
        fig = plot_2d(z,mygrids.kx,mygrids.ky,z_min,z_max,xlab,ylab,title+' (is= '+str(idx+1)+')',cmap)

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