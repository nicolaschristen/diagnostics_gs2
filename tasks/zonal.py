import numpy as np
import scipy.special as sp
import numpy.fft as fft
import matplotlib.pyplot as plt
from matplotlib import rcParams

import gs2_plotting as gplot
from plot_zonal_phikx_t import plot_zonal_phi_kx_vs_t


class zonalobj:

    def __init__(self, mygrids, mytime, myfields):
            
        ( self.zonal_phi, self.flow, self.shear,
            self.phi_gyro, self.flow_gyro, self.shear_gyro,
            self.flow_gyro_xt, self.shear_gyro_xt, self.dens_xt, self.upar_xt, self.tpar_xt, self.tperp_xt,
            self.flow_gyro_avg, self.shear_gyro_avg, self.flow_gyro_xt_avg, self.shear_gyro_xt_avg,
            self.zphi2, self.zflow2, self.zshear2, self.zgphi2, self.zgflow2,
            self.zgshear2 ) = self.get_attr(mygrids, mytime, myfields)

    def get_attr(self, mygrids, mytime, myfields):

        kx = mygrids.kx
        kx_gs2 = mygrids.kx_gs2
        xgrid = mygrids.xgrid
        nx = mygrids.nx

        time = mytime.time
        timeavg = mytime.timeavg
        ntime = mytime.ntime

        phi = myfields.phi
        phi_gs2 = myfields.phi_gs2
        dens_gs2 = myfields.dens_gs2
        upar_gs2 = myfields.upar_gs2
        tpar_gs2 = myfields.tpar_gs2
        tperp_gs2 = myfields.tperp_gs2

        # first want to compute real-space quantities
        # so use GS2-ordering for kx (same as FFT convention)
        # phi is ky=0 component of phi
        zonal_phi = phi_gs2[:,0,:]
        dens = np.copy(dens_gs2[:,:,0,:])
        upar = np.copy(upar_gs2[:,:,0,:])
        tpar = np.copy(tpar_gs2[:,:,0,:])
        tperp = np.copy(tperp_gs2[:,:,0,:])

        # next calculate phi, flow and shear
        # as well as gyro-averaged versions
        # in Fourier space
        phi_gyro=np.copy(zonal_phi)
        flow=np.copy(zonal_phi)
        flow_gyro=np.copy(zonal_phi)
        shear=np.copy(zonal_phi)
        shear_gyro=np.copy(zonal_phi)

        for idx1 in range(flow.shape[0]):
            for idx2 in range(flow.shape[1]):
                flow[idx1,idx2]=1j*kx_gs2[idx2]*zonal_phi[idx1,idx2]
                flow_gyro[idx1,idx2]=sp.j0(kx_gs2[idx2])*flow[idx1,idx2]
                shear[idx1,idx2]=-kx_gs2[idx2]*kx_gs2[idx2]*zonal_phi[idx1,idx2]
                shear_gyro[idx1,idx2]=-sp.j0(kx_gs2[idx2])*shear[idx1,idx2]

        # Fourier transform to get component of phi in real space
        flow_gyro_xt = np.real(fft.ifft(flow_gyro,axis=1))
        shear_gyro_xt = np.real(fft.ifft(shear_gyro,axis=1))
        dens_xt = np.real(fft.ifft(dens,axis=2))
        upar_xt = np.real(fft.ifft(upar,axis=2))
        tpar_xt = np.real(fft.ifft(tpar,axis=2))
        tperp_xt = np.real(fft.ifft(tperp,axis=2))

        flow_gyro_xt_avg = np.zeros(flow_gyro_xt.shape[1])
        shear_gyro_xt_avg = np.copy(flow_gyro_xt_avg)

        for idx in range(flow_gyro_xt.shape[1]):
            flow_gyro_xt_avg[idx] = timeavg(flow_gyro_xt[:,idx])
            shear_gyro_xt_avg[idx] = timeavg(shear_gyro_xt[:,idx])

        # now recalculate quantities using monotonic kx grid for
        # Fourier space plots
        zonal_phi = phi[:,0,:]

        for idx1 in range(flow.shape[0]):
            for idx2 in range(flow.shape[1]):
                phi_gyro[idx1,idx2]=sp.j0(kx[idx2])*zonal_phi[idx1,idx2]
                flow[idx1,idx2]=1j*kx[idx2]*zonal_phi[idx1,idx2]
                flow_gyro[idx1,idx2]=sp.j0(kx[idx2])*flow[idx1,idx2]
                shear[idx1,idx2]=-kx[idx2]*kx[idx2]*zonal_phi[idx1,idx2]
                shear_gyro[idx1,idx2]=-sp.j0(kx[idx2])*shear[idx1,idx2]

        flow_gyro_avg = np.zeros(nx)
        shear_gyro_avg = np.copy(flow_gyro_avg)

        for idx in range(flow.shape[1]):
            flow_gyro_avg[idx] = timeavg(np.abs(flow_gyro[:,idx])**2)
            shear_gyro_avg[idx] = timeavg(np.abs(shear_gyro[:,idx])**2)

        # get squared modulus of gyro-averaged fields
        zphi2 = np.arange(ntime*nx).reshape(ntime,nx)
        zphi2 = np.abs(zonal_phi)**2
        zgphi2 = np.arange(ntime*nx).reshape(ntime,nx)
        zgphi2 = np.abs(phi_gyro)**2
        zflow2 = np.arange(ntime*nx).reshape(ntime,nx)
        zflow2 = np.abs(flow)**2
        zgflow2 = np.arange(ntime*nx).reshape(ntime,nx)
        zgflow2 = np.abs(flow_gyro)**2
        zshear2 = np.arange(ntime*nx).reshape(ntime,nx)
        zshear2 = np.abs(shear)**2
        zgshear2 = np.arange(ntime*nx).reshape(ntime,nx)
        zgshear2 = np.abs(shear_gyro)**2

        return ( zonal_phi, flow, shear,
                phi_gyro, flow_gyro, shear_gyro,
                flow_gyro_xt, shear_gyro_xt, dens_xt, upar_xt, tpar_xt, tperp_xt,
                flow_gyro_avg, shear_gyro_avg, flow_gyro_xt_avg, shear_gyro_xt_avg,
                zphi2, zflow2, zshear2, zgphi2, zgflow2, zgshear2 )

    
    def plot(self, ifile, run, myout, mygrids, mytime):

        print()
        print("producing time-dependent zonal flow plots...",end='')

        tmp_pdf_id = 1
        pdflist = []

        plot_zonal_phi_kx_vs_t(mygrids, mytime, np.real(self.zonal_phi))
        tmp_pdfname = 'tmp'+tmp_pdf_id
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdf_name)
        tmp_pdf_id = tmp_pdf_id+1

        if (mygrids.nx > 1):

            # plot flow and flow shear vs (kx,t) and vs (x,t)
            title='$| J_{0}(k_{x}\\rho)k_{x}\\rho\Phi_{zf}(k_{x},t) |^2$'
            plot_zonal_vs_kxt(mygrids.kx,mytime.time,self.flow_gyro,title)
            tmp_pdfname = 'tmp'+tmp_pdf_id
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdf_name)
            tmp_pdf_id = tmp_pdf_id+1

            title='$| J_0(k_x\\rho)k_{x}^2\\rho^2\Phi_{zf}(k_{x},t) |^2$'
            plot_zonal_vs_kxt(mygrids.kx,mytime.time,self.shear_gyro,title)
            tmp_pdfname = 'tmp'+tmp_pdf_id
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdf_name)
            tmp_pdf_id = tmp_pdf_id+1

            title='$\left< \partial_x \Phi_{zf} (x,t) \\right> $'
            plot_zonal_vs_xt(mygrids.xgrid,mytime.time,self.flow_gyro_xt,title)
            tmp_pdfname = 'tmp'+tmp_pdf_id
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdf_name)
            tmp_pdf_id = tmp_pdf_id+1

            title='$\left<\partial_x^2 \Phi_{zf}(x,t)\\right>$'
            plot_zonal_vs_xt(mygrids.xgrid,mytime.time,self.shear_gyro_xt,title)
            tmp_pdfname = 'tmp'+tmp_pdf_id
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdf_name)
            tmp_pdf_id = tmp_pdf_id+1

            if myout['ntot_igomega_by_mode_present']:
                for ispec in range(myout['nspec']):
                    stitle=str(ispec+1) + ',zf} (x,t)$'
                    title = '$\delta n_{' + stitle
                    plot_zonal_vs_xt(mygrids.xgrid,mytime.time,self.dens_xt[:,ispec,:],title)
                    tmp_pdfname = 'tmp'+tmp_pdf_id
                    gplot.save_plot(tmp_pdfname, run, ifile)
                    pdflist.append(tmp_pdf_name)
                    tmp_pdf_id = tmp_pdf_id+1
                    
            if myout['upar_igomega_by_mode_present']:
                for ispec in range(myout['nspec']):
                    stitle=str(ispec+1) + ',zf} (x,t)$'
                    title = '$\delta u_{\parallel' + stitle
                    plot_zonal_vs_xt(mygrids.xgrid,mytime.time,self.upar_xt[:,ispec,:],title)
                    tmp_pdfname = 'tmp'+tmp_pdf_id
                    gplot.save_plot(tmp_pdfname, run, ifile)
                    pdflist.append(tmp_pdf_name)
                    tmp_pdf_id = tmp_pdf_id+1

            if myout['tpar_igomega_by_mode_present']:
                for ispec in range(myout['nspec']):
                    stitle=str(ispec+1) + ',zf} (x,t)$'
                    title = '$\delta T_{\parallel' + stitle
                    plot_zonal_vs_xt(mygrids.xgrid,mytime.time,self.tpar_xt[:,ispec,:],title)
                    tmp_pdfname = 'tmp'+tmp_pdf_id
                    gplot.save_plot(tmp_pdfname, run, ifile)
                    pdflist.append(tmp_pdf_name)
                    tmp_pdf_id = tmp_pdf_id+1

            if myout['tperp_igomega_by_mode_present']:
                for ispec in range(myout['nspec']):
                    stitle=str(ispec+1) + ',zf} (x,t)$'
                    title = '$\delta T_{\perp' + stitle
                    plot_zonal_vs_xt(mygrids.xgrid,mytime.time,self.tperp_xt[:,ispec,:],title)
                    tmp_pdfname = 'tmp'+tmp_pdf_id
                    gplot.save_plot(tmp_pdfname, run, ifile)
                    pdflist.append(tmp_pdf_name)

        # save plots
        merged_pdfname = 'zonal_vs_time'
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

        print('complete')

        if (mygrids.nx > 1):

            print()
            print("producing time-averaged zonal flow plots...",end='')

            tmp_pdf_id = 1
            pdflist = []

            title = '$| J_{0}(k_{x}\\rho)k_{x}\\rho\Phi_{zf}(k_{x}) |^2$'
            plot_zonal_vs_kx(mygrids.kx,self.flow_gyro_avg,title)
            tmp_pdfname = 'tmp'+tmp_pdf_id
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdf_name)
            tmp_pdf_id = tmp_pdf_id+1

            title = '$| J_0(k_x\\rho)k_{x}^2\\rho^2\Phi_{zf}(k_{x}) |^2$'
            plot_zonal_vs_kx(mygrids.kx,self.shear_gyro_avg,title)
            tmp_pdfname = 'tmp'+tmp_pdf_id
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdf_name)
            tmp_pdf_id = tmp_pdf_id+1

            title = '$\left< \partial_x \Phi_{zf} (x) \\right> $'
            plot_zonal_vs_x(mygrids.xgrid,self.flow_gyro_xt_avg,title)
            tmp_pdfname = 'tmp'+tmp_pdf_id
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdf_name)
            tmp_pdf_id = tmp_pdf_id+1

            title = '$\left<\partial_x^2 \Phi_{zf}(x)\\right>$'
            plot_zonal_vs_x(mygrids.xgrid,self.shear_gyro_xt_avg,title)
            tmp_pdfname = 'tmp'+tmp_pdf_id
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdf_name)

            # save plots
            merged_pdfname = 'zonal_steady'
            gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
            
            print('complete')
            print()


def plot_zonal_vs_kxt(kx, time, field, title):  ### Evolution of flow in fourier space versus time and kx

    z = (np.abs(field)**2)/(np.max(np.abs(field)**2))
    xlab = '$k_{x}\\rho_{i}$'
    ylab = '$t (v_{t}/a)$'
    cmap = 'YlGnBu'
    fig = gplot.plot_2d(z,kx,time,z.min(),z.max(),xlab,ylab,title,cmap)
    return fig

def plot_zonal_vs_xt(xgrid, time, field, title): ### Evolution of flow in real space versus time and x

    # ensure there is no divide by zero
#    denom = np.max(np.abs(field))
#    test = denom==0
#    denom[test] = 1
    z = field
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    xlab = '$x/\\rho_{i}$'
    ylab = '$t (v_{t}/a)$'
    cmap = 'RdBu'
    fig = gplot.plot_2d(z,xgrid,time,z_min,z_max,xlab,ylab,title,cmap)
    return fig

def plot_zonal_vs_kx(kx,field,title):

    xlab = '$k_{x}\\rho_{i}$'
    fig = gplot.plot_1d(kx,field,xlab,title=title)
    return fig

def plot_zonal_vs_x(xgrid,field,title):

#    xlab = '$x/\\rho_{i}$'
#    fig = gplot.plot_1d(xgrid,field,xlab,title=title)
#    return fig

    from scipy.integrate import simps

    dum = np.empty(xgrid.size)
    fldavg = simps(field,x=xgrid)/(xgrid.max()-xgrid.min())
    dum.fill(fldavg)
    fig = plt.figure(figsize=(12,8))
    plt.plot(xgrid,field)
    plt.plot(xgrid,dum)
    xlab = '$x/\\rho_{i}$'
    plt.xlabel(xlab)
    plt.title(title)
    return fig
