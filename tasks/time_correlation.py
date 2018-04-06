import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.integrate import simps
from scipy.fftpack import rfft, fftshift

import gs2_plotting as gplot
import gs2_fields as gfields

class tcorrobj:

    def __init__(self, mygrids, mytime, myfields, myzonal):

        ( self.time_corrint, self.it_halfint, self.freq, self.zgflow_Cinf, self.zgshear_Cinf,
            self.Ex_corr_time, self.Ey_corr_time,
            self.Ex_time_corrfnc, self.Ey_time_corrfnc, self.Ex_freq_spectrum, self.Ey_freq_spectrum,
            self.zphi_time_corrfnc, self.zflow_time_corrfnc, self.zshear_time_corrfnc,
            self.zgphi_time_corrfnc, self.zgflow_time_corrfnc, self.zgshear_time_corrfnc,
            self.zflow2_avg ) = self.get_attr(mygrids, mytime, myfields, myzonal)

    def get_attr(self, mygrids, mytime, myfields, myzonal):

        # get half number of time points in steady-state interval
        it_halfint = mytime.it_interval//2
        # get midpoint time index for steady-state
        it_mid = mytime.it_min + it_halfint
        # get final time index (might be a bit below ntime)
        it_final = it_mid + it_halfint - 1
        
        # get time values for first half of steady-state interval
        time_corrint = np.arange(it_halfint,dtype=float)
        time_corrint = mytime.time[mytime.it_min:it_mid]-mytime.time[mytime.it_min]
        
        zphi2_avg = np.arange(mygrids.nx,dtype=float)
        zphi2_avg = np.sum(simps(myzonal.zphi2[mytime.it_min:it_mid,:],x=time_corrint,axis=0),axis=0)
        zgphi2_avg = np.arange(mygrids.nx,dtype=float)
        zgphi2_avg = np.sum(simps(myzonal.zgphi2[mytime.it_min:it_mid,:],x=time_corrint,axis=0),axis=0)
        # this is |kx*phi(ky=0)|**2 integrated over
        # first half of steady-state time interval
        zflow2_avg = np.arange(mygrids.nx,dtype=float)
        zflow2_avg = np.sum(simps(myzonal.zflow2[mytime.it_min:it_mid,:],x=time_corrint,axis=0),axis=0)
        zgflow2_avg = np.arange(mygrids.nx,dtype=float)
        zgflow2_avg = np.sum(simps(myzonal.zgflow2[mytime.it_min:it_mid,:],x=time_corrint,axis=0),axis=0)
        # this is |kx^2*phi(ky=0)|**2 integrated over
        # first half of steady-state time interval
        zshear2_avg = np.arange(mygrids.nx,dtype=float)
        zshear2_avg = np.sum(simps(myzonal.zshear2[mytime.it_min:it_mid,:],x=time_corrint,axis=0),axis=0)
        zgshear2_avg = np.arange(mygrids.nx,dtype=float)
        zgshear2_avg = np.sum(simps(myzonal.zgshear2[mytime.it_min:it_mid,:],x=time_corrint,axis=0),axis=0)
        
        # this is Ex**2 and Ey**2 integrated over
        # first half of steady-state time interval
        Ex2_tavg = np.arange(mygrids.nx*mygrids.ny,dtype=float).reshape(mygrids.ny,mygrids.nx)
        Ex2_tavg = simps(myfields.Ex2[mytime.it_min:it_mid,:,:],x=time_corrint,axis=0)
        Ey2_tavg = np.arange(mygrids.nx*mygrids.ny,dtype=float).reshape(mygrids.ny,mygrids.nx)
        Ey2_tavg = simps(myfields.Ey2[mytime.it_min:it_mid,:,:],x=time_corrint,axis=0)
        # sum over kx and ky
        Ex2_avg = np.sum(Ex2_tavg)
        Ey2_avg = np.sum(Ey2_tavg)
        
        Ex_corrfnc_t = np.arange(it_halfint*mygrids.ny*mygrids.nx,dtype=complex).reshape(it_halfint,mygrids.ny,mygrids.nx)
        Ey_corrfnc_t = np.arange(it_halfint*mygrids.ny*mygrids.nx,dtype=complex).reshape(it_halfint,mygrids.ny,mygrids.nx)
        zphi_corrfnc_t = np.arange(it_halfint*mygrids.nx,dtype=complex).reshape(it_halfint,mygrids.nx)
        zflow_corrfnc_t = np.arange(it_halfint*mygrids.nx,dtype=complex).reshape(it_halfint,mygrids.nx)
        zshear_corrfnc_t = np.arange(it_halfint*mygrids.nx,dtype=complex).reshape(it_halfint,mygrids.nx)
        zgphi_corrfnc_t = np.arange(it_halfint*mygrids.nx,dtype=complex).reshape(it_halfint,mygrids.nx)
        zgflow_corrfnc_t = np.arange(it_halfint*mygrids.nx,dtype=complex).reshape(it_halfint,mygrids.nx)
        zgshear_corrfnc_t = np.arange(it_halfint*mygrids.nx,dtype=complex).reshape(it_halfint,mygrids.nx)
        
        # need to multiply second argument of np.correlate by weights for time integration
        for ikx in range(mygrids.nx):
            zphi_corrfnc_t[:,ikx] = \
                np.correlate(myzonal.zonal_phi[mytime.it_min:it_final,ikx], 
                             myzonal.zonal_phi[mytime.it_min:it_mid,ikx]*mytime.wgts_time[mytime.it_min:it_mid],mode='valid')
            zflow_corrfnc_t[:,ikx] = \
                np.correlate(myzonal.flow[mytime.it_min:it_final,ikx], 
                             myzonal.flow[mytime.it_min:it_mid,ikx]*mytime.wgts_time[mytime.it_min:it_mid],mode='valid')
            zshear_corrfnc_t[:,ikx] = \
                np.correlate(myzonal.shear[mytime.it_min:it_final,ikx], 
                             myzonal.shear[mytime.it_min:it_mid,ikx]*mytime.wgts_time[mytime.it_min:it_mid],mode='valid')
            zgphi_corrfnc_t[:,ikx] = \
                np.correlate(myzonal.phi_gyro[mytime.it_min:it_final,ikx], 
                             myzonal.phi_gyro[mytime.it_min:it_mid,ikx]*mytime.wgts_time[mytime.it_min:it_mid],mode='valid')
            zgflow_corrfnc_t[:,ikx] = \
                np.correlate(myzonal.flow_gyro[mytime.it_min:it_final,ikx], 
                             myzonal.flow_gyro[mytime.it_min:it_mid,ikx]*mytime.wgts_time[mytime.it_min:it_mid],mode='valid')
            zgshear_corrfnc_t[:,ikx] = \
                np.correlate(myzonal.shear_gyro[mytime.it_min:it_final,ikx], 
                             myzonal.shear_gyro[mytime.it_min:it_mid,ikx]*mytime.wgts_time[mytime.it_min:it_mid],mode='valid')
            for iky in range(mygrids.ny):
                Ex_corrfnc_t[:,iky,ikx] = \
                    np.correlate(myfields.Ex[mytime.it_min:it_final,iky,ikx], 
                                 myfields.Ex[mytime.it_min:it_mid,iky,ikx]*mytime.wgts_time[mytime.it_min:it_mid],mode='valid')
                Ey_corrfnc_t[:,iky,ikx] = \
                    np.correlate(myfields.Ey[mytime.it_min:it_final,iky,ikx], 
                                 myfields.Ey[mytime.it_min:it_mid,iky,ikx]*mytime.wgts_time[mytime.it_min:it_mid],mode='valid')
        
        Ex_time_corrfnc = np.arange(it_halfint,dtype=float)
        Ey_time_corrfnc = np.arange(it_halfint,dtype=float)
        for i in range(it_halfint):
            Ex_time_corrfnc[i] = np.real(np.sum(Ex_corrfnc_t[i,:,:]))/Ex2_avg
            Ey_time_corrfnc[i] = np.real(np.sum(Ey_corrfnc_t[i,:,:]))/Ey2_avg
        
        Ex_freq_spectrum = np.arange(it_halfint,dtype=float)
        Ey_freq_spectrum = np.arange(it_halfint,dtype=float)
        Ex_freq_spectrum = fftshift(rfft(Ex_time_corrfnc)/it_halfint)
        Ey_freq_spectrum = fftshift(rfft(Ey_time_corrfnc)/it_halfint)
        freq = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(it_halfint,mytime.time[mytime.ntime-1]-mytime.time[mytime.ntime-2]))
        
        ## normalized time correlation function vs ky (based on phi)
        #time_corrfnc_y = np.arange(ny*it_halfint,dtype=float).reshape(it_halfint,ny)
        #for i in range(ny):
        #    time_corrfnc_y[:,i] = np.real(np.sum(Ex_corrfnc_t[:,i,:],axis=1))/Ex2_avg_ky[i]
        # normalized time correlation function for zonal mode based on gyro-averaged flow
        zphi_time_corrfnc = np.arange(it_halfint,dtype=float)
        zphi_time_corrfnc = np.real(np.sum(zphi_corrfnc_t,axis=1))/zphi2_avg
        zflow_time_corrfnc = np.arange(it_halfint,dtype=float)
        zflow_time_corrfnc = np.real(np.sum(zflow_corrfnc_t,axis=1))/zflow2_avg
        zshear_time_corrfnc = np.arange(it_halfint,dtype=float)
        zshear_time_corrfnc = np.real(np.sum(zshear_corrfnc_t,axis=1))/zshear2_avg
        zgphi_time_corrfnc = np.arange(it_halfint,dtype=float)
        zgphi_time_corrfnc = np.real(np.sum(zgphi_corrfnc_t,axis=1))/zgphi2_avg
        zgflow_time_corrfnc = np.arange(it_halfint,dtype=float)
        zgflow_time_corrfnc = np.real(np.sum(zgflow_corrfnc_t,axis=1))/zgflow2_avg
        zgshear_time_corrfnc = np.arange(it_halfint,dtype=float)
        zgshear_time_corrfnc = np.real(np.sum(zgshear_corrfnc_t,axis=1))/zgshear2_avg
        
        #zgflow_frequency_spectrum = np.arange(it_halfint,dtype=float)
        #zgflow_frequency_spectrum = fftshift(fft(zgflow_time_corrfnc)/it_halfint)
        #frequency = np.fft.fftshift(np.fft.fftfreq(it_halfint,time[ntime-1]-time[ntime-2]))
        
        # correlation time (based on Ex,Ey)
        Ex_corr_time = np.abs(simps(Ex_time_corrfnc,x=time_corrint,axis=0))
        Ey_corr_time = np.abs(simps(Ey_time_corrfnc,x=time_corrint,axis=0))
        
        # calculate correlation function for zonal flow at t -> infinity
        zgflow_Cinf = simps(zgflow_time_corrfnc[it_halfint//2:it_halfint-1],
                      x=time_corrint[it_halfint//2:it_halfint-1]) \
                      / (time_corrint[it_halfint-1]-time_corrint[it_halfint//2])
        zgshear_Cinf = simps(zgshear_time_corrfnc[it_halfint//2:it_halfint-1],
                      x=time_corrint[it_halfint//2:it_halfint-1]) \
                      / (time_corrint[it_halfint-1]-time_corrint[it_halfint//2])
        
        #zonal_power_spectrum1 = np.arange(ntime_steady,dtype=float)
        #zonal_power_spectrum1 = np.sum(np.abs(np.fft.fft(zonal_E[it_min:,:],axis=0))**2,axis=1)/ntime_steady
        #zflow_time_corrfnc_2 = np.fft.ifftshift(np.real(np.fft.ifft(zonal_power_spectrum1))/(2*np.pi))
        #zflow_time_corrfnc_2 = zflow_time_corrfnc_2/zflow_time_corrfnc_2[0]
        #zonal_power_spectrum1 = np.fft.fftshift(zonal_power_spectrum1)
        
        return ( time_corrint, it_halfint, freq, zgflow_Cinf, zgshear_Cinf,
                Ex_corr_time, Ey_corr_time, Ex_time_corrfnc, Ey_time_corrfnc, Ex_freq_spectrum, Ey_freq_spectrum,
                zphi_time_corrfnc, zflow_time_corrfnc, zshear_time_corrfnc,
                zgphi_time_corrfnc, zgflow_time_corrfnc, zgshear_time_corrfnc, zflow2_avg )


    def plot(self, ifile, run, mytime, myfields, mytxt):

        write_correlation_times(mytxt, self)

        tmp_pdf_id = 1
        pdflist = []
        
        plot_timecorrfnc_nonzonal(self)
        tmp_pdfname = 'tmp'+tmp_pdf_id
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdf_name)
        tmp_pdf_id = tmp_pdf_id+1
        
        plot_timecorrfnc_zonal(self)
        tmp_pdfname = 'tmp'+tmp_pdf_id
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdf_name)
        tmp_pdf_id = tmp_pdf_id+1
        
        plot_timecorrfnc_gyrozonal(self)
        tmp_pdfname = 'tmp'+tmp_pdf_id
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdf_name)
        tmp_pdf_id = tmp_pdf_id+1
        
        plot_nonzonal_freq_spectrum(mytime, myfields)
        tmp_pdfname = 'tmp'+tmp_pdf_id
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdf_name)
        tmp_pdf_id = tmp_pdf_id+1
        
        plot_zonal_power_spectrum(mytime, myfields, self)
        tmp_pdfname = 'tmp'+tmp_pdf_id
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdf_name)
        tmp_pdf_id = tmp_pdf_id+1
        
        gfields.plot_power_spectrum(myfields, mytime)
        tmp_pdfname = 'tmp'+tmp_pdf_id
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdf_name)
        tmp_pdf_id = tmp_pdf_id+1

        merged_pdfname = 'time_correlation'
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

def write_correlation_times(mytxt, mytcorr):

    s = 'non-zonal E_x correlation time: '
    line = s + str(mytcorr.Ex_corr_time)
    print(line,file=mytxt)

    s = 'non-zonal E_y correlation time: '
    line = s + str(mytcorr.Ey_corr_time)
    print(line,file=mytxt)

    s = 'zonal E_x C(t_inf): '
    line = s + str(mytcorr.zgflow_Cinf)
    print(line,file=mytxt)

    s = 'zonal d(E_x)/dx C(t_inf): '
    line = s + str(mytcorr.zgshear_Cinf)
    print(line,file=mytxt)

def plot_timecorrfnc_nonzonal(mytcorr):

    xlab = '$\Delta t (v_{t}/a)$'
    ylab = '$\mathcal{C}_{nz}(\Delta t)$'
    fig = plt.figure(figsize=(12,8))
    plt.plot(mytcorr.time_corrint,mytcorr.Ex_time_corrfnc,label='$E_{x}$')
    plt.plot(mytcorr.time_corrint,mytcorr.Ey_time_corrfnc,label='$E_{y}$')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    return fig

def plot_timecorrfnc_zonal(mytcorr):
    
#    global zflow_time_corrfnc_2

    xlab = '$\Delta t (v_{t}/a)$'
    ylab = '$\mathcal{C}_{zf}(\Delta t)$'
    fig = plt.figure(figsize=(12,8))
    plt.plot(mytcorr.time_corrint,mytcorr.zphi_time_corrfnc,label='phi')
    plt.plot(mytcorr.time_corrint,mytcorr.zflow_time_corrfnc,label='flow')
    plt.plot(mytcorr.time_corrint,mytcorr.zshear_time_corrfnc,label='shear')
#    plt.plot(time_steady,zflow_time_corrfnc_2)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    return fig

def plot_timecorrfnc_gyrozonal(mytcorr):

    xlab = '$\Delta t (v_{t}/a)$'
    ylab = '$<\mathcal{C}_{zf}(\Delta t)>$'
    fig = plt.figure(figsize=(12,8))
    plt.plot(mytcorr.time_corrint,mytcorr.zgphi_time_corrfnc,label='phi')
    plt.plot(mytcorr.time_corrint,mytcorr.zgflow_time_corrfnc,label='flow')
    plt.plot(mytcorr.time_corrint,mytcorr.zgshear_time_corrfnc,label='shear')
    tavg = mytcorr.time_corrint[mytcorr.it_halfint//2:mytcorr.it_halfint]
    dum = np.empty(tavg.size)
    dum.fill(mytcorr.zgflow_Cinf)
    plt.plot(tavg, dum)
    dum.fill(mytcorr.zgshear_Cinf)
    plt.plot(tavg, dum)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    return fig

def plot_nonzonal_freq_spectrum(mytime, myfields):
   
    Ex_power_spectrum = np.arange(mytime.ntime_steady,dtype=float)
    Ex_power_spectrum = np.fft.fftshift(np.sum( \
            np.abs(np.fft.fft(myfields.Ex_gs2[mytime.it_min:mytime.it_max,:,:],axis=0))**2, \
            axis=(1,2)))/mytime.ntime_steady
    fig = plt.figure(figsize=(12,8))
    plt.plot(mytime.frequency,Ex_power_spectrum,label='$E_x$')
#    plt.plot(freq,Ey_freq_spectrum,label='$E_y$')
    plt.xlabel('frequency')
    plt.title('power spectrum')
    return fig

def plot_zonal_power_spectrum(mytime, myfields, mytcorr):
    
    import numpy as np
    
    zonal_power_spectrum1 = np.arange(mytime.ntime_steady,dtype=float)
    zonal_power_spectrum1 = np.fft.fftshift(np.sum(np.abs(np.fft.fft( \
            myfields.zonal_E[mytime.it_min:mytime.it_max,:],axis=0))**2,axis=1))/mytime.ntime_steady
    zonal_power_spectrum2 = np.arange(mytcorr.it_halfint,dtype=float)
    zonal_power_spectrum2 = 2*np.pi*np.fft.fftshift(np.real(np.fft.fft(mytcorr.zflow_time_corrfnc))) \
            *mytcorr.zflow2_avg/mytcorr.it_halfint
    fig = plt.figure(figsize=(12,8))
    plt.plot(mytime.frequency,zonal_power_spectrum1)
    plt.plot(mytcorr.freq, zonal_power_spectrum2)
    plt.xlabel('frequency')
    plt.title('zonal power spectrum')
    return fig

#     global frequency, zgflow_frequency_spectrum

#     fig = plt.figure(figsize=(12,8))
#     plt.plot(frequency,zgflow_frequency_spectrum)
#     plt.xlabel('frequency')
#     plt.title('power spectrum')
#     return fig

# def plot_timecorrfnc():

#     global time_corrint, time_corrfnc_y
#     z_min, z_max = -np.abs(time_corrfnc_y).max(), np.abs(time_corrfnc_y).max()
#     xlab = '$k_{y} \\rho_{i}$'
#     ylab = '$\Delta t (v_{t}/a)$'
#     title = '$Re[\mathcal{C}(\Delta t)]$'
#     cmap = 'RdBu'
#     fig = plot_2d(time_corrfnc_y,ky,time_corrint,z_min,z_max,xlab,ylab,title,cmap)
#     return fig

# def plot_corrtime():

#     global Ex_corr_time, Ey_corr_time
#     xlab = '$k_{x} \\rho_{i}$'
#     ylab = '$k_{y} \\rho_{i}$'
#     title = '$\\tau_{C}$'
#     cmap = 'YlGnBu'
#     fig = plot_2d(corr_time,kx,ky,corr_time.min(),corr_time.max(),xlab,ylab,title,cmap)
