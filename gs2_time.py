import numpy as np
from scipy.integrate import simps
from numpy import fft
from math import ceil


class timeobj:

    def __init__(self, myout, run):

        print()
        print('calculating time grid...',end='')

        self.time = np.copy(myout['t'])
        self.ntime = self.time.size

        # get starting index for selected time window
        self.twin = run.twin
        tmin = self.time[-1]*self.twin[0]
        self.it_min = 0
        while self.time[self.it_min] < tmin:
            self.it_min += 1
        tmax = self.time[-1]*self.twin[1]
        self.it_max = 0
        while self.time[self.it_max] < tmax and self.it_max < self.ntime-1:
            self.it_max += 1
        # get number of time points in steady-state interval
        self.it_interval = self.ntime - self.it_min

        self.time_steady = self.time[self.it_min:self.it_max]
        self.ntime_steady = self.time_steady.size

        if run.taumax is None:
            self.taumax = 0.2 * (self.time_steady[-1]-self.time_steady[0])
        else:
            self.taumax = run.taumax

        self.ntauwin = int((self.time[-1]-self.time[0])//(0.5*self.taumax)) - 1
        self.t_tauwinavg = self.tauwin_avg(self.time)

        # get set of frequencies sampled by data, assuming equal time steps
        self.frequency = 2*np.pi*np.fft.fftshift(
                np.fft.fftfreq(self.ntime_steady,self.time[self.ntime-1]-self.time[self.ntime-2]))

        self.wgts_time = np.arange(self.ntime,dtype=float)
        self.wgts_time[0] = 0.5*(self.time[1]-self.time[0])
        for i in range(1,self.ntime-1):
            self.wgts_time[i] = 0.5*(self.time[i+1]-self.time[i-1])
        self.wgts_time[self.ntime-1] = 0.5*(self.time[self.ntime-1]-self.time[self.ntime-2])

        print('complete')


    def timeavg(self,ft,it_min=None,it_max=None,use_ft_full=False):

        if it_min is None:
            it_min = self.it_min
            it_max = self.it_max
            mytime = self.time_steady
        else:
            mytime = self.time[it_min:it_max]

        if use_ft_full:
            myft = ft
        else:
            myft = ft[it_min:it_max]

        favg = simps(myft,x=mytime) \
            / (mytime[-1]-mytime[0])

        return favg


    def tauwin_avg(self, f_vs_t):

        intlolim = 0
        intuplim = 0
        f_tauwinavg = np.zeros(self.ntauwin, dtype=type(f_vs_t[0]))

        for itauwin in range(self.ntauwin):

            while (intuplim < self.ntime-1) and (self.time[intuplim] < self.time[intlolim]+self.taumax):
                intuplim += 1

            fwindow = f_vs_t[intlolim:intuplim]
            twindow = self.time[intlolim:intuplim]
            f_tauwinavg[itauwin] = simps(fwindow,x=twindow) \
                / (twindow[-1]-twindow[0])

            prev_tlolim = self.time[intlolim]
            while (intuplim < self.ntime-1) and (self.time[intlolim] < prev_tlolim + 0.5*self.taumax):
                intlolim += 1

        return f_tauwinavg


    def tauwin_sigma(self, f_vs_t, f_winavg):

        intlolim = 0
        intuplim = 0
        f_tauwinsig = np.zeros(self.ntauwin, dtype=type(f_vs_t[0]))

        for itauwin in range(self.ntauwin):

            while (intuplim < self.ntime-1) and (self.time[intuplim] < self.time[intlolim]+self.taumax):
                intuplim += 1

            fwindow = (f_vs_t[intlolim:intuplim] - f_winavg[itauwin])**2
            twindow = self.time[intlolim:intuplim]
            f_tauwinsig[itauwin] = ( simps(fwindow,x=twindow) \
                / (twindow[-1]-twindow[0]) ) ** 0.5

            prev_tlolim = self.time[intlolim]
            while (intuplim < self.ntime-1) and (self.time[intlolim] < prev_tlolim + 0.5*self.taumax):
                intlolim += 1

        return f_tauwinsig
