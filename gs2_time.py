import numpy as np
from scipy.integrate import simps
from numpy import fft
from math import ceil


class timeobj:

    def __init__(self, myout, twin):

        print()
        print('calculating time grid...',end='')

        self.time = np.copy(myout['t'])
        self.ntime = self.time.size

        # get starting index for steady-state
        self.twin = twin
        self.it_min = ceil((1.0-twin)*self.ntime)
        self.it_max = self.ntime-1
        # get number of time points in steady-state interval
        self.it_interval = self.ntime - self.it_min

        self.time_steady = self.time[self.it_min:self.it_max]
        self.ntime_steady = self.time_steady.size

        # get set of frequencies sampled by data, assuming equal time steps
        self.frequency = 2*np.pi*np.fft.fftshift(
                np.fft.fftfreq(self.ntime_steady,self.time[self.ntime-1]-self.time[self.ntime-2]))

        self.wgts_time = np.arange(self.ntime,dtype=float)
        self.wgts_time[0] = 0.5*(self.time[1]-self.time[0])
        for i in range(1,self.ntime-1):
            self.wgts_time[i] = 0.5*(self.time[i+1]-self.time[i-1])
        self.wgts_time[self.ntime-1] = 0.5*(self.time[self.ntime-1]-self.time[self.ntime-2])

        print('complete')


    def timeavg(self,ft):

        favg = simps(ft[self.it_min:self.it_max],x=self.time_steady) \
            / (self.time[self.it_max-1]-self.time[self.it_min])
        return favg
