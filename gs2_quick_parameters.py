# Avoid plotting (useful on HPC)
no_plot = False
#no_plot = True


# Avoid reading NETCDF files and plot from mat-files
only_plot = True
#only_plot = False

# Name(s) of simulation(s), without extension of file(s).

#fnames = ['miller_gs2_idam_jet_535211215_t48.000_rho0.300']
#fnames = ['shaping']
#fnames = ['delta_0.00/kappa_1.00/shaping','delta_0.00/kappa_1.50/shaping','delta_0.00/kappa_2.00/shaping','delta_0.17/kappa_1.00/shaping','delta_0.17/kappa_1.50/shaping','delta_0.17/kappa_2.00/shaping','delta_0.50/kappa_1.00/shaping','delta_0.50/kappa_1.50/shaping','delta_0.50/kappa_2.00/run1/shaping']
fnames = ['delta_0.00/kappa_1.00/shaping','delta_0.00/kappa_1.50/shaping','delta_0.00/kappa_2.00/shaping','delta_0.17/kappa_1.00/shaping','delta_0.17/kappa_1.50/shaping','delta_0.17/kappa_2.00/shaping','delta_0.50/kappa_1.00/shaping','delta_0.50/kappa_1.50/shaping','delta_0.50/kappa_2.00/run1/shaping_run0_restart']
#fnames=['run0/shaping','run1/shaping_run0_restart']
scan_name = 'dummy'

# Path to directory containing GS2 input and output files. Default is current directory.
work_dir = './'

# Path to directory where plots will be saved. Default is current directory.
#out_dir = work_dir
out_dir = ''

# Task(s) to complete. Default is 'fluxes'. Check gs2_analysis.py for all possibilities.
tasks = ['tri_kap_nl']
#tasks = ['along_tube']

# Specify fraction of time over which the solution is nonlinearly saturated (0.0 -> 1.0).
# Only used for averaging. Default is 0.5.
twin = 0.5
