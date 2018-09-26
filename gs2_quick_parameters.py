# Avoid plotting (useful on HPC)
no_plot = False

# Avoid reading NETCDF files and plot from mat-files
only_plot = True

# Name(s) of simulation(s), without extension of file(s).
fnames = ['myfolder1/myfile1','myfolder2/myfile2']

scan_name = 'myscan_name'

# Path to directory containing GS2 input and output files. Default is current directory.
work_dir = './'

# Path to directory where plots will be saved. Default is current directory.
out_dir = ''

# Task(s) to complete. Default is 'fluxes'. Check gs2_analysis.py for all possibilities.
tasks = ['mytask']

# Specify fraction of time over which the solution is nonlinearly saturated (0.0 -> 1.0).
# Only used for averaging. Default is 0.5.
twin = 0.5
