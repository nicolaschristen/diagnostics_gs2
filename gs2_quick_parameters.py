# Avoid plotting (useful on HPC)
no_plot = False

# Avoid reading NETCDF files and plot from mat-files
only_plot = True

# Name(s) of simulation(s), without extension of file(s).
#fnames = ['mix_dkx_1','mix_dkx_25','mix_dkx_50','mix_dkx_100']
#fnames = ['old_dkx_1','old_dkx_25','old_dkx_50','old_dkx_100']
#fnames = ['mix_dikx_100_2ky']
fnames = ['1661_0.8_g040_mix_id_1']
#fnames = ['ollie_badshear_old_id_1']

scan_name = 'dummy'

# Path to directory containing GS2 input and output files. Default is current directory.
work_dir = './'

# Path to directory where plots will be saved. Default is current directory.
out_dir = work_dir #+ 'postproc_test3/'

# Task(s) to complete. Default is 'fluxes'. Check gs2_analysis.py for all possibilities.
tasks = ['floquet']

# Specify fraction of time over which the solution is nonlinearly saturated (0.0 -> 1.0).
# Only used for averaging. Default is 0.5.
twin = 0.3
