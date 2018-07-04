# Avoid plotting (useful on HPC)
no_plot = False

# Avoid reading NETCDF files and plot from mat-files
only_plot = False

# Name(s) of simulation(s), without extension of file(s).
fnames = ['1661_tprim_1.3_ky_0.8_g_exb_0.06_old','1661_tprim_1.3_ky_0.8_g_exb_0.06_interp','1661_tprim_1.3_ky_0.8_g_exb_0.06_mixed']
#fnames = ['exp_michael','interp','exp_felix','mixed','old']

scan_name = 'dummy'

# Path to directory containing GS2 input and output files. Default is current directory.
work_dir = '/home/christenl/data/gs2/flowtest/linear/lin_tprim_scan/floquet/'

# Path to directory where plots will be saved. Default is current directory.
out_dir = work_dir + 'postproc/'

# Task(s) to complete. Default is 'fluxes'. Check gs2_analysis.py for all possibilities.
tasks = ['floquet']

# Specify fraction of time over which the solution is nonlinearly saturated (0.0 -> 1.0).
# Only used for averaging. Default is 0.5.
twin = 0.5
