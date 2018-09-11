# Avoid plotting (useful on HPC)
no_plot = False

# Avoid reading NETCDF files and plot from mat-files
only_plot = True

# Name(s) of simulation(s), without extension of file(s).
#fnames = ['1661_0.8_g120_mix_id_1']#'1661_0.8_m0_mix_id_1','1661_0.8_g065_mix_id_1','1661_0.8_g065_old_id_1','jtwist_8_old_id_3'
fnames = ['dkx_100']
#fnames = ['ollie_badshear_mix_id_1','ollie_badshear_mix_id_2','ollie_badshear_mix_id_3']
#fnames = ['1661_0.8_g020_old_id_1']
#fnames = ['cbc_y0_1.00','cbc_y0_1.50','cbc_y0_2.00','cbc_y0_2.50','cbc_y0_3.00','cbc_y0_3.50','cbc_y0_4.00']

scan_name = 'dummy'

# Path to directory containing GS2 input and output files. Default is current directory.
work_dir = './'

# Path to directory where plots will be saved. Default is current directory.
out_dir = work_dir + 'postproc/'

# Task(s) to complete. Default is 'fluxes'. Check gs2_analysis.py for all possibilities.
tasks = ['potential']

# Specify fraction of time over which the solution is nonlinearly saturated (0.0 -> 1.0).
# Only used for averaging. Default is 0.5.
twin = 0.5
