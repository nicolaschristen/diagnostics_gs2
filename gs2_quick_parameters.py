## Path to working directory (all other paths have to be specified relatively to it).
#work_dir = './'
work_dir = '/home/christenl/data/gs2/flowtest/linear/dkx_scan_new/tmp'

## 'path/from/work_dir/to/sim_file' without extension of file(s).
# Can be a single str or an array of str.
#single_files = ['myfolder1/mysim1','myfolder2/mysim2']
single_files = ['folder1/test1', 'folder2/test2']

## Task(s) to complete for each single file.
# Can be a single str or an array of str.
# Check gs2_run_parameters.py for all possibilities.
single_tasks = ['fluxes']

## Name of directory where plots will be saved.
# Must be located in same directory as in/out files.
# Will be created if no such dir.
single_outdirname = ''

## Task(s) to complete using the whole collection of files (e.g. for scans).
# Check gs2_run_parameters.py for all possibilities.
multi_tasks = ['stitch']

## 'path/from/work_dir/to/multi_file' to use when saving multi-file analysis,
# without any extension.
multi_fname = 'multi_folder/multi_postproc'

# Saving postproc results to .dat files, without producing plots
# (useful on HPCs where LaTeX is not installed).
no_plot = False

# Output from tasks has already been saved to .dat files,
# so no reading from NETCDF files required and only need to plot.
only_plot = False

## Fraction of time over which the solution is nonlinearly saturated
# (0.0 -> 1.0).
twin = 0.5
