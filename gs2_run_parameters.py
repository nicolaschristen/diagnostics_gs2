import sys
import argparse

# Get settings for quick-mode
import gs2_quick_parameters as qpar

class runobj:

    def __init__(self, tasks_choices):

        # Set defaults
        self.tasks = ['fluxes']
        self.work_dir = './'
        self.fnames = []
        self.scan_name = ''
        self.out_dir = './'
        self.twin = 0.5
        self.no_plot = False
        self.only_plot = False

        # Modify attributes according to command-line arguments.
        self.set_parameters(tasks_choices)


    def set_parameters(self, tasks_choices):

        args = self.get_commandline(tasks_choices)

        if args.quick:
            args.fnames = qpar.fnames
            args.scan_name = qpar.scan_name
            args.work_dir = qpar.work_dir
            args.out_dir = qpar.out_dir
            args.tasks = qpar.tasks
            args.twin = qpar.twin
            args.no_plot = qpar.no_plot
            args.only_plot = qpar.only_plot
            
        if args.tasks: self.tasks = args.tasks
        
        if args.work_dir: self.work_dir = args.work_dir
        if (not self.work_dir[-1]=='/'): self.work_dir = self.work_dir + '/'

        if args.fnames:
            self.fnames = args.fnames
        else:
            sys.exit("Please provide a filename or param in quick mode --quick (for help, use option -h).")

        if len(self.fnames) > 1:
            if args.scan_name:
                self.scan_name = args.scan_name
            else:
                sys.exit("Please provide a scan name or use quick mode --quick (for help, use option -h).")
        
        if args.out_dir: self.out_dir = args.out_dir
        if (not self.out_dir[-1]=='/'): self.out_dir = self.out_dir + '/'

        if args.twin: self.twin = float(args.twin)

        if args.no_plot: self.no_plot = True
        
        if args.only_plot: self.only_plot = True


    def get_commandline(self, tasks_choices):

        parser = argparse.ArgumentParser(description = 'Creating various plots from GS2 NETCDF output files.')
        
        parser.add_argument('-t', '--tasks', nargs = '*', choices = tasks_choices,
                help = 'Task(s) to complete. Default is \'fluxes\'. Check gs2_analysis.py for all possibilities.')
        
        parser.add_argument('-w', '--work_dir', nargs = '?', 
                help = 'Path to directory containing GS2 input and output files. Default is current directory.')
        
        parser.add_argument('-f', '--fnames', nargs = '*',
                help = 'Name(s) of simulation(s), without extension of file(s). Must be specified if not in quick mode (-q, --quick).')
        
        parser.add_argument('-s', '--scan_name', nargs = '?',
                help = 'Name of scan, without any extension. Must be specified if not in quick mode (-q, --quick).')
        
        parser.add_argument('-o', '--out_dir', nargs = '?',
                help = 'Path to directory where plots will be saved. Default is current directory.')

        parser.add_argument('--twin', nargs = '?',
                help = 'Specify fraction of time over which the solution is nonlinearly saturated (0.0 -> 1.0). Default is 0.5.')

        parser.add_argument('-q', '--quick', action = 'store_true', default = False,
                help = 'Quick mode: all parameters are set to values from my_quick_params.py file.')

        parser.add_argument('-n', '--no_plot', action = 'store_true', default = False,
                help = 'Saving output from tasks to my_file.my_task.mat, without plotting (useful on HPC).')

        parser.add_argument('-p', '--only_plot', action = 'store_true', default = False,
                help = 'Output from tasks has already been saved to my_file.my_task.mat files, so no reading from NETCFD files required and only need to plot.')
        
        args = parser.parse_args()
        
        return args
