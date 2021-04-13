import numpy as np
# Avoid plotting (useful on HPC)
no_plot = False
#no_plot = True

# Avoid reading NETCDF files and plot from mat-files
only_plot = False
#only_plot = True

# Plot with videos? If no_plot == True this has no effect.
make_movies = False
#make_movies = True

scan_name = 'dummy'

# Path to directory containing GS2 input and output files. Default is current directory.
work_dir = './'

# Path to directory where plots will be saved. Default is current directory.
out_dir = 'postproc'

# Task(s) to complete. Default is 'fluxes'. Check gs2_analysis.py for all possibilities.
tasks = ['floquet']
#tasks = ['fluxes']
tasks = ['range_pot']
#tasks = ['flxcompare']
#tasks = ['linflxcompare']
#tasks = ['ky_fsscan']
#tasks = ['nx_fsscan']
#tasks = ['dtheta0_fsscan']
#tasks = ['ky_gexb_fsscan']
#tasks = ['floquet', 'ky_fsscan']
#tasks = ['fluxes_stitch']
#tasks = ['boxballoon']
#tasks = ['gamma_omega_compare']

# Specify fraction of time over which the solution is nonlinearly saturated (0.0 -> 1.0).
# Only used for averaging. Default is 0.5.
twin = 0.5

pos = 'shaping'
pos = 'outer'
pos = 'itb'
#pos = 'foot'

# Name(s) of simulation(s), without extension of file(s).
#xs = [0.05,0.15,0.25,0.50,0.75,1.00,1.50,2.00,2.50,3.00]
#xs = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0]
#xs = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5]
#xname='ky'
#xfolder = '{}_{:.1f}'

#xs = np.linspace( 0, 0.5, 11 )
#xname = 'tri'
#xfolder = 'tripri_scan/{}_{:1.2f}'


#xfolder = 'hs10/{}_{:1.2f}/akappa_1.0'

#fnames = fnames + [xfolder.format(xname, x) + '/{}'.format(pos) for x in xs]

#hs
akappa  = 1.5203
akappri = 0.13506
tri     = 0.17224
tripri  = 0.36345
bprim = -0.15989*1.5
beta = 0.0370

#it
#akappa  = 1.5219
#akappri = 0.10301
#tri     = 0.14036
#tripri  = 0.29194

#x1s = np.linspace( 0, 0.5, 11 )
#x1s = np.linspace( 0, -0.22, 11 )
x1s = np.linspace( 0.005, 0.1, 20 )#[::19]
#x1s = np.linspace( 0, 0.5, 11 )[::10]
y1s = np.linspace( 1, 2, 11 )#[::10]
#x2s = x1s*tripri/tri
x2s = x1s*bprim/beta
y2s = y1s*akappri/akappa
fnames = []
for ix in range(len(x1s)):
    for iy in range(len(y1s)):
#       fnames.append("beta_prime_input_{:1.2f}/akappa_{:1.2f}_akappri_{:1.2f}/shaping".format(x1s[ix],y1s[iy],y2s[iy]))
        fnames.append("beta_{:1.3f}_beta_prime_input_{:1.2f}/akappa_{:1.2f}_akappri_{:1.2f}/shaping".format(x1s[ix],x2s[ix],y1s[iy],y2s[iy]))
#       fnames.append("tri_{:1.2f}_tripri_{:1.2f}/akappa_{:1.2f}_akappri_{:1.2f}/shaping".format(x1s[ix],x2s[ix],y1s[iy],y2s[iy]))


#xs = [[xs,xs*tripri/tri],[ys,ys*akappri/akappa]]
#xname = [['tri','tripri'], ['akappa','akappri']]

"""deut
 tprim =   3.502
 fprim =   2.51

elec
 tprim =   3.594
 fprim =   2.148
"""

#xs = 2.148*np.array([0.8,0.9,1.0,1.1,1.2])
#xname = 'fprim2'
#xfolder = '{}_{:1.1f}'

#xs = [-0.14, -0.23]
#xname = 'g_exb'
#xfolder = '{}_{:1.2f}'

#ys = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
#ys = [0.05, 0.1, 0.2, 0.3]
#ys = [1.0]
#yname = 'ky'
#yfolder = '{}_{:1.2f}'

#xs = [32,64,96,128,160,192]
xs = [16,48,80,112,144,176]
xs += [384,352,320,288,256,224,192,160,128,96,64,32]
#xs = [64,128,192,256,320,384]
#xs = [448,512,576,640,704,768]
xname = 'nx'
xfolder = '{}_{:d}/ky_0.7'

#fnames = [j for sub in [[(xfolder+'/'+yfolder+'/{}'.format(pos)).format(xname,x,yname,y) for x in xs ] for y in ys] for j in sub]
#fnames = [i for sub1 in [j for sub in [[[(xfolder+'/'+yfolder+'/'+zfolder+'/{}'.format(pos)).format(xname,x,yname,y,zname,z) for x in xs ] for y in ys] for z in zs] for j in sub] for i in sub1]
#fnames = [xfolder.format(xname, x) + '/{}'.format(pos) for x in xs]

fnames = ['{}'.format(pos)]
#fnames = ['{}_run0_restart'.format(pos)]
#fnames = ['fprim1_2.1_fprim2_2.5/outer','fprim1_2.1/outer', 'fprim1_3.5/outer', 'fprim2_1.5/outer', 'fprim2_2.5/outer', 'nom/outer', 'tprime_1.2/outer'] 
#fnames = ['nom/{}'.format(pos),'tprimi_1.8/outer', 'tprime_1.2/outer','fprim1_3.4/outer','fprim2_1.7/outer', 'f134_f217_ti18_te12/outer'] 
#fnames = ['es/nl/nom/kymax3/foot', 'es/nl/max_qie/foot', 'fs/nl/max_qie/foot']
#fnames = ['es/nl/itb', 'fs/nl/medres/itb', 'em/nl/itb', 'emfs/nl/shat0.1_gexb-0.14/itb']
#fnames = ['es/nl/nom/outer','es/nl/tprimi_1.8/outer', 'es/nl/tprime_1.2/outer','es/nl/fprim1_3.4/outer','es/nl/fprim2_1.7/outer', 
#    'es/nl/f134_f217_ti18_te12/outer', 'fs/nl/f134_f217_ti18_te12/outer', 'emfs/nl/fprim1_3.4_fprim2_1.7_tprimi_1.8_tprime_1.2/run1/outer_run0_restart']
#fnames = ['zeff1/{}'.format(pos),'zeff1_nocol/{}'.format(pos), 'nom/{}'.format(pos),'nom_nocol/{}'.format(pos)] 
#fnames = ['run0/outer', 'run1/outer_run0_restart']
#fnames = ['run1/outer_run0_restart']

print(fnames)
print(len(fnames))
