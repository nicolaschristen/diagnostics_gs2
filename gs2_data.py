from netCDF4 import Dataset
import numpy as np
import f90nml

out_varnames = [
        't', 'kx', 'ky',
        'theta', 'theta0',
        # modulus squared, avged over theta, kx and ky [t]
        'phi2',
        # modulus squared, avged over theta [t,ky,kx]
        'phi2_by_mode',
        # modulus squared, avged over theta and kx [t,ky]
        'phi2_by_ky',
        # complex potential fluctuation [t,ky,kx,theta,imag]
        'phi_t',
        # complex frequency [t,ky,kx,imag]
        'omega',
        # complex frequency avged over navg time-steps [t,ky,kx,imag]
        'omega_average',
        # electrostatic particle flux [t,spec]
        'es_part_flux',
        # electrostatic heat flux [t,spec]
        'es_heat_flux',
        # electrostatic momentum flux [t,spec]
        'es_mom_flux',
        # turbulent energy exchange [t,spec]
        'es_energy_exchange',
        # electrostatic particle flux by k [t,spec,ky,kx]
        'es_part_flux_by_mode',
        # electrostatic heat flux by k [t,spec,ky,kx]
        'es_heat_flux_by_mode',
        # electrostatic momentum flux by k [t,spec,ky,kx]
        'es_mom_flux_by_mode',
        # parallel velocity grid
        'vpa',
        # electrostatic particle flux function of vpa and theta [t,spec,vpa,theta]
        'es_part_sym',
        # electrostatic heat flux function of vpa and theta [t,spec,vpa,theta]
        'es_heat_sym',
        # electrostatic momentum flux function of vpa and theta [t,spec,vpa,theta]
        'es_mom_sym',
        # complex phi for a given theta, usually outboard mid-plane [t,ky,kx,imag]
        'phi_igomega_by_mode',
        # complex density fluctuation for a given theta, usually outboard mid-plane [t,spec,ky,kx,imag]
        'ntot_igomega_by_mode',
        # complex parallel flow fluctuation for a given theta, usually outboard mid-plane [t,spec,ky,kx,imag]
        'upar_igomega_by_mode',
        # complex parallel temperature fluctuation for a given theta, usually outboard mid-plane [t,spec,ky,kx,imag]
        'tpar_igomega_by_mode',
        # complex perpendicular temperature fluctuation for a given theta, usually outboard mid-plane [t,spec,ky,kx,imag]
        'tperp_igomega_by_mode',
        # gds2
        'gds2',
        # gds21
        'gds21',
        # gds22
        'gds22'
        ]

def get_output(ifile, run):

    fname = run.full_nc_fname(ifile)
    ncfile = Dataset(fname, 'r')

    myout = {}
    
    # Get variables from NETCDF file and add to myout
    myout['nspec'] = ncfile.dimensions['species'].size
    for varname in out_varnames:
        get_single_output(ncfile, varname, myout)

    # Name of phi_igomega_by_mode used to be phi0, so check for that varname too
    if (not myout['phi_igomega_by_mode_present']):
        get_single_output(ncfile, 'phi0', myout, 'phi_igomega_by_mode')

    return myout


def get_single_output(ncfile, varname, myout, newname=None):
    
    if (not newname):
        newname = varname
    
    try:
        myout[newname] = np.copy(ncfile.variables[varname][:])
        myout[newname+'_present'] = True
    except KeyError:
        print('INFO: ' + varname + ' not found in netcdf file')
        myout[newname] = None
        myout[newname+'_present'] = False
    
    return


def get_input(ifile, run):
    
    fname = run.full_in_fname(ifile)
    myin = f90nml.read(fname)

    return myin

