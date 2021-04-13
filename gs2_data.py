from netCDF4 import Dataset
import numpy as np
import f90nml

out_varnames = [
        't', 'kx', 'ky',
        'theta', 'theta0',
        'phi',
        'apar',
        'bpar',
        # modulus squared, avged over theta, kx and ky [t]
        'phi2',
        # modulus squared, avged over theta [t,ky,kx]
        'phi2_by_mode',
        # modulus squared, avged over theta and kx [t,ky]
        'phi2_by_ky',
        # modulus squared, avged over theta and ky [t,kx]
        'phi2_by_kx',
        # modulus squared, avged over theta, kx and ky [t]
        'apar2',
        # modulus squared, avged over theta [t,ky,kx]
        'apar2_by_mode',
        # modulus squared, avged over theta and kx [t,ky]
        'apar2_by_ky',
        # modulus squared, avged over theta and ky [t,kx]
        'apar2_by_kx',
        # modulus squared, avged over theta, kx and ky [t]
        'bpar2',
        # modulus squared, avged over theta [t,ky,kx]
        'bpar2_by_mode',
        # modulus squared, avged over theta and kx [t,ky]
        'bpar2_by_ky',
        # modulus squared, avged over theta and ky [t,kx]
        'bpar2_by_kx',
        # complex potential fluctuation [t,ky,kx,theta,imag]
        'phi_t',
        # complex parallel magnetic potential fluctuation [t,ky,kx,theta,ri]
        'apar_t',
        # complex parallel magnetic field fluctuation [t,ky,kx,theta,ri]
        'bpar_t',
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
        
        # electromagnetic (apar) particle flux [t,spec]
        'apar_part_flux',
        # electromagnetic (apar) heat flux [t,spec]
        'apar_heat_flux',
        # electromagnetic (apar) momentum flux [t,spec]
        'apar_mom_flux',
        # electromagnetic (apar) particle flux by k [t,spec,ky,kx]
        'apar_part_flux_by_mode',
        # electromagnetic (apar) heat flux by k [t,spec,ky,kx]
        'apar_heat_flux_by_mode',
        # electromagnetic (apar) momentum flux by k [t,spec,ky,kx]
        'apar_mom_flux_by_mode',
      
        # electromagnetic (bpar) particle flux [t,spec]
        'bpar_part_flux',
        # electromagnetic (bpar) heat flux [t,spec]
        'bpar_heat_flux',
        # electromagnetic (bpar) momentum flux [t,spec]
        'bpar_mom_flux',
        # electromagnetic (bpar) particle flux by k [t,spec,ky,kx]
        'bpar_part_flux_by_mode',
        # electromagnetic (bpar) heat flux by k [t,spec,ky,kx]
        'bpar_heat_flux_by_mode',
        # electromagnetic (bpar) momentum flux by k [t,spec,ky,kx]
        'bpar_mom_flux_by_mode',
        
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
        'gds2',
        'gds21',
        'gds22',
        'gbdrift',
        'cvdrift',
        'gbdrift0',
        'cvdrift0'
#        'bmag',
        ]

def get_output(ifile, run):

    fname = run.work_dir + run.fnames[ifile] + '.out.nc'
    ncfile = Dataset(fname, 'r')

    myout = {}
    
    # Get variables from NETCDF file and add to myout
    myout['nspec'] = ncfile.dimensions['species'].size
    for varname in out_varnames:
        get_single_output(ncfile, varname, myout)

    # Name of phi_igomega_by_mode used to be phi0, so check for that varname too
    if (not myout['phi_igomega_by_mode_present']):
        get_single_output(ncfile, 'phi0', myout, 'phi_igomega_by_mode')
    
    marconi_crashed = False
    indices_to_delete = np.arange(myout['t'].size-100, myout['t'].size)
    if(marconi_crashed):
        myout['t'] = np.delete(myout['t'],indices_to_delete)
        myout['phi2'] = np.delete(myout['phi2'],indices_to_delete)
        myout['phi2_by_mode'] = np.delete(myout['phi2_by_mode'],indices_to_delete,axis=0)
        myout['phi_igomega_by_mode'] = np.delete(myout['phi_igomega_by_mode'],indices_to_delete,axis=0)
        myout['ntot_igomega_by_mode'] = np.delete(myout['ntot_igomega_by_mode'],indices_to_delete,axis=0)
        myout['upar_igomega_by_mode'] = np.delete(myout['upar_igomega_by_mode'],indices_to_delete,axis=0)
        myout['tpar_igomega_by_mode'] = np.delete(myout['tpar_igomega_by_mode'],indices_to_delete,axis=0)
        myout['phi2_by_ky'] = np.delete(myout['phi2_by_ky'],indices_to_delete,axis=0)
        myout['omega'] = np.delete(myout['omega'],indices_to_delete,axis=0)
        myout['omega_average'] = np.delete(myout['omega_average'],indices_to_delete,axis=0)
        myout['es_part_flux'] = np.delete(myout['es_part_flux'],indices_to_delete,axis=0)
        myout['es_heat_flux'] = np.delete(myout['es_heat_flux'],indices_to_delete,axis=0)
        myout['es_mom_flux'] = np.delete(myout['es_mom_flux'],indices_to_delete,axis=0)
        myout['es_energy_exchange'] = np.delete(myout['es_energy_exchange'],indices_to_delete,axis=0)
        myout['es_part_flux_by_mode'] = np.delete(myout['es_part_flux_by_mode'],indices_to_delete,axis=0)
        myout['es_heat_flux_by_mode'] = np.delete(myout['es_heat_flux_by_mode'],indices_to_delete,axis=0)
        myout['es_mom_flux_by_mode'] = np.delete(myout['es_mom_flux_by_mode'],indices_to_delete,axis=0)
        myout['es_part_sym'] = np.delete(myout['es_part_sym'],indices_to_delete,axis=0)
        myout['es_heat_sym'] = np.delete(myout['es_heat_sym'],indices_to_delete,axis=0)
        myout['es_mom_sym'] = np.delete(myout['es_mom_sym'],indices_to_delete,axis=0)

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
    
    fname = run.work_dir + run.fnames[ifile] + '.in'
    myin = f90nml.read(fname)

    return myin

