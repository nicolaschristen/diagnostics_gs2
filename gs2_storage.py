import shelve

def save_to_file(fname, my_vars=None):

    # If only one arg is provided, then variable name list is the one from current global namespace
    if my_vars is None:
        varnames = dir()
        namespace = globals()
    # If second arg is a list of strings, use it directly as variable name list
    elif type(my_vars) is list:
        varnames = my_vars
        namespace = globals()
    # If second arg is a dict, use its key list as variable name list
    elif type(my_vars) is dict:
        varnames = list(my_vars.keys())
        namespace = my_vars

    # Copy variables from namespace to my_shelf
    my_shelf = shelve.open(fname, 'n') # 'n' for new
    for key in varnames:
        try:
            my_shelf[key] = namespace[key]
        except:
            print('Cannot shelve: {0}'.format(key))
    my_shelf.close()

def read_from_file(fname, my_dict=None):

    # If only one arg is provided, we restore variables from my_shelf to the current global namespace
    if my_dict is None:
        namespace = globals()
    # Otherwise we restore variables from my_shelf to the provided dictionary
    else:
        namespace = my_dict

    # Copy variables from my_shelf to namespace
    my_shelf = shelve.open(fname)
    for key in my_shelf:
        namespace[key] = my_shelf[key]
    my_shelf.close()
