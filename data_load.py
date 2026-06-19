import h5py
from matplotlib import pyplot as plt


def load_data(type, slice, echo, coil):
    h5_path = r'C:...'
    with h5py.File(h5_path, 'r') as f:
        kspace = f['kspace'][()]   # (51, 4, 32, 290, 234), complex64 --> shape=(slice, echo, coil, ky, kx)
        sens_map   = f['sensitivity_map'][()]     # (51, 32, 290, 234), complex64 --> shape=(slice, coil, ky, kx)
        target = f['target'][()]    # (51, 4, 290, 234), complex64 --> shape=(slice, echo, kx, ky)

    if type == 'kspace':
        loaded_data = kspace[slice][echo][coil]
    elif type == 'sens_map':
        loaded_data = sens_map[slice][coil]
    elif type == 'target':
        loaded_data = target[slice][echo]

    return loaded_data
