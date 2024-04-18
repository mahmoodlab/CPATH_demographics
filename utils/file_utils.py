import pickle
import h5py

def save_pkl(filename, save_object):
    """
    Save an object to a pickle file.

    Args:
        filename (str): Storage path.
        save_object: Object to be saved.
    """
    writer = open(filename,'wb')
    pickle.dump(save_object, writer)
    writer.close()

def load_pkl(filename):
    """
    Load an object from a pickle file.

    Args:
        filename (str): Load path.

    Returns:
        object: Loaded object.
    """
    loader = open(filename,'rb')
    file = pickle.load(loader)
    loader.close()
    return file


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    """
    Save data to an HDF5 file.

    Args:
        output_path (str): Path to the HDF5 file.
        asset_dict (dict): Dictionary containing data to be saved.
        attr_dict (dict): Dictionary containing attributes for each dataset (optional).
        mode (str): Mode in which to open the HDF5 file (default is 'a').

    Returns:
        str: Path to the saved HDF5 file.
    """
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path