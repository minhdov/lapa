import h5py
from pathlib import Path

hdf5_path = "/storage/minh/philo/datasets/LIBERO/datasets/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"

def print_hdf5_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"{name}/")

with h5py.File(hdf5_path, "r") as f:
    f.visititems(print_hdf5_structure)
    
    
