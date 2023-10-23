import os
import h5py
import numpy as np
from mmpretrain.registry import DATASETS
from mmengine.dataset import BaseDataset


@DATASETS.register_module()
class MultiHDF5Dataset(BaseDataset):
    def __init__(self, data_root, samples_per_file=10000, test_mode=False, pipeline=[]):
        self.data_root = data_root
        self.hdf5_files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.h5')]
        self.samples_per_file = samples_per_file
        super().__init__(test_mode=test_mode, lazy_init=True, pipeline=pipeline)

    def full_init(self):
        # Your custom initialization logic here
        self._fully_initialized = True

    def __len__(self):
        return len(self.hdf5_files) * self.samples_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_file
        sample_idx = idx % self.samples_per_file
        with h5py.File(self.hdf5_files[file_idx], 'r') as f:
            image = np.array(f['images'][sample_idx])+1

        image = image.transpose((1, 2, 0))  # Change order to (H, W, channel)

        # Prepare the data in the format expected by the pipeline
        data = dict(img=image, supp_data=[])

        # Process the data through the pipeline
        data = self.pipeline(data)

        return data