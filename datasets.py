import os
import csv
import glob
import h5py
import numpy as np
import torch
import torch.utils.data as data

train_files = ['005', '014', '015', '016']

# train_files = ['005', '014', '015', '016', '025', '036', '038', '041', '045',
#                '047', '052', '054', '057', '061', '062', '066', '071', '073', '078', '080',
#                '084', '087', '089', '096', '098', '109', '201', '202', '209', '217', '223',
#                '225', '227', '231', '234', '237', '240', '243', '249', '251', '255', '260',
#                '263', '265', '270', '276', '279', '286', '294', '308', '522', '609', '613',
#                '614', '623', '700']
test_files = ['011', '021', '065', '032', '093', '246', '086', '069', '206',
              '252', '273', '527', '621', '076', '082', '049', '207', '213', '272', '074']


class SceneNNDataset(data.Dataset):
    def __init__(self, root, training=True):
        self.root = root
        self.training = training

        if self.training:
            self.filenames = train_files
        else:
            self.filenames = test_files

        for fn in self.filenames:
            fp = os.path.join(self.root, 'scenenn_seg_' + fn + '.hdf5')
            print(fp)
            with h5py.File(fp, 'r') as f:
                data = np.array(f['data'])
                label = np.array(f['label'])
                if not hasattr(self, 'data'):
                    self.data = data
                    self.label = label
                    self.num_points = data.shape[1]
                    self.num_channels = data.shape[2]
                elif data.shape[0] > 0:
                    self.data = np.concatenate((self.data, data))
                    self.label = np.concatenate((self.label, label))

    def __getitem__(self, index):
        points = self.data[index, :, 0:3]
        labels = self.label[index, :]
        return points, labels

    def __len__(self):
        return self.data.shape[0]
