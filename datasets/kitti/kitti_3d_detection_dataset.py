
import os
import os.path as osp

import numpy as np
from chainer.dataset import dataset_mixin

def _load_kitti_points(path, intensity=True):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points if intensity else points[:, :3]

def _load_kitti_labels(path):
    records = []
    with open(path, 'r') as fin:
        for line in fin:
            class_name, trunc, occlusion, alpha,\
                x1, y1, x2, y2,\
                h, w, l, x, y, z, yaw = line.strip().split(' ')
            records.append((class_name, float(trunc), int(occlusion), float(alpha),
                            float(x1), float(y1), float(x2), float(y2),
                            float(h), float(w), float(l),
                            float(x), float(y), float(z), float(yaw)))

    return records

def _load_kitti_calib(path):
    float_chars = set("0123456789.e+- ")

    data = {}
    with open(path, 'r') as f:
        for line in [l for l in f.read().split('\n') if len(l) > 0]:
            key, value = line.split(' ', 1)
            if key.endswith(':'):
                key = key[:-1]
            value = value.strip()
            if float_chars.issuperset(value):
                data[key] = np.array([float(v) for v in value.split(' ')])
            else:
                print('warning: unknown value!')

    return data

class Kitti3DDetectionDataset(dataset_mixin.DatasetMixin):
    '''
    Should be used along with TransformDataset
    transform input: point_cloud (ndarray), labels (list of tuple), calibration (dict of ndarray)
    '''
    def __init__(self, root_path, split='train'):
        if split is 'test':
            mid = 'testing'
        elif split in ('train', 'val'):
            mid = 'training'
        else:
            raise ValueError('Invalid dataset split')

        point_path = osp.join(root_path, mid, 'velodyne')
        label_path = osp.join(root_path, mid, 'label_2')
        calib_path = osp.join(root_path, mid, 'calib')

        paths = []
        for pfile in filter(lambda fname: fname.endswith('.bin'), os.listdir(point_path)):
            pname = pfile[:-4]
            pfile = osp.join(point_path, pfile)
            lfile = osp.join(label_path, pname + '.txt')
            cfile = osp.join(calib_path, pname + '.txt')
            paths.append((pfile, lfile, cfile))
        splitidx = int(len(paths) * 0.1) # split by file list

        if split is 'train':
            self.data = paths[splitidx:]
        elif split is 'val':
            self.data = paths[:splitidx]

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        pfile, lfile, cfile = self.data[i]
        return _load_kitti_points(pfile), \
            _load_kitti_labels(lfile), _load_kitti_calib(cfile)
