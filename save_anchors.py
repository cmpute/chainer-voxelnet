import os.path as osp
import numpy as np

from datasets.kitti.kitti_utils import VoxelPreprocessor, AnchorPreprocessor
from datasets.kitti.kitti_3d_detection_dataset import _load_kitti_points,\
    _load_kitti_labels, _load_kitti_calib
from utils.prepare_args import create_args, get_params_from_target

def main():
    args = create_args('train')
    targs = get_params_from_target(args.target)
    targs['A'] = args.anchors_per_position
    targs['T'] = args.max_points_per_voxel
    targs['K'] = args.max_voxels

    prep = VoxelPreprocessor(**targs)
    anchor = AnchorPreprocessor(**targs)

    point = _load_kitti_points(osp.join(args.kitti_path, 'training', 'velodyne', '000010.bin'))
    label = _load_kitti_labels(osp.join(args.kitti_path, 'training', 'label_2', '000010.txt'))
    calib = _load_kitti_calib(osp.join(args.kitti_path, 'training', 'calib', '000010.txt'))
    
    _, _, gt_params = prep(point, label, calib)
    pos_equal_one, neg_equal_one, _ = anchor(gt_params)
    np.savetxt('anchors_positive.txt', anchor.anchors[pos_equal_one == 1], fmt='%.4f')
    np.savetxt('anchors_negative.txt', anchor.anchors[neg_equal_one == 1], fmt='%.4f')

if __name__ == '__main__':
    main()
