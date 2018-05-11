from models import VoxelNet
from datasets.kitti.kitti_utils import VoxelPreprocessor,\
    AnchorPreprocessor, compute_voxelgrid_size
from datasets.kitti.kitti_3d_detection_dataset import _load_kitti_points
from utils.prepare_args import create_args, get_params_from_target

from chainer.serializers import load_hdf5
import chainer.cuda as cuda
import numpy as np

def main():
    args = create_args('test')
    targs = get_params_from_target(args.target)
    targs['A'] = args.anchors_per_position
    targs['T'] = args.max_points_per_voxel
    targs['K'] = args.max_voxels

    # Prepare devices
    devices = {}
    for gid in [int(i) for i in args.gpus.split(',')]:
        if 'main' not in devices:
            devices['main'] = gid
        else:
            devices['gpu{}'.format(gid)] = gid

    # Build network
    targs['target'] = None
    prep = VoxelPreprocessor(**targs)
    anchor = AnchorPreprocessor(**targs)
    model = VoxelNet(**targs)
    load_hdf5(args.net_weight_path, model)

    # Network forward
    points = _load_kitti_points(args.input_path)
    voxel_features, voxel_coord = prep(points, None, None)
    if cuda.available and args.gpus:
        gpu = devices['main']
        model.to_gpu(gpu)
        voxel_features = cuda.to_gpu(voxel_features, gpu)
        voxel_coord = cuda.to_gpu(voxel_coord, gpu)

    xp = cuda.get_array_module(voxel_features)
    voxel_features = xp.expand_dims(voxel_features, 0) # TODO: support more batches
    voxel_coord = xp.expand_dims(voxel_coord, 0)
    maps = model(voxel_features, voxel_coord)

    for pos_map, reg_map in zip(*maps): # process each batch
        # pos_map: (2, H/2, W/2)
        # reg_map: (14, H/2, W/2)
        mask = pos_map.array > args.rpn_score_thres
        reg_map = reg_map.array.reshape((-1, 7) + reg_map.shape[-2:])
        locs = reg_map.transpose(0, 2, 3, 1)[mask]

        if cuda.available and args.gpus:
            locs = cuda.to_cpu(locs)
            mask = cuda.to_cpu(mask)

        # restore label values
        c = np.full(len(locs), 1, dtype=int) # TODO: different class value
        x = locs[:, 0] * anchor.anchord + anchor.anchors.transpose(3, 2, 0, 1)[0][mask]
        y = locs[:, 1] * anchor.anchord + anchor.anchors.transpose(3, 2, 0, 1)[1][mask]
        z = locs[:, 2] * targs['AH'] + anchor.anchorz
        h = np.exp(locs[:, 3]) * targs['AH']
        w = np.exp(locs[:, 4]) * targs['AW']
        l = np.exp(locs[:, 5]) * targs['AL']
        r = locs[:, 6] + anchor.anchors.transpose(3, 2, 0, 1)[6][mask]

        np.savetxt('result.txt', np.vstack((c, x, y, z, h, w, l, r)).T, fmt='%.4f')
        print(len(locs), "boxes in total.")
        print("Saved labels to result.txt!")
        
if __name__ == '__main__':
    main()
