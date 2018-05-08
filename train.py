from models import VoxelNet
from utils.prepare_train import create_args, \
    create_result_dir, get_optimizer, get_params_from_target
from utils.model_train import CRMapDetector

from chainer.datasets import TransformDataset
from chainer.iterators import MultiprocessIterator
from chainer.training import Trainer, ParallelUpdater
from chainer.training import extensions
from datasets import Kitti3DDetectionDataset as KITTI
from datasets.kitti.kitti_utils import VoxelRPNPreprocessor

def main():
    args = create_args()
    targs = get_params_from_target(args.target)
    targs['T'] = args.max_points_per_voxel
    targs['A'] = args.anchors_per_position
    result_dir = create_result_dir(args.model_name)

    # Prepare devices
    devices = {}
    for gid in [int(i) for i in args.gpus.split(',')]:
        if 'main' not in devices:
            devices['main'] = gid
        else:
            devices['gpu{}'.format(gid)] = gid
           
    # Instantiate a model
    model = CRMapDetector(VoxelNet, args.loss_alpha, args.loss_beta, **targs)

    # Instantiate a optimizer
    optimizer = get_optimizer(model, **vars(args))
    
    # Setting up datasets
    prep = VoxelRPNPreprocessor(**targs)
    train = TransformDataset(KITTI(args.kitti_path, 'train'), prep)
    valid = TransformDataset(KITTI(args.kitti_path, 'val'), prep)
    print('train: {}, valid: {}'.format(len(train), len(valid)))

    # Iterator
    train_iter = MultiprocessIterator(train, args.batchsize)
    valid_iter = MultiprocessIterator(valid, args.valid_batchsize, repeat=False, shuffle=False)

    # Updater
    updater = ParallelUpdater(train_iter, optimizer, devices=devices)
    trainer = Trainer(updater, (args.epoch, 'epoch'), out=result_dir)

    # Extentions
    trainer.extend(extensions.Evaluator(valid_iter, model, device=devices['main']),
        trigger=(args.valid_freq, 'epoch'))
    trainer.extend(extensions.snapshot(),
        trigger=(args.snapshot_iter, 'iteration'))
    trainer.extend(extensions.LogReport(),
        trigger=(args.show_log_iter, 'iteration'))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

if __name__ == '__main__':
    main()
