import argparse
import os
import sys
import shutil
import time

import chainer
from chainer import optimizers

def create_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group('Dataset settings')
    group.add_argument(
        '--kitti_path', type=str,
        help='The path to the root of KITTI dataset')
    
    group = parser.add_argument_group('Model parameters')
    group.add_argument(
        '--model_name', type=str, default='VoxelNet',
        help='The model type name')
    group.add_argument(
        '--target', type=str, default='Car',
        choices=['Car', 'Pedestrian', 'Cyclist'],
        help='The target category for the network')
    group.add_argument(
        '-T', '--max_points_per_voxel', type=int, default=35,
        help='Maximum points per voxel')
    group.add_argument(
        '-K', '--max_voxels', type=int, default=20000,
        help='Maximum nonempty voxels')
    group.add_argument(
        '-A', '--anchors_per_position', type=int, default=2,
        help='Anchors per position')

    group = parser.add_argument_group('Train settings')
    group.add_argument(
        '--epoch', type=int, default=100,
        help='When the trianing will finish')
    group.add_argument(
        '--gpus', type=str, default='0',
        help='GPU Ids to be used')
    group.add_argument(
        '--batchsize', type=int, default=4,
        help='minibatch size')
    group.add_argument(
        '--snapshot_iter', type=int, default=1000,
        help='The current learnt parameters in the model is saved every'
             'this iteration')
    group.add_argument(
        '--valid_freq', type=int, default=1,
        help='Perform test every this iteration (0 means no test)')
    group.add_argument(
        '--valid_batchsize', type=int, default=1,
        help='The mini-batch size during validation loop')
    group.add_argument(
        '--show_log_iter', type=int, default=10,
        help='Show loss value per this iterations')
    group.add_argument(
        '--loss_alpha', type=float, default=1.5,
        help='Loss coefficient for positive anchors')
    group.add_argument(
        '--loss_beta', type=float, default=1,
        help='Loss coefficient for negative anchors')

    group = parser.add_argument_group('Optimization settings')
    group.add_argument(
        '--opt', type=str, default='Adam',
        choices=['MomentumSGD', 'Adam', 'AdaGrad', 'RMSprop'],
        help='Optimization method')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--adam_alpha', type=float, default=0.001)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_eps', type=float, default=1e-8)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    return args

def create_result_dir(model_name):
    result_dir = 'results/{}_{}'.format(
        model_name, time.strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(result_dir):
        result_dir += '_{}'.format(time.clock())
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir

def get_optimizer(model, opt, lr, adam_alpha, adam_beta1,
                  adam_beta2, adam_eps, weight_decay, **discard):
    if opt == 'MomentumSGD':
        optimizer = optimizers.MomentumSGD(lr=lr, momentum=0.9)
    elif opt == 'Adam':
        optimizer = optimizers.Adam(
            alpha=adam_alpha, beta1=adam_beta1,
            beta2=adam_beta2, eps=adam_eps)
    elif opt == 'AdaGrad':
        optimizer = optimizers.AdaGrad(lr=lr)
    elif opt == 'RMSprop':
        optimizer = optimizers.RMSprop(lr=lr)
    else:
        raise Exception('No optimizer is selected')

    # The first model as the master model
    optimizer.setup(model)
    if opt == 'MomentumSGD':
        optimizer.add_hook(
            chainer.optimizer.WeightDecay(weight_decay))

    return optimizer

def get_params_from_target(target):
    # TODO: add voxel size to command-line params
    params = dict()
    if target == 'Car':
        params['BX'] = (0, 70.4)
        params['BY'] = (-40, 40)
        params['BZ'] = (-3, 1)
        params['VD'] = 0.4
        params['VH'] = 0.2
        params['VW'] = 0.2
        params['AL'] = 3.9
        params['AW'] = 1.6
        params['AH'] = 1.56
        params['AZ'] = -1.0
        params['pos_thres'] = 0.6
        params['neg_thres'] = 0.45

    params['B'] = (params['BX'], params['BY'], params['BZ'])
    params['V'] = (params['VW'], params['VH'], params['VD'])
    params['AS'] = (params['AW'], params['AL'], params['AH'])
    return params
