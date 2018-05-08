# TODO: Fix batchnorm arguments
from chainer import Chain, Sequential, Variable, Function
from chainer.backends import cuda
import chainer.links as L
import chainer.functions as F

from datasets.kitti.kitti_utils import \
    compute_voxelgrid_size, ensure_bounding, ensure_voxel

# conv2d + bn + relu
class Conv2d(Chain):
    def __init__(self, cin, cout, k, s, p, activation=True, batch_norm=True):
        super(Conv2d, self).__init__()
        self.activation = activation
        with self.init_scope():
            self.conv = L.Convolution2D(cin, cout, ksize=k, stride=s, pad=p)
            self.bn = L.BatchNormalization(cout) if batch_norm else None

    def __call__(self, x):
        x = self.conv(x)
        if self.bn: x = self.bn(x)
        if self.activation: x = F.relu(x)
        return x

# conv3d + bn + relu
class Conv3d(Chain):
    def __init__(self, cin, cout, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        with self.init_scope():
            self.conv = L.ConvolutionND(3, cin, cout, ksize=k, stride=s, pad=p)
            self.bn = L.BatchNormalization(cout) if batch_norm else None

    def __call__(self, x):
        x = self.conv(x)
        if self.bn: x = self.bn(x)
        return F.relu(x)

# Fully Connected Network
class FCN(Chain):
    def __init__(self, cin, cout):
        super(FCN, self).__init__()
        self.cout = cout
        with self.init_scope():
            self.linear = L.Linear(cin, cout)
            self.bn = L.BatchNormalization(cout)

    def __call__(self, x):
        kk, t, _ = x.shape # KK is the stacked k across batch
        x = self.linear(x.reshape(kk * t, -1))
        x = self.bn(x)
        x = F.relu(x)
        return x.reshape(kk, t, -1)

# Voxel Feature Encoding layer
class VFE(Chain):
    def __init__(self, cin, cout, T):
        super(VFE, self).__init__()
        assert cout % 2 == 0
        self.units, self.T = cout // 2, T
        with self.init_scope():
            self.fcn = FCN(cin, self.units)

    def __call__(self, x, mask):
        xp = cuda.get_array_module(x)
        # point-wise feauture
        pwf = self.fcn(x)
        # locally aggregated feature
        laf = xp.max(pwf, axis=1)[0].expand_dims(1).repeat(1, self.T, 1)
        # point-wise concat feature
        pwcf = xp.concatenate((pwf,laf), axis=2)
        # apply mask
        mask = mask.expand_dims(2).repeat(1, 1, self.units * 2)
        pwcf *= mask

        return pwcf

# Stacked Voxel Feature Encoding
class SVFE(Chain):
    def __init__(self, T):
        super(SVFE, self).__init__()
        with self.init_scope():
            self.vfe_1 = VFE(7, 32, T)
            self.vfe_2 = VFE(32, 128, T)
            self.fcn = FCN(128, 128)

    def __call__(self, x):
        xp = cuda.get_array_module(x)
        mask = xp.max(x, axis=2)[0] != 0
        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)
        x = xp.max(x, axis=1)[0] # element-wise max pooling
        return x

# Convolutional Middle Layer
class CML(Chain):
    def __init__(self):
        super(CML, self).__init__()
        with self.init_scope():
            self.conv3d_1 = Conv3d(128, 64, k=3, s=(2, 1, 1), p=(1, 1, 1))
            self.conv3d_2 = Conv3d(64, 64, k=3, s=(1, 1, 1), p=(0, 1, 1))
            self.conv3d_3 = Conv3d(64, 64, k=3, s=(2, 1, 1), p=(1, 1, 1))

    def __call__(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return x

# Region Proposal Network
# TODO: fix block params
class RPN(Chain):
    def __init__(self, anchors_per_position):
        super(RPN, self).__init__()

        self.block_1s = [Conv2d(128, 128, 3, 2, 1)]
        self.block_1s += [Conv2d(128, 128, 3, 1, 1) for _ in range(3)]
        self.block_2s = [Conv2d(128, 128, 3, 2, 1)]
        self.block_2s += [Conv2d(128, 128, 3, 1, 1) for _ in range(5)]
        self.block_3s = [Conv2d(128, 256, 3, 2, 1)]
        self.block_3s += [Conv2d(128, 256, 3, 1, 1) for _ in range(5)]

        with self.init_scope():
            self.block_1 = Sequential(*self.block_1s)
            self.block_2 = Sequential(*self.block_2s)
            self.block_3 = Sequential(*self.block_3s)

            self.deconv_1 = Sequential(L.Deconvolution2D(256, 256, 4, 4, 0), L.BatchNormalization(256))
            self.deconv_2 = Sequential(L.Deconvolution2D(128, 256, 2, 2, 0), L.BatchNormalization(128))
            self.deconv_3 = Sequential(L.Deconvolution2D(128, 256, 1, 1, 0), L.BatchNormalization(128))

            self.score_map = Conv2d(768, anchors_per_position, 1, 1, 0, activation=False, batch_norm=False)
            self.reg_map = Conv2d(768, 7 * anchors_per_position, 1, 1, 0, activation=False, batch_norm=False)

    def __call__(self, x):
        x = self.block_1(x)
        x_skip_1 = x
        x = self.block_2(x)
        x_skip_2 = x
        x = self.block_3(x)
        x_0 = self.deconv_1(x)
        x_1 = self.deconv_2(x_skip_2)
        x_2 = self.deconv_3(x_skip_1)

        xp = cuda.get_array_module(x)
        x = xp.concatenate((x_0, x_1, x_2), axis=1)
        return self.score_map(x), self.reg_map(x)

# Convert voxels to dense tensor
class Unravel(Function):
    def __init__(self, dims):
        self.dims = dims

    def forward(self, inputs):
        x, coords = inputs
        xp = cuda.get_array_module(x)
        batch_size = len(coords)
        y = xp.zeros(x.shape[-1], batch_size, *self.dims)
        y[coords[:,0], :, coords[:,1], coords[:,2], coords[:,3]] = x
        return y
    
    def backward(self, inputs, grad_outputs):
        x, coords = inputs
        gy, = grad_outputs
        return gy[coords[:,0], :, coords[:,1], coords[:,2], coords[:,3]], None

class VoxelNet(Chain):
    def __init__(self, A, T, V, B, **discard):
        '''
        A: Anchors per position
        T: Maxiumum number of points per voxel
        V: Size of a voxel (int or tuple)
        B: Bouding box (tuple or 3-tuple list)
        K: maximum number of non-empty voxels
        discard: discard extra args
        '''
        super(VoxelNet, self).__init__()
        self.W, self.H, self.D = compute_voxelgrid_size(
            ensure_bounding(B), ensure_voxel(V))

        with self.init_scope():
            self.svfe = SVFE(T)
            self.cml = CML()
            self.rpn = RPN(A)
            self.dense = Unravel((self.D, self.H, self.W))

    def __call__(self, voxel_features, voxel_coords):
        # feature learning network
        vwfs = self.svfe(voxel_features)
        vwfs = self.dense(vwfs, voxel_coords)
        # convolutional middle network
        cml_out = self.cml(vwfs)
        # merge the depth and feature dim into one, output probability score map and regression map
        psm, rm = self.rpn(cml_out.reshape(len(voxel_coords), -1, self.H, self.W))

        return psm, rm
