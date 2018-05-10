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
        with self.init_scope():
            self.linear = L.Linear(cin, cout)
            self.bn = L.BatchNormalization(cout)

    def __call__(self, x):
        b, kk, t, _ = x.shape # KK is the stacked k across batch
        x = self.linear(x.reshape(b * kk * t, -1))
        x = self.bn(x)
        x = F.relu(x)
        return x.reshape(b, kk, t, -1)

# Voxel Feature Encoding layer
class VFE(Chain):
    def __init__(self, cin, cout, T):
        # (n_voxel, T, cin) -> (n_voxel, T, cout)
        super(VFE, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.T = T

        with self.init_scope():
            self.fcn = FCN(cin, self.units)

    def __call__(self, x, mask):
        xp = cuda.get_array_module(x)

        # point-wise feauture
        pwf = self.fcn(x)
        # locally aggregated feature
        laf = F.repeat(F.expand_dims(F.max(pwf, axis=2), 2), self.T, axis=2)
        # point-wise concat feature
        pwcf = F.concat((pwf, laf), axis=-1)
        # apply mask
        mask = xp.expand_dims(mask, -1).repeat(self.units * 2, axis=-1)
        pwcf *= mask

        return pwcf

# Stacked Voxel Feature Encoding
class SVFE(Chain):
    def __init__(self, T):
        # (n_voxel, T, 7) -> (n_voxel, 128)
        super(SVFE, self).__init__()
        with self.init_scope():
            self.vfe_1 = VFE(7, 32, T) 
            self.vfe_2 = VFE(32, 128, T)
            self.fcn = FCN(128, 128)

    def __call__(self, x):
        xp = cuda.get_array_module(x)

        # mask for filled points (in voxel with points less than T)
        mask = xp.max(x, axis=-1) != 0
        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)
        x = F.max(x, axis=2) # element-wise max pooling
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

# Convert voxels to dense tensor
# TODO: do voxel convolution using sparse matrix
class Unravel(Function):
    # TODO: emit K limit in test phase (use all information)
    def __init__(self, D, H, W):
        # (n_voxel, voxel_feature_size) -> (voxel_feature_size, D, H, W)
        self.dims = D, H, W

    def forward(self, inputs):
        x, coords, mask = inputs
        xp = cuda.get_array_module(x)

        batch_size = len(coords)
        y = xp.zeros((batch_size, x.shape[-1]) + self.dims, dtype=x.dtype)
        for b in range(batch_size):
            bcoord = coords[b][mask[b]] # XXX: cupy doesn't support coords[b, mask[b]]
            y[b, :, bcoord[:, 0], bcoord[:, 1], bcoord[:, 2]] = x[b][mask[b]]
        return y,
    
    def backward(self, inputs, grad_outputs):
        x, coords, mask = inputs
        gy, = grad_outputs
        xp = cuda.get_array_module(x)

        batch_size = len(coords)
        gx = xp.zeros_like(x, dtype=gy.dtype)
        for b in range(batch_size):
            bcoord = coords[b][mask[b]]
            gx[b][mask[b]] = gy[b, :, bcoord[:, 0], bcoord[:, 1], bcoord[:, 2]]
        return gx, None, None

# Region Proposal Network
# TODO: fix block params
class RPN(Chain):
    def __init__(self, anchors_per_position):
        super(RPN, self).__init__()

        self.block_1s = [Conv2d(None, 128, 3, 2, 1)] # refer from output
        self.block_1s += [Conv2d(128, 128, 3, 1, 1) for _ in range(3)]
        self.block_2s = [Conv2d(128, 128, 3, 2, 1)]
        self.block_2s += [Conv2d(128, 128, 3, 1, 1) for _ in range(5)]
        self.block_3s = [Conv2d(128, 256, 3, 2, 1)]
        self.block_3s += [Conv2d(256, 256, 3, 1, 1) for _ in range(5)]

        with self.init_scope():
            self.block_1 = Sequential(*self.block_1s)
            self.block_2 = Sequential(*self.block_2s)
            self.block_3 = Sequential(*self.block_3s)

            self.deconv_1 = Sequential(L.Deconvolution2D(256, 256, 4, 4, 0), L.BatchNormalization(256))
            self.deconv_2 = Sequential(L.Deconvolution2D(128, 256, 2, 2, 0), L.BatchNormalization(256))
            self.deconv_3 = Sequential(L.Deconvolution2D(128, 256, 1, 1, 0), L.BatchNormalization(256))

            self.score_map = Conv2d(768, anchors_per_position, 1, 1, 0, activation=False, batch_norm=False)
            self.reg_map = Conv2d(768, 7 * anchors_per_position, 1, 1, 0, activation=False, batch_norm=False)

    def __call__(self, x):
        x = x_skip_1 = self.block_1(x)
        x = x_skip_2 = self.block_2(x)
        x = self.block_3(x)
        x_0 = self.deconv_1(x)
        x_1 = self.deconv_2(x_skip_2)
        x_2 = self.deconv_3(x_skip_1)

        x = F.concat((x_0, x_1, x_2), axis=1)
        return self.score_map(x), self.reg_map(x)

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
        
        print("VoxelNet Params:")
        print("    Bounds in x:", B[0])
        print("    Bounds in y:", B[1])
        print("    Bounds in z:", B[2])
        print("    D, H, W:", self.D, self.H, self.W)
        print("    T:", T)

        if self.W % 8 !=0 or self.H % 8 != 0:
            # this will make it unable to concatenate rpn feature blocks
            raise ValueError("W and H should be able to be divided by 8")

        with self.init_scope():
            self.svfe = SVFE(T)
            self.cml = CML()
            self.rpn = RPN(A)
            self.dense = Unravel(self.D, self.H, self.W)

    def __call__(self, voxel_features, voxel_coords):
        xp = cuda.get_array_module(voxel_features)
        vmask = xp.max(voxel_features, axis=(-1, -2)) != 0

        # feature learning network
        vwfs = self.svfe(voxel_features) # TODO: add voxel mask
        vwfs = self.dense(vwfs, voxel_coords, vmask)
        # convolutional middle network
        cml_out = self.cml(vwfs)
        # merge the depth and feature dim into one, output probability score map and regression map
        batch_size = len(voxel_coords)
        psm, rm = self.rpn(cml_out.reshape(batch_size, -1, self.H, self.W))

        return psm, rm
