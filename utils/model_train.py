from chainer import Chain, reporter
from chainer.backends import cuda
import chainer.functions as F

class CRMapDetector(Chain):
    '''
    Detector with classificaion & regression map as output
    '''
    def __init__(self, base, alpha, beta, **kwargs):
        super(CRMapDetector, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        with self.init_scope():
            self.net = base(**kwargs)
    
    def __call__(self, voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets):
        psm, rm = self.net(voxel_features, voxel_coords)

        p_pos = F.sigmoid(psm.transpose(0, 2, 3, 1))
        rm = rm.transpose(0, 2, 3, 1)
        rm = rm.reshape(rm.shape[:3] + (-1, 7))
        targets = targets.reshape(targets.shape[:3] + (-1, 7))
        pos_equal_one_for_reg = F.tile(F.expand_dims(pos_equal_one, -1), 7)
        
        rm_pos = rm * pos_equal_one_for_reg
        targets_pos = targets * pos_equal_one_for_reg

        cls_pos_loss = -pos_equal_one * F.log(p_pos + 1e-6)
        cls_pos_loss = F.sum(cls_pos_loss) / (F.sum(pos_equal_one) + 1e-6)

        cls_neg_loss = -neg_equal_one * F.log(1 - p_pos + 1e-6)
        cls_neg_loss = F.sum(cls_neg_loss) / (F.sum(neg_equal_one) + 1e-6)

        reg_loss = F.huber_loss(rm_pos, targets_pos, 1, 'no')
        reg_loss = F.sum(reg_loss) / (F.sum(pos_equal_one) + 1e-6)
        conf_loss = self.alpha * cls_pos_loss + self.beta * cls_neg_loss

        reporter.report({'conf_loss': conf_loss,
                         'reg_loss': reg_loss})
        return reg_loss + conf_loss
