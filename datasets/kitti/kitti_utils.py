
import numpy as np
import numpy.random as npr
import numpy.linalg as npl
from collections import namedtuple

from .box_overlaps import bbox_overlaps

Label = namedtuple(
    'Label', ['name', 'id', 'trainId', 'color'])

kitti_labels = tuple([
    Label('DontCare', 0, 0, (0, 0, 0)),
    Label('Car', 1, 1, (0, 0, 0)),
    Label('Van', 2, 2, (0, 0, 0)),
    Label('Truck', 3, 3, (0, 0, 0)),
    Label('Pedestrian', 4, 4, (0, 0, 0)),
    Label('Person_sitting', 5, 5, (0, 0, 0)),
    Label('Cyclist', 6, 6, (0, 0, 0)),
    Label('Tram', 7, 7, (0, 0, 0)),
    Label('Misc', 8, 255, (0, 0, 0)),
])

def label_cam_to_lidar(box_labels, Tr):
    '''
    Convert bounding box from camera coordinate to lidar coordinate
    '''
    h, w, l, tx, ty, tz, ry = box_labels

    # project center from cam to lidar
    cam = np.array([tx, ty, tz, 1]).reshape(-1, 1)
    T = np.vstack((Tr, [0, 0, 0, 1]))
    T_inv = npl.inv(T)
    lidar_loc = np.dot(T_inv, cam)
    tx, ty, tz = lidar_loc[:3].reshape(-1)

    # ry in camera => rz in lidar
    rz = -ry - np.pi / 2
    if rz >= np.pi:
        rz -= np.pi
    if rz < -np.pi:
        rz = 2*np.pi + rz

    # reorder to fit anchors
    return tx, ty, tz, h, w, l, rz

def ensure_bounding(B):
    '''
    B: Bouding box (tuple or 3-tuple list)
        [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
    '''
    if isinstance(B[0], (tuple, list)):
        return np.array(B)
    elif isinstance(B, (tuple, list)):
        return np.array([B]*3)
    else:
        raise ValueError("Invalid bounding box parameter!")

def ensure_voxel(V):
    '''
    V: Size of a voxel (int or tuple)
        (W, H, D) i.e. (vx, vy, vz)
    '''
    if isinstance(V, (tuple, list)):
        return np.array(V)
    elif np.isreal(V):
        return np.array([V]*3)
    else:
        raise ValueError("Invalid voxel size parameter!")

def compute_voxelgrid_size(B, V):
    '''
    should call _ensure_XXX on B, V first
    return W, H, D
    '''
    return np.around((B[:, 1]- B[:, 0]) / V).astype(int)

def center_to_bounding_2d(boxes_center):
    # (N, 7) -> (N, 4)
    anchor_box = []
    for anchor in boxes_center:
        tx, ty, tz, h, w, l, rz = anchor
        box = np.array([
            [-l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2]])

        rotMat = np.array([
            [np.cos(rz), -np.sin(rz)],
            [np.sin(rz), np.cos(rz)]])
        velo_box = np.dot(rotMat, box) + np.array([[tx], [ty]]) # broadcast
        assert len(velo_box.shape) == 2

        bound_box = np.hstack((
            np.min(velo_box, axis=1),
            np.max(velo_box, axis=1)
        ))
        anchor_box.append(bound_box)
    return np.array(anchor_box)

class VoxelPreprocessor:
    def __init__(self, B, V, T, K, target='Car', **discard):
        '''
        T: Maximum number of points per voxel
        K: Maximum number of non-empty voxels
        target: category of the object to train, `None` indicates test phase
        '''
        self.bounds = ensure_bounding(B)
        self.vsize = ensure_voxel(V)
        self.pointpv = T
        self.maxv = K
        self.type = target

    def __call__(self, lidar, labels, calib):
        # shuffle points in the cloud
        npr.shuffle(lidar)

        # TODO: Augment and truncate point cloud

        # calculate voxel location
        voxel_coords = (lidar[:, :3] - self.bounds[:, 0]) / self.vsize
        # convert to (D, H, W)
        voxel_coords = voxel_coords.astype(int)[:, [2, 1, 0]]

        # group points, TODO: speed up using cupy
        voxel_coords, inv_idx = np.unique(voxel_coords,
            axis=0, return_inverse=True)

        # calculate voxel features
        voxel_counter = 0
        voxel_features = np.empty((self.maxv, self.pointpv, 7), dtype=np.float32)
        voxel_coords_final = np.empty((self.maxv, 3), dtype=int)
        for i in npr.permutation(len(voxel_coords))[:self.maxv]: # shuffle
            # select points within the voxel
            voxel = np.empty((self.pointpv, 7), dtype=np.float32)
            pts = lidar[inv_idx == i]

            # remove the part of points more than T
            if len(pts) > self.pointpv:
                pts = pts[:self.pointpv, :]

            # augment each point with the relative offset
            voxel[:len(pts), :] = np.hstack((pts, pts[:, :3] - np.mean(pts[:, :3], 0)))
            voxel[len(pts):, :] = 0 # pad points

            voxel_features[voxel_counter, :, :] = voxel
            voxel_coords_final[voxel_counter, :] = voxel_coords[i]
            voxel_counter += 1
        voxel_features[voxel_counter:, :, :] = 0 # pad voxels

        if self.type:
            # transform ground truth
            Tr = calib['Tr_velo_to_cam'].reshape(3, 4)
            gt_params = [label_cam_to_lidar(l[8:], Tr) for l in labels if l[0] == self.type]
            
            return voxel_features, voxel_coords_final, gt_params
        else:
            return voxel_features, voxel_coords_final

class AnchorPreprocessor:
    def __init__(self, B, V, A, AS, pos_thres, neg_thres, AZ=2, dtype='f4', **discard):
        '''
        A: Anchors per position
        AS: Anchor size (W, L, H)
        AZ: Anchor center at Z direction
        pos_thres: overlap threshold of positive anchor
        neg_thres: overlap threshold of negative anchor
        '''
        # init params
        self.anchorpp = A
        self.bounds = ensure_bounding(B)
        self.vsize = ensure_voxel(V)
        self.pos_threshold = pos_thres
        self.neg_threshold = neg_thres
        self.dtype = dtype

        # compute anchors
        W, H, D = compute_voxelgrid_size(self.bounds, self.vsize)
        x = np.linspace(self.bounds[0, 0] + self.vsize[0], self.bounds[0, 1] - self.vsize[0], W/2)
        y = np.linspace(self.bounds[1, 0] + self.vsize[1], self.bounds[1, 1] - self.vsize[1], H/2)
        cx, cy = np.meshgrid(x, y)
        cx = np.tile(cx[..., np.newaxis], A)
        cy = np.tile(cy[..., np.newaxis], A)
        cz = np.full_like(cx, AZ)
        w = np.full_like(cx, AS[0])
        l = np.full_like(cx, AS[1])
        h = np.full_like(cx, AS[2])
        r = np.ones_like(cx) * np.linspace(0, np.pi, A + 1)[:A]
        self.anchors = np.stack([cx, cy, cz, h, w, l, r], axis=-1) # shape: (H/2, W/2, A, 7)
        self.feature_map_shape = (H // 2, W // 2)

        # compute other used variable
        self.anchor_bbox = center_to_bounding_2d(self.anchors.reshape(-1, 7))
        self.anchorz = AZ # anchor Z
        self.anchord = np.sqrt(AS[0]**2 + AS[1]**2) # anchor dimension
        self.anchorsize = AS[2], AS[0], AS[1] # anchor size (H, W, L)

    def __call__(self, gt_boxes):
        pos_equal_one = np.zeros(self.feature_map_shape + (self.anchorpp,), dtype=self.dtype)
        neg_equal_one = np.zeros(self.feature_map_shape + (self.anchorpp,), dtype=self.dtype)
        targets = np.zeros(self.feature_map_shape + (self.anchorpp, 7), dtype=self.dtype)

        # return if there are no ground truth boxes
        if len(gt_boxes) == 0:
            return pos_equal_one, neg_equal_one, targets

        # compute overlaps
        gt_bbox = center_to_bounding_2d(gt_boxes)
        iou = bbox_overlaps(
            np.ascontiguousarray(self.anchor_bbox).astype(np.float32),
            np.ascontiguousarray(gt_bbox).astype(np.float32),
        )

        # mark anchors with highest overlap
        id_highest = np.argmax(iou, axis=0)
        id_highest_gt = np.arange(iou.shape[1])
        mask = iou[id_highest, id_highest_gt] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # mark anchors by overlap thresholds
        id_pos, id_pos_gt = np.where(iou > self.pos_threshold)
        id_neg, = np.where(np.all(iou < self.neg_threshold, axis=0))

        # join positive anchors
        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]

        # index back into feature map
        index_x, index_y, index_z = np.unravel_index(
            id_neg, self.feature_map_shape + (self.anchorpp,))
        neg_equal_one[index_x, index_y, index_z] = 1
        index_x, index_y, index_z = np.unravel_index( # to avoid a box be pos/neg in the same time
            id_highest, self.feature_map_shape + (self.anchorpp,))
        neg_equal_one[index_x, index_y, index_z] = 0

        index_x, index_y, index_z = np.unravel_index(
            id_pos, self.feature_map_shape + (self.anchorpp,))
        pos_equal_one[index_x, index_y, index_z] = 1

        # compute box coefficients for positive anchors
        gt_boxes = np.array(gt_boxes, copy=False)
        targets[index_x, index_y, index_z, :] = np.array([
            (gt_boxes[id_pos_gt, 0] - self.anchors[index_x, index_y, index_z, 0]) / self.anchord,
            (gt_boxes[id_pos_gt, 1] - self.anchors[index_x, index_y, index_z, 1]) / self.anchord,
            (gt_boxes[id_pos_gt, 2] - self.anchorz) / self.anchorsize[0],
            np.log(gt_boxes[id_pos_gt, 3] / self.anchorsize[0]),
            np.log(gt_boxes[id_pos_gt, 4] / self.anchorsize[1]),
            np.log(gt_boxes[id_pos_gt, 5] / self.anchorsize[2]),
            (gt_boxes[id_pos_gt, 6] - self.anchors[index_x, index_y, index_z, 6])
        ], copy=False).T

        return pos_equal_one, neg_equal_one, targets

class VoxelRPNPreprocessor:
    def __init__(self, B, V, T, K, A, AS, pos_thres, neg_thres, AZ=2, train_cls='Car', **discard):
        self.pvoxel = VoxelPreprocessor(B, V, T, K, train_cls)
        self.panchor = AnchorPreprocessor(B, V, A, AS, pos_thres, neg_thres, AZ)

    def __call__(self, args):
        lidar, labels, calib = args
        voxel_features, voxel_coords, gt_params = self.pvoxel(lidar, labels, calib)
        pos_equal_one, neg_equal_one, targets = self.panchor(gt_params)

        # input shapes:
        #   voxel_features: (n_voxel, T, 7)
        #   voxel_coords: (n_voxel, 3)
        #   pos_equal_one: (H/2, W/2, 2)
        #   neg_equal_one: (H/2, W/2, A)
        #   targets: (H/2, W/2, A, 7)
        return voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets
