
from mxnet import nd
import cv2
from YOLO.config import Config as cfg
import random
from mxnet import image as mx_img
def transform_center(xy):
    """Given x, y prediction after sigmoid(), convert to relative coordinates (0, 1) on image."""
    b, h, w, n, s = xy.shape
    offset_y = nd.tile(nd.arange(0, h, repeat=(w * n * 1), ctx=xy.context).reshape((1, h, w, n, 1)), (b, 1, 1, 1, 1))
    # print(offset_y[0].asnumpy()[:, :, 0, 0])
    offset_x = nd.tile(nd.arange(0, w, repeat=(n * 1), ctx=xy.context).reshape((1, 1, w, n, 1)), (b, h, 1, 1, 1))
    # print(offset_x[0].asnumpy()[:, :, 0, 0])
    x, y = xy.split(num_outputs=2, axis=-1)
    x = (x + offset_x) / w
    y = (y + offset_y) / h
    return x, y

def transform_size(wh, anchors):
    """Given w, h prediction after exp() and anchor sizes, convert to relative width/height (0, 1) on image"""
    b, h, w, n, s = wh.shape
    aw, ah = nd.tile(nd.array(anchors, ctx=wh.context).reshape((1, 1, 1, -1, 2)), (b, h, w, 1, 1)).split(num_outputs=2, axis=-1)
    w_pred, h_pred = nd.exp(wh).split(num_outputs=2, axis=-1)
    w_out = w_pred * aw / w
    h_out = h_pred * ah / h
    return w_out, h_out


def corner2center(boxes, concat=True):
    """Convert left/top/right/bottom style boxes into x/y/w/h format"""
    left, top, right, bottom = boxes.split(axis=-1, num_outputs=4)
    x = (left + right) / 2
    y = (top + bottom) / 2
    width = right - left
    height = bottom - top
    if concat:
        last_dim = len(x.shape) - 1
        return nd.concat(*[x, y, width, height], dim=last_dim)
    return x, y, width, height

def center2corner(boxes, concat=True):
    """Convert x/y/w/h style boxes into left/top/right/bottom format"""
    x, y, w, h = boxes.split(axis=-1, num_outputs=4)
    w2 = w / 2
    h2 = h / 2
    left = x - w2
    top = y - h2
    right = x + w2
    bottom = y + h2
    if concat:
        last_dim = len(left.shape) - 1
        return nd.concat(*[left, top, right, bottom], dim=last_dim)
    return left, top, right, bottom


def random_flip(data, label):
	p = random.random()
	h, w, c = data.shape
	if p < 0.5:
		data = cv2.flip(data, 1)
		x1 = label[:, 1].copy()
		x3 = label[:, 3].copy()
		label[:, 1] = w - x3
		label[:, 3] = w - x1
	return data, label

def img_resize(img):
	h, w, c = img.shape
	img = cv2.resize(img, (cfg.img_size, cfg.img_size))
	return img, [cfg.img_size/float(w), cfg.img_size/float(h)]

def random_crop(img, label):
	#TODO:
	return img, label

def img_norm(img, mean, std):
	#img is a ndarray
	img = nd.array(img) / 255.
	return mx_img.color_normalize(img, nd.array(mean), nd.array(std))

def transformation(data, label):

    data, label = random_flip(data, label)
    #data, label = random_square_crop(data, label)
    return data, label


def img_resize(img):
    h, w, c = img.shape
    img = cv2.resize(img, (cfg.img_size, cfg.img_size))
    return img, [cfg.img_size/float(w), cfg.img_size/float(h)]