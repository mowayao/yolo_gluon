import sys
import os
import cv2
from YOLO.config import Config as cfg
import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
import mxnet.ndarray as F
from mxnet.gluon.model_zoo import vision
from YOLO.net import YOLO2Output, yolo2_target, yolo2_forward
import mxnet.gluon.nn as nn
import random
from mxnet.ndarray.contrib import MultiBoxDetection
def detect_image(img_path):
	if not os.path.exists(img_path):
		print('can not find image: ', img_path)
	# img = Image.open(img_file)
	#print img_path
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (cfg.img_size, cfg.img_size))

	# img = ImageOps.fit(img, [data_shape, data_shape], Image.ANTIALIAS)
	origin_img = img.copy()
	img = F.array(img)
	img = (img/255. - cfg.rgb_mean) / cfg.rgb_std
	img = F.transpose(img, (2, 0, 1))

	img = F.expand_dims(img, axis=0)
	#img = img[np.newaxis, :]
	#img = F.array(img)

	print('input image shape: ', img.shape)


	#net = build_ssd("test", 300, ctx)
	#net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
	#net.collect_params().reset_ctx(ctx)
	#params = 'model/ssd.params'
	#net.load_params(params, ctx=ctx)

	#anchors, cls_preds, box_preds = net(img.as_in_context(ctx))
	pretrained = vision.get_model('resnet34_v2', pretrained=True).features
	net = nn.HybridSequential()

	for i in range(len(pretrained) - 2):
		net.add(pretrained[i])

	# anchor scales, try adjust it yourself
	scales = [[3.3004, 3.59034],
			  [9.84923, 8.23783]]

	# use 2 classes, 1 as dummy class, otherwise softmax won't work
	predictor = YOLO2Output(21, scales)
	predictor.initialize()
	net.add(predictor)

	ctx = mx.gpu(0)
	net.collect_params().reset_ctx(ctx)
	net.collect_params().load("gluon.params", ctx=ctx)
	#net.load_params("gluon.params", ctx=ctx)
	#net.collect_params().load
	img = img.as_in_context(ctx)
	x = net(img)
	output, cls_prob, score, xywh = yolo2_forward(x, 21, scales)

	output =  F.contrib.box_nms(output.reshape((0, -1, 6))).asnumpy()


	pens = dict()

	plt.imshow(origin_img)

	thresh = 0.8
	for det in output[0]:
		cid = int(det[0])
		if cid < 0:
			continue
		score = det[1]
		if score < thresh:
			continue
		if cid not in pens:
			pens[cid] = (random.random(), random.random(), random.random())
		scales = [origin_img.shape[1], origin_img.shape[0]] * 2
		xmin, ymin, xmax, ymax = [int(p * s) for p, s in zip(det[2:6].tolist(), scales)]
		rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=pens[cid], linewidth=3)
		plt.gca().add_patch(rect)
		voc_class_name = ['person', 'bird', 'cat', 'cow', 'dog',
						  'horse', 'sheep', 'aeroplane', 'bicycle', 'boat',
						  'bus', 'car', 'motorbike', 'train', 'bottle',
						  'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
		text = voc_class_name[cid]
		plt.gca().text(xmin, ymin - 2, '{:s} {:.3f}'.format(text, score),
					   bbox=dict(facecolor=pens[cid], alpha=0.5),
					   fontsize=12, color='white')
	plt.axis('off')
	# plt.savefig('result.png', dpi=100)
	plt.show()
if __name__ == '__main__':
	if len(sys.argv[1]) != 0:
		img_path =sys.argv[1]
		#try:
		detect_image(img_path)
		#except:
			#pass
		#except Exception as e:
			#print(e)
			#print('for detect please provide image file path.')