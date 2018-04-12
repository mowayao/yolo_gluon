import mxnet.ndarray as nd

class Config:
	batch_size = 32
	rgb_mean = nd.array([123, 117, 104])
	rgb_std = nd.array([58.395, 57.12, 57.375])

	annotation_dir = "/media/mowayao/data/object_detection/VOC2007/VOC2007/Annotations"
	img_dir = "/media/mowayao/data/object_detection/VOC2007/VOC2007/JPEGImages"
	dataset_index = "/media/mowayao/data/object_detection/VOC2007/VOC2007/ImageSets/Main/train.txt"

	img_size = 256