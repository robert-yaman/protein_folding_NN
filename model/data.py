import tensorflow as tf
import numpy as np

def get_data(path):
	np_data = np.load(path)
	np_data = np_data.reshape(len(np_data), 700, 57)

	features = np_data[...,  :22]
	# does this go to 30 or 31?
		# 31 because there is an extra value for features and labels for blanks
	labels = np_data[..., 22 : 31]

	dataset = tf.data.Dataset.from_tensor_slices((features, labels))
	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()

	return next_element[0], next_element[1]