import tensorflow as tf
import numpy as np

TRAINING_PATH = "data/cullpdb+profile_6133_filtered.npy"

def get_training_data():
	return _get_data(TRAINING_PATH)

def _get_data(path):
	np_data = np.load(path)
	np_data = np_data.reshape(len(np_data), 700, 57)

	features = np_data[...,  :22]
	# does this go to 30 or 31?
		# 31 because there is an extra value for features and labels for blanks
	labels = np_data[..., 22 : 31]

	dataset = tf.data.Dataset.from_tensor_slices((features, labels))
	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()

	example = tf.cast(next_element[0], dtype=tf.float32)
	label = tf.cast(next_element[1], dtype=tf.float32)
	
	return example, label