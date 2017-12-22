import tensorflow as tf
import numpy as np

TRAINING_PATH = "data/cullpdb+profile_6133_filtered.npy"
TESTING_PATH = "data/cb513+profile_split1.npy"

def get_training_data():
	np_data = np.load(TRAINING_PATH)
	np_data = np_data.reshape(len(np_data), 700, 57)
	features = np_data[0:5022, ..., :22]
	# Goes to 31 because there is an extra value in features and labels for 
	# blanks.
	labels = np_data[0:5022, ..., 22 : 31]
	return _dataset(features, labels)

def get_validation_data():
	np_data = np.load(TRAINING_PATH)
	np_data = np_data.reshape(len(np_data), 700, 57)
	features = np_data[5022:, ..., :22]
	# Goes to 31 because there is an extra value in features and labels for 
	# blanks.
	labels = np_data[5022:, ..., 22 : 31]
	return _dataset(features, labels)

def get_testing_data():
	np_data = np.load(TESTING_PATH)
	np_data = np_data.reshape(len(np_data), 700, 57)
	features = np_data[..., :22]
	# Goes to 31 because there is an extra value in features and labels for 
	# blanks.
	labels = np_data[..., 22 : 31]
	return _dataset(features, labels)

def _dataset(features, labels):
	dataset = tf.data.Dataset.from_tensor_slices((features, labels))
	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()

	example = tf.cast(next_element[0], dtype=tf.float32)
	label = tf.cast(next_element[1], dtype=tf.float32)

	return example, label