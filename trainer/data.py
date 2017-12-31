import tensorflow as tf
import numpy as np

# TRAINING_PATH = "data/cullpdb+profile_6133_filtered.npy"
# TESTING_PATH = "data/cb513+profile_split1.npy"

def _decode_line(line):
	items = tf.decode_csv(line, [[0.0]] * 39900)
	matrix = tf.reshape(items, [700, 57])
	# Goes to 31 because there is an extra value in features and labels for 
	# blanks.
	return matrix[..., :22], matrix[..., 22:31]

def get_training_data(path, num_epochs, batch_size):
	base_dataset = tf.data.TextLineDataset(path)
	tr_data = base_dataset.map(_decode_line).batch(batch_size
		).repeat(num_epochs)
	iterator = tr_data.make_one_shot_iterator()
	next_element = iterator.get_next()

	return next_element

def get_validation_data(path, batch_size):
	base_dataset = tf.data.TextLineDataset(path)
	val_data = base_dataset.map(_decode_line).batch(batch_size)
	# Make an initializable interator so that we can use the whole dataset for
	# each validation step. 
	iterator = val_data.make_initializable_iterator()
	next_element = iterator.get_next()

	return next_element, iterator.initializer