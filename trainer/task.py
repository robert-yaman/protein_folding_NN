import argparse
import tensorflow as tf
import os
import sys

import data

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from model.model import ProteinFoldingModel

def main(args):
	# Since we are doing batch normalization, we have to keep track of whether
	# we are training or testing.
	training = tf.placeholder(tf.bool, name='training')
	training_data, training_labels = data.get_training_data()
	model = ProteinFoldingModel(training_data, training)

	# We need to execute update_ops before training for batch normalization.
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	# 	with tf.control_dependencies(update_ops):
    	# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	main(parser.parse_args())
