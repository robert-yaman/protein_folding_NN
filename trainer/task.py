import argparse
import tensorflow as tf
import os
import sys

import data

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from model.model import ProteinFoldingModel

def main(args):
	# Since we are doing batch normalization, we have to keep track of whether
	# we are training or testing. Also determines dropout probability.
	training = tf.placeholder(tf.bool, name='training')
	example = tf.placeholder(tf.float32, [700, 22])
	labels = tf.placeholder(tf.float32, [700, 9])
	model = ProteinFoldingModel(example, training)
	losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, 
		logits=model.logits)
	loss = tf.reduce_mean(losses)
	tf.summary.scalar('loss', loss)

	# We need to execute update_ops before training for batch normalization.
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		training_step = tf.train.AdamOptimizer(.001).minimize(loss)

	training_data, training_labels = data.get_training_data()

	summary = tf.summary.merge_all()
	with tf.Session() as sess:
		print "BEGINNING TRANING..."
		sess.run(tf.global_variables_initializer())
		summary_writer = tf.summary.FileWriter(args.job_dir, sess.graph)
		step = 0
		while True:
			try:
			    step += 1
			    print step
			    itr_ex, itr_label = sess.run([training_data, training_labels])
			    _, s, l = sess.run([training_step, summary, loss], 
			    	feed_dict={example:itr_ex, labels:itr_label, 
			    	training: True})
			    # Log every step for now
			    summary_writer.add_summary(s, step)
			    print "LOSS: " + str(l)
			except tf.errors.OutOfRangeError:
			    print("DONE TRAINING")
			    break

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--job-dir',
		help='GCS location to write checkpoints, tensorboard, and models',
		default="/tmp/"
	)

	main(parser.parse_args())
