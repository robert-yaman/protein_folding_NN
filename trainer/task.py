import argparse
import tensorflow as tf
import os
import sys

import data
import validate

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from model.model import ProteinFoldingModel

def main(args):
	with tf.device('/gpu:0'):
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

		accuracy = validate.accuracy(model.logits, labels)

		# We need to execute update_ops before training for batch normalization.
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			training_step = tf.train.AdamOptimizer(.001).minimize(loss)

		training_data, training_labels = data.get_training_data(
			args.train_files[0], int(args.num_epochs))
		validation_step, validation_initializer = data.get_validation_data(
			args.eval_files[0])
		(validation_data, validation_labels) = validation_step

		summary = tf.summary.merge_all()
	with tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True)) as sess:
		print "AVAILABLE DEVICES:"
		print [device.name for device in tf.python.clientdevice_lib.list_local_devices()]
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

				# Validation
				if step % 10000 == 0:
					print "VALIDATING..."
					sess.run([validation_initializer])
					losses = []
					accuracies = []
					count_val = 0
					while True:
						print count_val
						count_val += 1
						try:
							val_ex, val_label = sess.run([validation_data, 
								validation_labels])
							_loss, _accuracy = sess.run([loss, accuracy],
								feed_dict={example:val_ex, labels:val_label, 
								training:False})
							losses.append(_loss)
							accuracies.append(_accuracy)
						except tf.errors.OutOfRangeError:
							break
					total_loss = sum(losses) / float(len(losses))
					total_accuracy = sum(accuracies) / float(len(accuracies))
					print " -- TOTAL LOSS: " + str(total_loss)
					print " -- TOTAL ACCURACY: " + str(total_accuracy)

			except tf.errors.OutOfRangeError:
				print("DONE TRAINING")
				break
		

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--train-files',
		help='GCS or local paths to training data',
		nargs='+',
		required=True
	)
	parser.add_argument(
		'--eval-files',
		help='GCS or local paths to evaluation data',
		nargs='+',
	)
	parser.add_argument(
		'--job-dir',
		help='Location to write checkpoints, tensorboard, and models',
		default="/tmp/"
	)
	parser.add_argument(
		'--num-epochs',
		default=1
	)

	main(parser.parse_args())
