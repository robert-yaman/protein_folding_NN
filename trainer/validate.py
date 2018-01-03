import tensorflow as tf

def accuracy(logits, labels):
	# Each input has shape [batch_size, 700, 9]
	prediction_indices = tf.argmax(logits, axis=2)
	label_indices = tf.argmax(labels, axis=2)
	correct_predictions = tf.equal(prediction_indices, label_indices)
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
	tf.summary.scalar('accuracy', accuracy)
	return accuracy
