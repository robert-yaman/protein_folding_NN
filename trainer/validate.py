import tensorflow as tf

def accuracy(logits, labels):
	# Each input has shape [batch_size, 700, 9]
	prediction_indices = tf.argmax(logits, axis=2)
	label_indices = tf.argmax(labels, axis=2)
	correct_predictions = tf.equal(prediction_indices, label_indices)
	accuracy_sum = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))

	# An 8 means that the protein has terminated. Don't count 8 predictions for
	# accuracy.
	terminated = tf.equal(label_indices, 8)
	total_terminated = tf.reduce_sum(tf.cast(terminated, tf.float32))
	total_examples = tf.cast(tf.size(label_indices), tf.float32)
	accuracy = (accuracy_sum - total_terminated) / (total_examples - 
		total_terminated)

	return accuracy
