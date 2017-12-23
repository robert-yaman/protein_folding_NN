import tensorflow as tf

def accuracy(logits, labels):
	prediction_indices = tf.argmax(logits, axis=1)
	label_indices = tf.argmax(labels, axis=1)
	correct_predictions = tf.equal(prediction_indices, label_indices)
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
	tf.summary.scalar('accuracy', accuracy)
	return accuracy
