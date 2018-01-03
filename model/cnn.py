import tensorflow as tf

NUM_CHANNELS = 64

def cnns(inputs, training):
	# Inputs is [700, 72]
	# Three convolution layers with strides 3,7, and 11. 64 filters each.
	# Concat at the end for [700, 64*3=192]

	# Add dimensions to end of the tensor for |channels| params.
	inputs_expanded = tf.expand_dims(inputs, -1)

	# We can't have different padding styles for different dimensions, we we
	# manually add padding to dim=0 and use VALID style.
	conv3_filter = tf.get_variable("conv3_filter", [3, 72, 1, NUM_CHANNELS],
		initializer=tf.truncated_normal_initializer(stddev=.1))
	inputs_expanded_padding3 = tf.pad(inputs_expanded, 
		[[0,0],[1,1],[0,0],[0,0]])
	conv3 = tf.nn.conv2d(inputs_expanded_padding3, conv3_filter,
		strides=[1,1,1,1], padding='VALID')
	conv3_biases = tf.get_variable("conv3_biases", [NUM_CHANNELS], 
				initializer=tf.constant_initializer(0.1))
	conv3_layer = tf.nn.relu(conv3 + conv3_biases)

	conv7_filter = tf.get_variable("conv7_filter", [7, 72, 1, NUM_CHANNELS],
		initializer=tf.truncated_normal_initializer(stddev=.1))
	inputs_expanded_padding7 = tf.pad(inputs_expanded, 
		[[0,0],[3,3],[0,0],[0,0]])
	conv7 = tf.nn.conv2d(inputs_expanded_padding7, conv7_filter,
		strides=[1,1,1,1], padding='VALID')
	conv7_biases = tf.get_variable("conv7_biases", [NUM_CHANNELS], 
				initializer=tf.constant_initializer(0.1))
	conv7_layer = tf.nn.relu(conv7 + conv7_biases)

	conv11_filter = tf.get_variable("conv11_filter", [11, 72, 1, NUM_CHANNELS],
		initializer=tf.truncated_normal_initializer(stddev=.1))
	inputs_expanded_padding11 = tf.pad(inputs_expanded, 
		[[0,0],[5,5],[0,0],[0,0]])
	conv11 = tf.nn.conv2d(inputs_expanded_padding11, conv11_filter,
		strides=[1,1,1,1], padding='VALID')
	conv11_biases = tf.get_variable("conv11_biases", [NUM_CHANNELS], 
				initializer=tf.constant_initializer(0.1))
	conv11_layer = tf.nn.relu(conv11 + conv11_biases)

	conv_layers = [conv3_layer, conv7_layer, conv11_layer]
	conv_layers = [tf.squeeze(layer, axis=2) for layer in conv_layers]
	conv_layers = [tf.contrib.layers.batch_norm(layer, center=True, scale=True,
		is_training=training) for layer in conv_layers]		

	concat_layer = tf.concat(conv_layers, axis=2)

	return tf.contrib.layers.batch_norm(concat_layer, center=True, scale=True,
		is_training=training)
	