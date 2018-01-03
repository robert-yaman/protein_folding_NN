import tensorflow as tf

import embedding
import fc
import cnn
import rnn

class ProteinFoldingModel(object):
	def __init__(self, input_sequence, input_profile, training):
		self.embedding_layer = embedding.embedding(input_sequence)
		self.input_layer = tf.concat([self.embedding_layer, input_profile], 
			axis=2)
		self.cnn_layer = cnn.cnns(self.input_layer, training)
		self.rnn_layer = self._recurrentLayer(training)
		self.logits = fc.fc(self.rnn_layer, self.cnn_layer, training)
		self.readout = tf.nn.softmax(self.logits)

		tf.summary.histogram('logits', self.logits)

	def _recurrentLayer(self, training):
		return rnn.rnns(self.cnn_layer, training)
