import data
import embedding
import cnn

class ProteinFoldingModel(object):
	def __init__(self, input_tensor, training):
		embedding_layer = embedding.embedding(input_tensor)
		cnn_layer = cnn.cnns(embedding_layer, training)
		print "CURRENT SHAPE: "
		print cnn_layer.shape
