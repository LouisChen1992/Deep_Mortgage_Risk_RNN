import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicRNNCell, LSTMCell, GRUCell, DropoutWrapper, MultiRNNCell

def create_rnn_cell(cell_type,
					num_units,
					num_layers=1,
					dp_input_keep_prob=1.0,
					dp_output_keep_prob=1.0,
					activation=None):

	def single_cell(num_units):
		if cell_type == 'rnn':
			cell_class = BasicRNNCell
		elif cell_type == 'lstm':
			cell_class = LSTMCell
		elif cell_type == 'gru':
			cell_class = GRUCell
		else:
			raise ValueError('Cell Type Not Supported! ')

		if activation is not None:
			if activation == 'relu':
				activation_f = tf.nn.relu
			elif activation == 'sigmoid':
				activation_f = tf.sigmoid
			elif activation == 'elu':
				activation_f = tf.nn.elu
			else:
				raise ValueError('Activation Function Not Supported! ')
		else:
			activation_f = None

		if dp_input_keep_prob != 1.0 or dp_output_keep_prob != 1.0:
			return DropoutWrapper(cell_class(num_units=num_units, activation=activation_f),
								input_keep_prob=dp_input_keep_prob,
								output_keep_prob=dp_output_keep_prob)
		else:
			return cell_class(num_units=num_units)

	if isinstance(num_units, list):
		num_layers = len(num_units)
		if num_layers > 1:
			return MultiRNNCell([single_cell(num_units[i]) for i in range(num_layers)])
		else:
			return single_cell(num_units[0])
	else:
		if num_layers > 1:
			return MultiRNNCell([single_cell(num_units) for _ in range(num_layers)])
		else:
			return single_cell(num_units)

def initial_state_size(cell_type, num_units, num_layers=1):
	if isinstance(num_units, list):
		state_size = sum(num_units)
	else:
		state_size = num_units * num_layers
	if cell_type == 'rnn' or cell_type == 'gru':
		return state_size
	elif cell_type == 'lstm':
		return state_size * 2