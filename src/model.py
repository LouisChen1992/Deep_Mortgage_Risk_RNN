import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell import BasicRNNCell, LSTMCell, GRUCell, DropoutWrapper, MultiRNNCell
from .utils import deco_print

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

class Model:
	def __init__(self, config, global_step=None, force_var_reuse=False, is_training=True, use_valid_set=False):
		self._config = config
		self._force_var_reuse = force_var_reuse
		self._is_training = is_training
		self._use_valid_set = use_valid_set
		self._global_step = global_step if global_step is not None else tf.contrib.framework.get_or_create_global_step()

		self._x_rnn_placeholder = tf.placeholder(dtype=tf.float32, shape=[self._config['global_batch_size'], None, self._config['feature_dim_rnn']], name='input_placeholder')
		self._x_ff_placeholder = tf.placeholder(dtype=tf.float32, shape=[self._config['global_batch_size'], None, self._config['feature_dim_ff']], name='input_placeholder')
		self._y_placeholder = tf.placeholder(dtype=tf.int32, shape=[self._config['global_batch_size'], None], name='output_placeholder')
		self._tDimSplit_placeholder = tf.placeholder(dtype=tf.int32, shape=[self._config['global_batch_size'], 3])

		xs_rnn = tf.split(value=self._x_rnn_placeholder, num_or_size_splits=self._config['num_gpus'], axis=0)
		xs_ff = tf.split(value=self._x_ff_placeholder, num_or_size_splits=self._config['num_gpus'], axis=0)
		ys = tf.split(value=self._y_placeholder, num_or_size_splits=self._config['num_gpus'], axis=0)
		tDimSplits = tf.split(value=self._tDimSplit_placeholder, num_or_size_splits=self._config['num_gpus'], axis=0)

		self._sum_loss = 0.0 # sum of loss in example
		self._num = 0 # count

		if self._is_training and self._use_valid_set:
			self._sum_loss_valid = 0.0
			self._num_valid = 0

		for gpu_idx in range(self._config['num_gpus']):
			with tf.device('/gpu:{}'.format(gpu_idx)), tf.variable_scope(
				name_or_scope=tf.get_variable_scope(),
				initializer=tf.random_uniform_initializer(minval=-0.5, maxval=0.5),
				reuse=self._force_var_reuse or (gpu_idx > 0)):

				deco_print('Building graph on GPU:{}'.format(gpu_idx))

				if self._is_training:
					x_length = tf.reduce_sum(tDimSplits[gpu_idx][:,:2], axis=1)
					logits_i = self._build_forward_pass_graph(x_rnn=xs_rnn[gpu_idx], x_ff=xs_ff[gpu_idx], x_length=x_length)
					loss_i_sum_train, loss_i_sum_valid = self._add_loss(logits=logits_i, labels=ys[gpu_idx], lengths=tDimSplits[gpu_idx][:,:2])
					self._sum_loss += loss_i_sum_train
					if not self._use_valid_set:
						self._num += tf.reduce_sum(tDimSplits[gpu_idx][:,:2])
					else:
						self._num += tf.reduce_sum(tDimSplits[gpu_idx][:,0])
						self._sum_loss_valid += loss_i_sum_valid
						self._num_valid += tf.reduce_sum(tDimSplits[gpu_idx][:,1])
				else:
					x_length = tf.reduce_sum(tDimSplits[gpu_idx], axis=1)
					logits_i = self._build_forward_pass_graph(x_rnn=xs_rnn[gpu_idx], x_ff=xs_ff[gpu_idx], x_length=x_length)
					self._sum_loss += self._add_loss(logits=logits_i, labels=ys[gpu_idx], lengths=tDimSplits[gpu_idx])
					self._num += tf.reduce_sum(tDimSplits[gpu_idx][:,2])

		self._loss = self._sum_loss / tf.to_float(self._num) # training or test loss

		if self._is_training:
			self._add_train_op()
			if self._use_valid_set:
				self._loss_valid = self._sum_loss_valid / tf.to_float(self._num_valid)

	def _build_forward_pass_graph(self, x_rnn, x_ff, x_length):
		with tf.variable_scope('{}_layer'.format(self._config['cell_type'])):
			rnn_cell = create_rnn_cell(
				cell_type=self._config['cell_type'],
				num_units=self._config['num_units_rnn'],
				num_layers=self._config['num_layers_rnn'],
				dp_input_keep_prob=self._config['dropout'],
				dp_output_keep_prob=1.0,
				activation=self._config['activation'] if 'activation' in self._config else None)
			outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=x_rnn, sequence_length=x_length, dtype=tf.float32)

		ts = tf.reduce_max(x_length)
		outputs_shape = outputs.get_shape()
		x_ff_shape = x_ff.get_shape()
		outputs_slice = tf.slice(outputs, begin=[0,0,0], size=[-1,ts,-1])
		x_ff_slice = tf.slice(x_ff, begin=[0,0,0], size=[-1,ts,-1])

		outputs = tf.concat([outputs_slice, x_ff_slice], axis=2)
		outputs.set_shape([outputs_shape[0], None, outputs_shape[2]+x_ff_shape[2]])

		# outputs = tf.concat([outputs, x_ff], axis=2)

		for l in range(self._config['num_layers_ff']):
			with tf.variable_scope('FF_layer_%d' %l):
				layer_l = Dense(units=self._config['num_units_ff'][l], activation=tf.nn.relu)
				outputs = tf.nn.dropout(layer_l(outputs), self._config['dropout'])

		with tf.variable_scope('Output_layer'):
			layer = Dense(units=self._config['num_category'])
			logits = layer(outputs)			

		return logits

	def _add_loss(self, logits, labels, lengths):
		if self._is_training:
			x_length_part = lengths[:,0]
			x_length = tf.reduce_sum(lengths, axis=1)
			ts = tf.reduce_max(x_length)

			logits = tf.slice(logits, begin=[0,0,0], size=[-1,ts,-1])
			labels = tf.slice(labels, begin=[0,0], size=[-1,ts])

			if self._use_valid_set:
				mask_train = tf.sequence_mask(lengths=x_length_part, maxlen=ts, dtype=tf.float32)
				mask = tf.sequence_mask(lengths=x_length, maxlen=ts, dtype=tf.float32)
				mask_valid = mask - mask_train
			else:
				mask_train = tf.sequence_mask(lengths=x_length, maxlen=ts, dtype=tf.float32)

			loss_sum_train = tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
				logits=logits,
				targets=labels,
				weights=mask_train,
				average_across_timesteps=False,
				average_across_batch=False,
				softmax_loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits))

			if self._use_valid_set:
				loss_sum_valid = tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
					logits=logits,
					targets=labels,
					weights=mask_valid,
					average_across_timesteps=False,
					average_across_batch=False,
					softmax_loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits))
				return loss_sum_train, loss_sum_valid
			else:
				return loss_sum_train, None
		else:
			x_length_part = tf.reduce_sum(lengths[:,:2], axis=1)
			x_length = tf.reduce_sum(lengths, axis=1)
			ts = tf.reduce_max(x_length)

			logits = tf.slice(logits, begin=[0,0,0], size=[-1,ts,-1])
			labels = tf.slice(labels, begin=[0,0], size=[-1,ts])

			mask = tf.sequence_mask(lengths=x_length, maxlen=ts, dtype=tf.float32)
			mask_part = tf.sequence_mask(lengths=x_length_part, maxlen=ts, dtype=tf.float32)
			mask_test = mask - mask_part

			loss_sum = tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
				logits=logits,
				targets=labels,
				weights=mask_test,
				average_across_timesteps=False,
				average_across_batch=False,
				softmax_loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits))
			return loss_sum, num
	
	def _add_train_op(self):
		deco_print('Trainable Variables')
		for var in tf.trainable_variables():
			deco_print('Name: {} \t Shape: {}'.format(var.name, var.get_shape()))

		if self._config['optimizer'] == 'Momentum':
			optimizer = lambda lr: tf.train.MomentumOptimizer(lr, momentum=0.9)
		elif self._config['optimizer'] == 'AdaDelta':
			optimizer = lambda lr: tf.train.AdadeltaOptimizer(lr, rho=0.95, epsilon=1e-08)
		else:
			optimizer = self._config['optimizer']

		if self._config['decay_steps'] and self._config['decay_rate']:
			learning_rate_decay_fn = lambda lr, global_step: tf.train.exponential_decay(
				learning_rate=lr,
				global_step=global_step,
				decay_steps=self._config['decay_steps'],
				decay_rate=self._config['decay_rate'],
				staircase=True)
		else:
			learning_rate_decay_fn = None

		self._train_op = tf.contrib.layers.optimize_loss(
			loss=self._loss,
			global_step=self._global_step,
			learning_rate=self._config['learning_rate'],
			optimizer=optimizer,
			gradient_noise_scale=None,
			gradient_multipliers=None,
			clip_gradients=None,
			learning_rate_decay_fn=learning_rate_decay_fn,
			update_ops=None,
			variables=None,
			name='Loss_optimization',
			summaries=['learning_rate', 'loss', 'gradients', 'gradient_norm'],
			colocate_gradients_with_ops=True,
			increment_global_step=True)