import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell, DropoutWrapper, MultiRNNCell
from .utils import deco_print

def create_rnn_cell(cell_type,
					num_units,
					num_layers=1,
					dp_input_keep_prob=1.0,
					dp_output_keep_prob=1.0):

	def single_cell(num_units):
		if cell_type == 'lstm':
			cell_class = LSTMCell
		elif cell_type == 'gru':
			cell_class = GRUCell
		else:
			raise ValueError('Cell Type Not Supported! ')

		if dp_input_keep_prob != 1.0 or dp_output_keep_prob != 1.0:
			return DropoutWrapper(cell_class(num_units=num_units),
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

class Config:
	def __init__(self,
				feature_dim,
				num_category,
				cell_type,
				num_units=[200, 140, 140, 140, 140],
				num_layers=5,
				num_gpus=2,
				batch_size_per_gpu=2000,
				optimizer='Momentum',
				learning_rate=0.1,
				decay_steps=None,
				decay_rate=None,
				dropout=1.0):
		self._feature_dim = feature_dim
		self._num_category = num_category
		self._cell_type = cell_type
		self._num_units = num_units
		self._num_layers = num_layers
		self._num_gpus = num_gpus
		self._batch_size_per_gpu = batch_size_per_gpu
		self._optimizer = optimizer
		self._learning_rate = learning_rate
		self._decay_steps = decay_steps
		self._decay_rate = decay_rate
		self._dropout = dropout

	@property
	def feature_dim(self):
		return self._feature_dim

	@property
	def num_category(self):
		return self._num_category

	@property
	def cell_type(self):
		return self._cell_type

	@property
	def num_units(self):
		return self._num_units

	@property
	def num_layers(self):
		return self._num_layers

	@property
	def num_gpus(self):
		return self._num_gpus

	@property
	def batch_size_per_gpu(self):
		return self._batch_size_per_gpu

	@property
	def optimizer(self):
		return self._optimizer

	@property
	def learning_rate(self):
		return self._learning_rate

	@property
	def decay_steps(self):
		return self._decay_steps

	@property
	def decay_rate(self):
		return self._decay_rate

	@property
	def dropout(self):
		return self._dropout

class Model:
	def __init__(self, config, global_step=None, force_var_reuse=False, is_training=True):
		self._config = config
		self._force_var_reuse = force_var_reuse
		self._is_training = is_training
		self._global_batch_size = self._config.num_gpus * self._config.batch_size_per_gpu
		self._global_step = global_step is global_step is not None else tf.contrib.framework.get_or_create_global_step()

		self._x_placeholder = tf.placeholder(dtype=tf.float32, shape=[self._global_batch_size, None, self._config.feature_dim], name='input_placeholder')
		self._x_length = tf.placeholder(dtype=tf.int32, shape=[self._global_batch_size])

		xs = tf.split(value=self._x_placeholder, num_or_size_splits=self._config.num_gpus, axis=0)
		x_lengths = tf.split(value=self._x_length, num_or_size_splits=self._config.num_gpus, axis=0)

		if self._is_training:
			self._y_placeholder = tf.placeholder(dtype=tf.int32, shape=[self._global_batch_size, None], name='output_placeholder')
			ys = tf.split(value=self._y_placeholder, num_or_size_splits=self._config.num_gpus, axis=0)
			self._loss = 0

		for gpu_idx in range(self._config.num_gpus):
			with tf.device('/gpu:{}'.format(gpu_idx)), tf.variable_scope(
				name_or_scope=tf.get_variable_scope(),
				initializer=tf.random_uniform_initializer(minval=-0.5, maxval=0.5),
				reuse=self._force_var_reuse or (gpu_idx > 0)):

				deco_print('Building graph on GPU:{}'.format(gpu_idx))

				logits_i = self._build_forward_pass_graph(x=xs[gpu_idx], x_length=x_lengths[gpu_idx])
				if self._is_training:
					loss_i = self._add_loss(logits=logits_i, labels=ys[gpu_idx], lengths=x_lengths[gpu_idx])
					weight_i = tf.reduce_sum(x_lengths[gpu_idx]) / tf.reduce_sum(self._x_length)
					self._loss += loss_i * weight_i

		if self._is_training:
			self._add_train_op()

	def _build_forward_pass_graph(self, x, x_length):
		with tf.variable_scope('{}_layer'.format(self._config.cell_type)):
			rnn_cell = create_rnn_cell(
				cell_type=self._config.cell_type,
				num_units=self._config.num_units,
				num_layers=self._config.num_layers,
				dp_input_keep_prob=self._config.dropout,
				dp_output_keep_prob=1.0)
			outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=x, sequence_length=x_length, dtype=tf.float32)

		with tf.variable_scope('output_layer'):
			layer = Dense(units=self._config.num_category)
			logits = layer(outputs)

		return logits

	def _add_loss(self, logits, labels, lengths):
		ts = tf.reduce_max(lengths)
		logits = tf.slice(logits, begin=[0, 0, 0], size=[-1, ts, -1])
		labels = tf.slice(labels, begin=[0, 0], size=[-1, ts])
		mask = tf.sequence_mask(lengths=lengths, maxlen=ts, dtype=tf.float32)
		loss = tf.contrib.seq2seq.sequence_loss(
			logits=logits,
			targets=labels,
			weights=mask,
			average_across_timesteps=True,
			average_across_batch=True,
			softmax_loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits)
		return loss
	
	def _add_train_op(self):
		deco_print('Trainable Variables')
		for var in tf.trainable_variables():
			deco_print('Name: {} and shape: {}'.format(var.name, var.get_shape()))

		if self._config.optimizer = 'Momentum':
			optimizer = lambda lr: tf.train.MomentumOptimizer(lr, momentum=0.9)
		elif self._config.optimizer = 'AdaDelta':
			optimizer = lambda lr: tf.train.AdadeltaOptimizer(lr, rho=0.95, epsilon=1e-08)
		else:
			optimizer = self._config.optimizer

		if self._config.decay_steps is not None and self._config.decay_rate is not None:
			learning_rate_decay_fn = lambda lr, global_step: tf.train.exponential_decay(
				learning_rate=lr,
				global_step=global_step,
				decay_steps=self._config.decay_steps,
				decay_rate=self._config.decay_rate,
				staircase=True)
		else:
			learning_rate_decay_fn = None

		self._train_op = tf.contrib.layers.optimize_loss(
			loss=self._loss,
			global_step=self._global_step,
			learning_rate=self._config.learning_rate,
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