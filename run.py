import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2

from src.model import Config, Model
from src.data_layer import DataInRamInputLayer
from src.utils import deco_print, deco_print_dict

tf.flags.DEFINE_string('logdir', '', 'Path to save logs and checkpoints')
tf.flags.DEFINE_string('mode', 'valid', 'Mode: train/valid/test')
tf.flags.DEFINE_string('dataset', 'subprime', 'Dataset: subprime/prime/all')
tf.flags.DEFINE_integer('num_epochs', 50, 'Number of training epochs')
tf.flags.DEFINE_integer('summary_frequency', 100, 'iterations after which summary takes place')
FLAGS = tf.flags.FLAGS

### Create Data Layer
deco_print('Creating Data Layer...')
path = os.path.join(os.path.expanduser('~'), 'data/RNNdata')
if FLAGS.dataset == 'subprime':
	path = os.path.join(path, 'subprime_new')
elif FLAGS.dataset == 'prime':
	path = os.path.join(path, 'prime_new')
elif FLAGS.dataset == 'all':
	pass
else:
	raise ValueError('Dataset Not Found! ')

if FLAGS.mode == 'train' or FLAGS.mode == 'valid':
	dl = DataInRamInputLayer(path=path, shuffle=True)
elif FLAGS.mode == 'test':
	dl = DataInRamInputLayer(path=path, shuffle=False)
else:
	raise ValueError('Mode Not Implemented! ')
deco_print('Data Layer Created! ')
###

### Create Model
deco_print('Creating Model...')
if FLAGS.mode == 'train' or FLAGS.mode == 'valid':
	config = Config(feature_dim=291, num_category=7, cell_type='lstm', dropout=0.9)
	model = Model(config, is_training=True, use_valid_set=(FLAGS.mode=='valid'))
elif FLAGS.mode == 'test':
	config = Config(feature_dim=291, num_category=7, cell_type='lstm', dropout=1.0)
	model = Model(config, is_training=False, use_valid_set=False)
deco_print('Read Following Config')
deco_print_dict(vars(config))
deco_print('Model Created! ')
###

sess_config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=sess_config) as sess:
	saver = tf.train.Saver(max_to_keep=50)
	if tf.train.latest_checkpoint(FLAGS.logdir) is not None:
		saver.restore(sess, tf.train.latest_checkpoint(FLAGS.logdir))
		deco_print('Restored Checkpoint! ')
	else:
		sess.run(tf.global_variables_initializer())
		deco_print('Random Initialization! ')

	if FLAGS.mode == 'train' or FLAGS.mode == 'valid':
		deco_print('Executing Training Mode...\n')
		summary_op = tf.summary.merge_all()
		sw = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

		for epoch in range(FLAGS.num_epochs):
			epoch_start = time.time()

			### SGD step
			total_loss = 0.0
			count = 0
			for i, (X, Y, tDimSplit, p) in enumerate(dl.iterate_one_epoch(batch_size=config.global_batch_size)):
				feed_dict = {model._x_placeholder:X, model._y_placeholder:Y, model._tDimSplit_placeholder:tDimSplit}
				loss_i, _ = sess.run(fetches=[model._loss, model._train_op], feed_dict=feed_dict)
				total_loss += loss_i
				count += 1
				if count % FLAGS.summary_frequency == 0:
					sm, = sess.run(fetches=[summary_op], feed_dict=feed_dict)
					sw.add_summary(sm, global_step=sess.run(model._global_step))
					sw.flush()
					time_last = time.time() - epoch_start
					time_est = time_last / p
					deco_print('Training Loss Update: %f, Elapse / Estimate: %.2fs / %.2fs     ' %(total_loss / count, time_last, time_est), end='\r')

			### Epoch loss summary
			total_train_loss = 0.0
			count_train = 0
			if FLAGS.mode == 'valid':
				total_valid_loss = 0.0
				count_valid = 0

			for i, (X, Y, tDimSplit, _) in enumerate(dl.iterate_one_epoch(batch_size=config.global_batch_size)):
				feed_dict = {model._x_placeholder:X, model._y_placeholder:Y, model._tDimSplit_placeholder:tDimSplit}
				if FLAGS.mode == 'train':
					loss_i, num_i = sess.run(fetches=[model._sum_loss, model._num], feed_dict=feed_dict)
					total_train_loss += loss_i
					count_train += num_i
				else:
					loss_i, loss_i_valid, num_i, num_i_valid = sess.run(fetches=[model._sum_loss, model._sum_loss_valid, model._num, model._num_valid], feed_dict=feed_dict)
					total_train_loss += loss_i
					total_valid_loss += loss_i_valid
					count_train += num_i
					count_valid += num_i_valid

			train_loss = total_train_loss / count_train
			deco_print('Epoch {} Training Loss: {}                              '.format(epoch, train_loss))
			train_loss_value = summary_pb2.Summary.Value(tag='Train_Epoch_Loss', simple_value=train_loss)
			if FLAGS.mode == 'train':
				summary = summary_pb2.Summary(value=[train_loss_value])
			else:
				valid_loss = total_valid_loss / count_valid
				deco_print('Epoch {} Validation Loss: {}                              '.format(epoch, valid_loss))
				valid_loss_value = summary_pb2.Summary.Value(tag='Valid_Epoch_Loss', simple_value=valid_loss)
				summary = summary_pb2.Summary(value=[train_loss_value, valid_loss_value])

			sw.add_summary(summary=summary, global_step=epoch)
			sw.flush()
			deco_print('Saving Epoch Checkpoint')
			saver.save(sess, save_path=os.path.join(FLAGS.logdir, 'model-epoch'), global_step=epoch)
			epoch_end = time.time()
			deco_print('Did Epoch {} In {} Seconds \n'.format(epoch, epoch_end - epoch_start))

	else:
		pass

