import os
import copy
import json
import numpy as np
from .utils import weighted_choice, create_file_dict

class DataInRamInputLayer():
	def __init__(self, path, shuffle=True, load_file_list=True, selected_int=False, selected_float=False):
		self._path = path
		self._shuffle = shuffle
		self._selected_int = selected_int
		self._selected_float = selected_float
		self._create_covariate_idx_associations()
		if load_file_list:
			self._create_file_list()

	def _create_covariate_idx_associations(self):
		with open('covariate/covariate2idx_int.json', 'r') as f:
			self._covariate2idx_int = json.load(f)
			self._idx2covariate_int = {value:key for key, value in self._covariate2idx_int.items()}
			self._covariate_count_int = len(self._covariate2idx_int.keys())

		with open('covariate/covariate2idx_float.json', 'r') as f:
			self._covariate2idx_float = json.load(f)
			self._idx2covariate_float = {value:key for key, value in self._covariate2idx_float.items()}
			self._covariate_count_float = len(self._covariate2idx_float.keys())

		self._idx2covariate = {}
		for key in self._idx2covariate_int.keys():
			self._idx2covariate[key] = self._idx2covariate_int[key]
		for key in self._idx2covariate_float.keys():
			self._idx2covariate[key+self._covariate_count_int] = self._idx2covariate_float[key]
		self._covariate_count = self._covariate_count_int + self._covariate_count_float
		self._covariate2idx = {value:key for key, value in self._idx2covariate.items()}

		with open('covariate/outcome2idx.json', 'r') as f:
			self._outcome2idx = json.load(f)
			self._idx2outcome = {value:key for key, value in self._outcome2idx.items()}
			self._outcome_count = len(self._outcome2idx.keys())

	def _create_file_list(self):
		# loanID_list = []
		# X_int_list = []
		# X_float_list = []
		# outcome_list = []
		# tDimSplit_list = []

		# for file in os.listdir(self._path):
		# 	if file.startswith('loanID_np'):
		# 		loanID_list.append(file)
		# 	elif file.startswith('X_data_np_int'):
		# 		X_int_list.append(file)
		# 	elif file.startswith('X_data_np_float'):
		# 		X_float_list.append(file)
		# 	elif file.startswith('outcome_data_np'):
		# 		outcome_list.append(file)
		# 	elif file.startswith('tDimSplit_np'):
		# 		tDimSplit_list.append(file)

		# self._buckets = list(set(map(lambda s: s.split('_')[2], loanID_list)))

		# self._bucket_loanID = {bucket:[] for bucket in self._buckets}
		# self._bucket_X_int = {bucket:[] for bucket in self._buckets}
		# self._bucket_X_float = {bucket:[] for bucket in self._buckets}
		# self._bucket_outcome = {bucket:[] for bucket in self._buckets}
		# self._bucket_tDimSplit = {bucket:[] for bucket in self._buckets}

		# for file in loanID_list:
		# 	self._bucket_loanID[file.split('_')[2]].append(file)
		# for file in X_int_list:
		# 	self._bucket_X_int[file.split('_')[4]].append(file)
		# for file in X_float_list:
		# 	self._bucket_X_float[file.split('_')[4]].append(file)
		# for file in outcome_list:
		# 	self._bucket_outcome[file.split('_')[3]].append(file)
		# for file in tDimSplit_list:
		# 	self._bucket_tDimSplit[file.split('_')[2]].append(file)

		# for bucket in self._buckets:
		# 	self._bucket_loanID[bucket] = sorted(self._bucket_loanID[bucket])
		# 	self._bucket_X_int[bucket] = sorted(self._bucket_X_int[bucket])
		# 	self._bucket_X_float[bucket] = sorted(self._bucket_X_float[bucket])
		# 	self._bucket_outcome[bucket] = sorted(self._bucket_outcome[bucket])
		# 	self._bucket_tDimSplit[bucket] = sorted(self._bucket_tDimSplit[bucket])

		bucket_data, self._buckets = create_file_dict(self._path)
		self._bucket_loanID = bucket_data['loanID']
		self._bucket_X_int = bucket_data['X_int']
		self._bucket_X_float = bucket_data['X_float']
		self._bucket_outcome = bucket_data['outcome']
		self._bucket_tDimSplit = bucket_data['tDimSplit']

		self._bucket_count = {bucket:len(self._bucket_loanID[bucket]) for bucket in self._buckets}
		self._bucket_outseq = {bucket:np.arange(self._bucket_count[bucket]) for bucket in self._buckets}

	def iterate_one_epoch(self, batch_size):
		if self._shuffle:
			for bucket in self._buckets:
				np.random.shuffle(self._bucket_outseq[bucket])

		bucket_idx = {bucket:0 for bucket in self._buckets}
		bucket_count_left = copy.deepcopy(self._bucket_count)

		total_count = sum(self._bucket_count.values())
		current_count = 0.0

		bucket = weighted_choice(bucket_count_left, self._buckets)
		while bucket is not None:
			current_count += 1
			idx_file = self._bucket_outseq[bucket][bucket_idx[bucket]]
			X_int = np.load(os.path.join(self._path, self._bucket_X_int[bucket][idx_file]))[:,:,:-2]  # remove last two integer feature
			X_float = np.load(os.path.join(self._path, self._bucket_X_float[bucket][idx_file]))
			outcome = np.load(os.path.join(self._path, self._bucket_outcome[bucket][idx_file]))
			tDimSplit = np.load(os.path.join(self._path, self._bucket_tDimSplit[bucket][idx_file]))

			if self._selected_int and self._selected_float:
				selected_int_FF = [i for i in range(self._covariate_count_int) if i not in self._selected_int]
				selected_float_FF = [i for i in range(self._covariate_count_float) if i not in self._selected_float]
				X_int_RNN = X_int[:,:,self._selected_int]
				X_float_RNN = X_float[:,:,self._selected_float]
				X_int_FF = X_int[:,:,selected_int_FF]
				X_float_FF = X_float[:,:,selected_float_FF]
			else:
				X_int_RNN = X_int
				X_float_RNN = X_float
				X_int_FF = X_int[:,:,[]]
				X_float_FF = X_float[:,:,[]]

			num_example = X_int_RNN.shape[0]
			num_batch = num_example // batch_size
			idx_example = np.arange(num_example)
			if self._shuffle:
				np.random.shuffle(idx_example)

			for idx_batch in range(num_batch):
				batch_start = idx_batch*batch_size
				batch_end = (idx_batch+1)*batch_size

				idx_input = idx_example[batch_start:batch_end]
				X_int_RNN_input = X_int_RNN[idx_input]
				X_float_RNN_input = X_float_RNN[idx_input]
				X_int_FF_input = X_int_FF[idx_input]
				X_float_FF_input = X_float_FF[idx_input]
				X_RNN_input = np.concatenate((X_int_RNN_input, X_float_RNN_input), axis=2)
				X_FF_input = np.concatenate((X_int_FF_input, X_float_FF_input), axis=2)
				Y_input = outcome[idx_input]
				tDimSplit_input = tDimSplit[idx_input]
				yield X_RNN_input, X_FF_input, Y_input, tDimSplit_input, current_count / total_count

			bucket_count_left[bucket] -= 1
			bucket_idx[bucket] += 1
			bucket = weighted_choice(bucket_count_left, self._buckets)
