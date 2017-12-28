import os
import copy
import json
import numpy as np
from .utils import weighted_choice

class DataInRamInputLayer():
	def __init__(self, path, shuffle=False, load_file_list=True):
		self._path = path
		self._shuffle = shuffle
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
		loanID_list = []
		X_int_list = []
		X_float_list = []
		outcome_list = []
		tDimSplit_list = []

		for file in os.listdir(self._path):
			if file.startswith('loanID_np'):
				loanID_list.append(file)
			elif file.startswith('X_data_np_int'):
				X_int_list.append(file)
			elif file.startswith('X_data_np_float'):
				X_float_list.append(file)
			elif file.startswith('outcome_data_np'):
				outcome_list.append(file)
			elif file.startswith('tDimSplit_np'):
				tDimSplit_list.append(file)

		self._buckets = list(set(map(lambda s: s.split('_')[2], loanID_list)))

		self._bucket_loanID = {bucket:[] for bucket in self._buckets}
		self._bucket_X_int = {bucket:[] for bucket in self._buckets}
		self._bucket_X_float = {bucket:[] for bucket in self._buckets}
		self._bucket_outcome = {bucket:[] for bucket in self._buckets}
		self._bucket_tDimSplit = {bucket:[] for bucket in self._buckets}

		for file in loanID_list:
			self._bucket_loanID[file.split('_')[2]].append(file)
		for file in X_int_list:
			self._bucket_X_int[file.split('_')[4]].append(file)
		for file in X_float_list:
			self._bucket_X_float[file.split('_')[4]].append(file)
		for file in outcome_list:
			self._bucket_outcome[file.split('_')[3]].append(file)
		for file in tDimSplit_list:
			self._bucket_tDimSplit[file.split('_')[2]].append(file)

		for bucket in self._buckets:
			self._bucket_loanID[bucket] = sorted(self._bucket_loanID[bucket])
			self._bucket_X_int[bucket] = sorted(self._bucket_X_int[bucket])
			self._bucket_X_float[bucket] = sorted(self._bucket_X_float[bucket])
			self._bucket_outcome[bucket] = sorted(self._bucket_outcome[bucket])
			self._bucket_tDimSplit[bucket] = sorted(self._bucket_tDimSplit[bucket])

		self._bucket_count = {bucket:len(self._bucket_loanID[bucket]) for bucket in self._buckets}
		self._bucket_outseq = {bucket:np.arange(self._bucket_count[bucket]) for bucket in self._buckets}

	def iterate_one_epoch(self, batch_size):
		if self._shuffle:
			for bucket in self._buckets:
				np.random.shuffle(self._bucket_outseq[bucket])

		bucket_idx = {bucket:0 for bucket in self._buckets}
		bucket_count_left = copy.deepcopy(self._bucket_count)

		bucket = weighted_choice(bucket_count_left, self._buckets)
		while bucket is not None:
			idx_file = self._bucket_outseq[bucket][bucket_idx[bucket]]
			X_int = np.load(os.path.join(self._path, self._bucket_X_int[bucket][idx_file]))
			X_float = np.load(os.path.join(self._path, self._bucket_X_float[bucket][idx_file]))
			outcome = np.load(os.path.join(self._path, self._bucket_outcome[bucket][idx_file]))
			tDimSplit = np.load(os.path.join(self._path, self._bucket_tDimSplit[bucket][idx_file]))

			num_example = X_int.shape[0]
			num_batch = num_example // batch_size
			idx_example = np.arange(num_example)
			if self._shuffle:
				np.random.shuffle(idx_example)

			for idx_batch in range(num_batch):
				idx_input = idx_example[idx_batch*batch_size:(idx_batch+1)*batch_size]
				X_int_input = X_int[idx_input]
				X_float_input = X_float[idx_input]
				X_input = np.concatenate((X_int_input[:,:,:-2], X_float_input), axis=2) # remove last two integer feature
				Y_input = outcome[idx_input]
				tDimSplit_input = tDimSplit[idx_input]
				yield X_input, Y_input, tDimSplit_input

			bucket_count_left[bucket] -= 1
			bucket_idx[bucket] += 1
			bucket = weighted_choice(bucket_count_left, self._buckets)
