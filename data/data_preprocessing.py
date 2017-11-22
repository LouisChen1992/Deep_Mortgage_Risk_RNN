import argparse
import os
import time
import numpy as np

parser = argparse.ArgumentParser(description='Generate random buckets. ')
parser.add_argument('--dataset', help='prime/subprime/all')
parser.add_argument('--bucket', help='50/75/100/150/200')
args = parser.parse_args()

def longitudinal_separation(lValue, tValue, sep=[121,281,287,306]):
	###retun the length in train/valid/test
	len_train = min(max(sep[1]-tValue,0),lValue)
	len_valid = min(max(sep[2]-tValue,0),max(tValue+lValue-sep[1],0),lValue,sep[2]-sep[1])
	len_test = min(max(tValue+lValue-sep[2],0),lValue)
	return np.array([len_train,len_valid,len_test], dtype='int16')

def decide_bucket(lValue, buckets):
	for bucket in buckets:
		if lValue <= bucket:
			return bucket

path = os.path.expanduser('~')

if args.dataset == 'subprime':
	# path_subprime = os.path.join(path, 'data/RNNdata')
	# path_src = os.path.join(path_subprime, 'subprime')
	# path_tgt = os.path.join(path_subprime, 'subprime_new')
	# ###use bucket size [50, 75, 100, 150, 200]
	# ###num of bucket [2800, 290, 210, 120, 15]
	# ###~4000 loan in each bucket
	# num_bucket = {50:2800, 75:290, 100:210, 150:120, 200:15}
	# bucket_count = {50:11421062, 75:1222169, 100:900547, 150:494037, 200:61138}

	# count_subprime = 0
	# time_start = time.time()

	# for lValue in os.listdir(path_src):
	# 	bucket = decide_bucket(int(lValue), sorted(list(num_bucket.keys())))
	# 	if bucket != int(args.bucket):
	# 		continue
	# 	for tValue in os.listdir(os.path.join(path_src,lValue)):
	# 		len_sep = longitudinal_separation(lValue=int(lValue), tValue=int(tValue))

	# 		path_i = os.path.join(path_src, lValue, tValue)
	# 		X_int_list = []
	# 		X_float_list = []
	# 		outcome_list = []
	# 		loanID_list = []
	# 		for file in os.listdir(path_i):
	# 			if file.startswith('X_data_np_int'):
	# 				X_int_list.append(file)
	# 			elif file.startswith('X_data_np_float'):
	# 				X_float_list.append(file)
	# 			elif file.startswith('outcome'):
	# 				outcome_list.append(file)
	# 			elif file.startswith('loanID'):
	# 				loanID_list.append(file)
	# 		X_int_list = sorted(X_int_list)
	# 		X_float_list = sorted(X_float_list)
	# 		outcome_list = sorted(outcome_list)
	# 		loanID_list = sorted(loanID_list)
	# 		num_file = len(loanID_list)

	# 		for idx_src in range(num_file):
	# 			loanID = np.load(os.path.join(path_i, loanID_list[idx_src]))
	# 			loanID = loanID.reshape((-1, int(lValue)))[:,0]

	# 			X_float = np.load(os.path.join(path_i, X_float_list[idx_src]))
	# 			X_float = X_float.reshape((-1, int(lValue), 54))

	# 			X_int = np.load(os.path.join(path_i, X_int_list[idx_src]))
	# 			X_int = X_int.reshape((-1, int(lValue), 239))

	# 			outcome = np.load(os.path.join(path_i, outcome_list[idx_src]))
	# 			outcome = outcome.reshape((-1, int(lValue)))

	# 			idx_tgt = np.random.choice(num_bucket[bucket], len(loanID))
	# 			count_subprime += len(loanID)

	# 			for i in range(len(loanID)):
	# 				idx = idx_tgt[i]

	# 				path_loanID = os.path.join(path_tgt, 'loanID_np_%d_%d.npy' %(bucket, idx))
	# 				if os.path.exists(path_loanID):
	# 					loanID_new = np.load(path_loanID)
	# 					loanID_new = np.append(loanID_new, [loanID[i]], axis=0)
	# 				else:
	# 					loanID_new = np.array([loanID[i]], dtype=loanID.dtype)
	# 				np.save(path_loanID, loanID_new)
	# 				loanID_new = None

	# 				path_X_int = os.path.join(path_tgt, 'X_data_np_int_%d_%d.npy' %(bucket, idx))
	# 				if os.path.exists(path_X_int):
	# 					X_int_new = np.load(path_X_int)
	# 					X_int_new = np.append(X_int_new, [np.lib.pad(X_int[i], ((0, bucket-int(lValue)),(0,0)), 'constant')], axis=0)
	# 				else:
	# 					X_int_new = np.array([np.lib.pad(X_int[i], ((0, bucket-int(lValue)),(0,0)), 'constant')], dtype=X_int.dtype)
	# 				np.save(path_X_int, X_int_new)
	# 				X_int_new = None

	# 				path_X_float = os.path.join(path_tgt, 'X_data_np_float_%d_%d.npy' %(bucket, idx))
	# 				if os.path.exists(path_X_float):
	# 					X_float_new = np.load(path_X_float)
	# 					X_float_new = np.append(X_float_new, [np.lib.pad(X_float[i], ((0, bucket-int(lValue)),(0,0)), 'constant')], axis=0)
	# 				else:
	# 					X_float_new = np.array([np.lib.pad(X_float[i], ((0, bucket-int(lValue)),(0,0)), 'constant')], dtype=X_float.dtype)
	# 				np.save(path_X_float, X_float_new)
	# 				X_float_new = None

	# 				path_outcome = os.path.join(path_tgt, 'outcome_data_np_%d_%d.npy' %(bucket, idx))
	# 				if os.path.exists(path_outcome):
	# 					outcome_new = np.load(path_outcome)
	# 					outcome_new = np.append(outcome_new, [np.lib.pad(outcome[i], (0, bucket-int(lValue)), 'constant')], axis=0)
	# 				else:
	# 					outcome_new = np.array([np.lib.pad(outcome[i], (0, bucket-int(lValue)), 'constant')], dtype=outcome.dtype)
	# 				np.save(path_outcome, outcome_new)
	# 				outcome_new = None

	# 				path_tDimSplit = os.path.join(path_tgt, 'tDimSplit_np_%d_%d.npy' %(bucket, idx))
	# 				if os.path.exists(path_tDimSplit):
	# 					tDimSplit_new = np.load(path_tDimSplit)
	# 					tDimSplit_new = np.append(tDimSplit_new, [len_sep], axis=0)
	# 				else:
	# 					tDimSplit_new = np.array([len_sep], dtype=len_sep.dtype)
	# 				np.save(path_tDimSplit, tDimSplit_new)
	# 				tDimSplit_new = None
	# 		time_elapse = time.time() - time_start
	# 		time_estimate = time_elapse / count_subprime * bucket_count[bucket]
	# 		print('%s Completed! \t %d / %d \t Elapse/Estimate: %0.2fs / %0.2fs' %(path_i, count_subprime, bucket_count[bucket], time_elapse, time_estimate))
	raise ValueError('Not Implemented!')
elif args.dataset == 'prime':
	path_subprime = os.path.join(path, 'data/RNNdata')
	path_src = os.path.join(path_subprime, 'subprime')
	path_tgt = os.path.join(path_subprime, 'subprime_new_50')
	###use bucket size [50, 75, 100, 150, 200]
	###num of bucket [2800, 290, 210, 120, 15]
	###~4000 loan in each bucket
	num_bucket = {50:2800, 75:290, 100:210, 150:120, 200:15}
	bucket_count = {50:11421062, 75:1222169, 100:900547, 150:494037, 200:61138}

	count_subprime = 0
	time_start = time.time()

	for lValue in os.listdir(path_src):
		bucket = decide_bucket(int(lValue), sorted(list(num_bucket.keys())))
		if bucket != int(args.bucket):
			continue

		###
		loanID_lValue = np.empty((0,), dtype='S8')
		X_int_lValue = np.empty((0, int(lValue), 239), dtype='int8')
		X_float_lValue = np.empty((0, int(lValue), 54), dtype='float32')
		outcome_lValue = np.empty((0, int(lValue)), dtype='int64')
		tDimSplit_lValue = np.empty((0, 3), dtype='int16')
		###

		print('Processing data in folder %s' %lValue)

		for tValue in os.listdir(os.path.join(path_src,lValue)):
			len_sep = longitudinal_separation(lValue=int(lValue), tValue=int(tValue))

			path_i = os.path.join(path_src, lValue, tValue)
			X_int_list = []
			X_float_list = []
			outcome_list = []
			loanID_list = []
			for file in os.listdir(path_i):
				if file.startswith('X_data_np_int'):
					X_int_list.append(file)
				elif file.startswith('X_data_np_float'):
					X_float_list.append(file)
				elif file.startswith('outcome'):
					outcome_list.append(file)
				elif file.startswith('loanID'):
					loanID_list.append(file)
			X_int_list = sorted(X_int_list)
			X_float_list = sorted(X_float_list)
			outcome_list = sorted(outcome_list)
			loanID_list = sorted(loanID_list)
			num_file = len(loanID_list)

			for idx_src in range(num_file):
				loanID = np.load(os.path.join(path_i, loanID_list[idx_src]))
				loanID = loanID.reshape((-1, int(lValue)))[:,0]
				###
				loanID_lValue = np.append(loanID_lValue, loanID, axis=0)
				###

				X_float = np.load(os.path.join(path_i, X_float_list[idx_src]))
				X_float = X_float.reshape((-1, int(lValue), 54))
				###
				X_float_lValue = np.append(X_float_lValue, X_float, axis=0)
				###

				X_int = np.load(os.path.join(path_i, X_int_list[idx_src]))
				X_int = X_int.reshape((-1, int(lValue), 239))
				###
				X_int_lValue = np.append(X_int_lValue, X_int, axis=0)
				###

				outcome = np.load(os.path.join(path_i, outcome_list[idx_src]))
				outcome = outcome.reshape((-1, int(lValue)))
				###
				outcome_lValue = np.append(outcome_lValue, outcome, axis=0)
				###

				tDimSplit_lValue = np.append(tDimSplit_lValue, np.tile(len_sep, (len(loanID),1)), axis=0)
			print('Finished loading %s! ' %path_i, end='\r')

		print('Finished loading data! ')
		count_subprime += len(loanID_lValue)
		idx_tgt = np.random.choice(num_bucket[bucket], len(loanID_lValue))

		for idx in range(num_bucket[bucket]):
			print('Writing to bucket %d...' %idx, end='\r')

			path_loanID = os.path.join(path_tgt, 'loanID_np_%d_%d.npy' %(bucket, idx))
			if os.path.exists(path_loanID):
				loanID_new = np.load(path_loanID)
			else:
				loanID_new = np.empty((0,), dtype='S8')
			loanID_new = np.append(loanID_new, loanID_lValue[idx_tgt==idx], axis=0)
			np.save(path_loanID, loanID_new)
			loanID_new = None

			path_X_int = os.path.join(path_tgt, 'X_data_np_int_%d_%d.npy' %(bucket, idx))
			if os.path.exists(path_X_int):
				X_int_new = np.load(path_X_int)
			else:
				X_int_new = np.empty((0, bucket, 239), dtype='int8')
			X_int_new = np.append(X_int_new, np.lib.pad(X_int_lValue[idx_tgt==idx], ((0,0),(0, bucket-int(lValue)),(0,0)), 'constant'), axis=0)
			np.save(path_X_int, X_int_new)
			X_int_new = None

			path_X_float = os.path.join(path_tgt, 'X_data_np_float_%d_%d.npy' %(bucket, idx))
			if os.path.exists(path_X_float):
				X_float_new = np.load(path_X_float)
			else:
				X_float_new = np.empty((0, bucket, 54), dtype='float32')
			X_float_new = np.append(X_float_new, np.lib.pad(X_float_lValue[idx_tgt==idx], ((0,0),(0, bucket-int(lValue)),(0,0)), 'constant'), axis=0)
			np.save(path_X_float, X_float_new)
			X_float_new = None

			path_outcome = os.path.join(path_tgt, 'outcome_data_np_%d_%d.npy' %(bucket, idx))
			if os.path.exists(path_outcome):
				outcome_new = np.load(path_outcome)
			else:
				outcome_new = np.empty((0, bucket), dtype='int64')
			outcome_new = np.append(outcome_new, np.lib.pad(outcome_lValue[idx_tgt==idx], ((0,0),(0, bucket-int(lValue))), 'constant'), axis=0)
			np.save(path_outcome, outcome_new)
			outcome_new = None

			path_tDimSplit = os.path.join(path_tgt, 'tDimSplit_np_%d_%d.npy' %(bucket, idx))
			if os.path.exists(path_tDimSplit):
				tDimSplit_new = np.load(path_tDimSplit)
			else:
				tDimSplit_new = np.empty((0,3), dtype='int16')
			tDimSplit_new = np.append(tDimSplit_new, tDimSplit_lValue[idx_tgt==idx], axis=0)
			np.save(path_tDimSplit, tDimSplit_new)
			tDimSplit_new = None

		time_elapse = time.time() - time_start
		time_estimate = time_elapse / count_subprime * bucket_count[bucket]
		print('%s Completed! \t %d / %d \t Elapse/Estimate: %0.2fs / %0.2fs' %(lValue, count_subprime, bucket_count[bucket], time_elapse, time_estimate))

elif args.dataset == 'all':
	raise ValueError('Not Implemented!')
else:
	raise ValueError('Dataset Not Found! ')