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
	path_subprime = os.path.join(path, 'data/RNNdata')
	path_src = os.path.join(path_subprime, 'subprime')
	path_tgt = os.path.join(path_subprime, 'subprime_new')

	buckets = [5,10,15,20,25,30,35,40,45,50,55,60,70,80,90,100,110,120,150,200]
	num_bucket = {5:250, 10:200, 15:150, 20:100, 25:100, 30:80, 35:80, 40:50, 45:50, 50:40, 55:30, 60:25, 70:40, 80:40, 90:35, 100:35, 110:20, 120:10, 150:10, 200:5}
	bucket_count = {5:2438272, 10:2132845, 15:1632765, 20:1007436, 25:1170984, 30:809388, 35:835433, 40:515515, 45:477298, 50:401126, 55:347931, 60:257416, 70:414083, 80:387684, 90:365817, 100:349785, 110:236957, 120:124985, 150:132095, 200:61138}

	count_subprime = 0
	time_start = time.time()

	for lValue in range(1, 187):
		bucket = decide_bucket(lValue, buckets)
		if bucket != int(args.bucket):
			continue

		loanID_lValue = np.empty((0,), dtype='S8')
		X_int_lValue = np.empty((0, lValue, 239), dtype='int8')
		X_float_lValue = np.empty((0, lValue, 54), dtype='float32')
		outcome_lValue = np.empty((0, lValue), dtype='int64')
		tDimSplit_lValue = np.empty((0, 3), dtype='int16')

		print('Processing data with lValue %d...' %lValue)

		for tValue in os.listdir(os.path.join(path_src,str(lValue))):
			path_i = os.path.join(path_src, str(lValue), tValue)
			len_sep = longitudinal_separation(lValue=lValue, tValue=int(tValue))

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
				loanID = loanID.reshape((-1, lValue))[:,0]
				loanID_lValue = np.append(loanID_lValue, loanID, axis=0)

				X_float = np.load(os.path.join(path_i, X_float_list[idx_src]))
				X_float = X_float.reshape((-1, lValue, 54))
				X_float_lValue = np.append(X_float_lValue, X_float, axis=0)

				X_int = np.load(os.path.join(path_i, X_int_list[idx_src]))
				X_int = X_int.reshape((-1, lValue, 239))
				X_int_lValue = np.append(X_int_lValue, X_int, axis=0)

				outcome = np.load(os.path.join(path_i, outcome_list[idx_src]))
				outcome = outcome.reshape((-1, lValue))
				outcome_lValue = np.append(outcome_lValue, outcome, axis=0)

				tDimSplit_lValue = np.append(tDimSplit_lValue, np.tile(len_sep, (len(loanID),1)), axis=0)

			print('Finished loading %s! ' %path_i, end='\r')

		print('Finished loading data with lValue %d!                              ' %lValue)
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
			X_int_new = np.append(X_int_new, np.lib.pad(X_int_lValue[idx_tgt==idx], ((0,0),(0, bucket-lValue),(0,0)), 'constant'), axis=0)
			np.save(path_X_int, X_int_new)
			X_int_new = None

			path_X_float = os.path.join(path_tgt, 'X_data_np_float_%d_%d.npy' %(bucket, idx))
			if os.path.exists(path_X_float):
				X_float_new = np.load(path_X_float)
			else:
				X_float_new = np.empty((0, bucket, 54), dtype='float32')
			X_float_new = np.append(X_float_new, np.lib.pad(X_float_lValue[idx_tgt==idx], ((0,0),(0, bucket-lValue),(0,0)), 'constant'), axis=0)
			np.save(path_X_float, X_float_new)
			X_float_new = None

			path_outcome = os.path.join(path_tgt, 'outcome_data_np_%d_%d.npy' %(bucket, idx))
			if os.path.exists(path_outcome):
				outcome_new = np.load(path_outcome)
			else:
				outcome_new = np.empty((0, bucket), dtype='int64')
			outcome_new = np.append(outcome_new, np.lib.pad(outcome_lValue[idx_tgt==idx], ((0,0),(0, bucket-lValue)), 'constant'), axis=0)
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
		print('Finished data with lValue %d! \t %d / %d \t Elapse/Estimate: %0.2fs / %0.2fs' %(lValue, count_subprime, bucket_count[bucket], time_elapse, time_estimate))

elif args.dataset == 'prime':
	path_prime = os.path.join(path, 'data/RNNdata')
	path_src = os.path.join(path_prime, 'prime')
	path_tgt = os.path.join(path_prime, 'prime_new')

	buckets = [5,10,15,20,25,30,35,40,45,50,55,60,70,80,90,100,110,120,150,200]
	num_bucket = {5:500, 10:500, 15:650, 20:650, 25:650, 30:500, 35:450, 40:450, 45:500, 50:400, 55: 400, 60:300, 70:450, 80:350, 90:300, 100:250, 110:200, 120:150, 150:250, 200:50}
	bucket_count = {5:5083424, 10:4967764, 15:6735858, 20:6535129, 25:6513694, 30:5237239, 35:4703641, 40:4462590, 45:5196881, 50:4203606, 55:3842256, 60:3052786, 70:4608179, 80:3387314, 90:3188225, 100:2412927, 110:2264771, 120:1786723, 150:2789394, 200:645933}

	count_prime = 0
	time_start = time.time()

	for lValue in range(1, 186):
		bucket = decide_bucket(lValue, buckets)
		if bucket != int(args.bucket):
			continue

		loanID_lValue = np.empty((0,), dtype='S8')
		X_int_lValue = np.empty((0, lValue, 239), dtype='int8')
		X_float_lValue = np.empty((0, lValue, 54), dtype='float32')
		outcome_lValue = np.empty((0, lValue), dtype='int64')
		tDimSplit_lValue = np.empty((0, 3), dtype='int16')

		print('Processing data with lValue %d...' %lValue)

		for folder in os.listdir(path_src):
			for tValue in os.listdir(os.path.join(path_src, folder, str(lValue))):
				path_i = os.path.join(path_src, folder, str(lValue), tValue)
				len_sep = longitudinal_separation(lValue=lValue, tValue=int(tValue))

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
					loanID = loanID.reshape((-1, lValue))[:,0]
					loanID_lValue = np.append(loanID_lValue, loanID, axis=0)

					X_float = np.load(os.path.join(path_i, X_float_list[idx_src]))
					X_float = X_float.reshape((-1, lValue, 54))
					X_float_lValue = np.append(X_float_lValue, X_float, axis=0)

					X_int = np.load(os.path.join(path_i, X_int_list[idx_src]))
					X_int = X_int.reshape((-1, lValue, 239))
					X_int_lValue = np.append(X_int_lValue, X_int, axis=0)

					outcome = np.load(os.path.join(path_i, outcome_list[idx_src]))
					outcome = outcome.reshape((-1, lValue))
					outcome_lValue = np.append(outcome_lValue, outcome, axis=0)

					tDimSplit_lValue = np.append(tDimSplit_lValue, np.tile(len_sep, (len(loanID),1)), axis=0)

				print('Finished loading %s! ' %path_i, end='\r')

		print('Finished loading data with lValue %d!                               ' %lValue)
		count_prime += len(loanID_lValue)
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
			X_int_new = np.append(X_int_new, np.lib.pad(X_int_lValue[idx_tgt==idx], ((0,0),(0, bucket-lValue),(0,0)), 'constant'), axis=0)
			np.save(path_X_int, X_int_new)
			X_int_new = None

			path_X_float = os.path.join(path_tgt, 'X_data_np_float_%d_%d.npy' %(bucket, idx))
			if os.path.exists(path_X_float):
				X_float_new = np.load(path_X_float)
			else:
				X_float_new = np.empty((0, bucket, 54), dtype='float32')
			X_float_new = np.append(X_float_new, np.lib.pad(X_float_lValue[idx_tgt==idx], ((0,0),(0, bucket-lValue),(0,0)), 'constant'), axis=0)
			np.save(path_X_float, X_float_new)
			X_float_new = None

			path_outcome = os.path.join(path_tgt, 'outcome_data_np_%d_%d.npy' %(bucket, idx))
			if os.path.exists(path_outcome):
				outcome_new = np.load(path_outcome)
			else:
				outcome_new = np.empty((0, bucket), dtype='int64')
			outcome_new = np.append(outcome_new, np.lib.pad(outcome_lValue[idx_tgt==idx], ((0,0),(0, bucket-lValue)), 'constant'), axis=0)
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
		time_estimate = time_elapse / count_prime * bucket_count[bucket]
		print('Finished data with lValue %d! \t %d / %d \t Elapse/Estimate: %0.2fs / %0.2fs' %(lValue, count_prime, bucket_count[bucket], time_elapse, time_estimate))

elif args.dataset == 'all':
	raise ValueError('Not Implemented!')
else:
	raise ValueError('Dataset Not Found! ')