import os
import random
import six

def deco_print(line, end='\n'):
	six.print_('>==================> ' + line, end=end)

def deco_print_dict(dic):
	for key, value in dic.items():
		deco_print('{} : {}'.format(key, value))

def weighted_choice(bucket_count, buckets):
	total_count = sum(bucket_count.values())
	if total_count == 0:
		return None
	else:
		r = random.randint(1, total_count)
		idx = 0
		s = bucket_count[buckets[0]]
		while s < r:
			idx += 1
			s += bucket_count[buckets[idx]]
		return buckets[idx]

def create_file_dict(path):
	loanID_list = []
	X_int_list = []
	X_float_list = []
	outcome_list = []
	tDimSplit_list = []

	for file in os.listdir(path):
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

	buckets = list(set(map(lambda s: s.split('_')[2], loanID_list)))

	bucket_loanID = {bucket:[] for bucket in buckets}
	bucket_X_int = {bucket:[] for bucket in buckets}
	bucket_X_float = {bucket:[] for bucket in buckets}
	bucket_outcome = {bucket:[] for bucket in buckets}
	bucket_tDimSplit = {bucket:[] for bucket in buckets}

	for file in loanID_list:
		bucket_loanID[file.split('_')[2]].append(file)
	for file in X_int_list:
		bucket_X_int[file.split('_')[4]].append(file)
	for file in X_float_list:
		bucket_X_float[file.split('_')[4]].append(file)
	for file in outcome_list:
		bucket_outcome[file.split('_')[3]].append(file)
	for file in tDimSplit_list:
		bucket_tDimSplit[file.split('_')[2]].append(file)

	for bucket in buckets:
		bucket_loanID[bucket] = sorted(bucket_loanID[bucket])
		bucket_X_int[bucket] = sorted(bucket_X_int[bucket])
		bucket_X_float[bucket] = sorted(bucket_X_float[bucket])
		bucket_outcome[bucket] = sorted(bucket_outcome[bucket])
		bucket_tDimSplit[bucket] = sorted(bucket_tDimSplit[bucket])

	bucket_data = {'loanID':bucket_loanID, 'X_int':bucket_X_int, 'X_float':bucket_X_float, 'outcome':bucket_outcome, 'tDimSplit':bucket_tDimSplit}
	return (bucket_data, buckets)

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

def RNNdata_count(path):
	count = dict()
	for lValue in os.listdir(path):
		ts = int(lValue)
		count[ts] = 0
		for tValue in os.listdir(os.path.join(path, lValue)):
			path_i = os.path.join(path, lValue, tValue)
			for file in os.listdir(path_i):
				if file.startswith('numRows_np'):
					numRows = np.load(os.path.join(path_i, file))
					count[ts] += numRows[0] // ts
	return count