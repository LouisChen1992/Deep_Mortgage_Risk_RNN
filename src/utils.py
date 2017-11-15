import os
import six

def deco_print(line, end='\n'):
	six.print_('>==================> ' + line, end=end)

def deco_print_dict(dic):
	for key, value in dic.items():
		deco_print('{} : {}'.format(key, value))

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