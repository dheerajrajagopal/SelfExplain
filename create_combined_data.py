
# -*- coding: utf-8 -*-
'''
Created on : Monday 12 Apr, 2021 : 17:43:57
Last Modified : Monday 12 Apr, 2021 : 17:57:08

@author       : Rishabh Joshi
Institute     : Carnegie Mellon University
'''
import random
from collections import defaultdict as ddict
path = 'data/original_combined_data/'
for file in ['train.tsv', 'dev.tsv']:
	f = open(path + file)
	lines = f.readlines()
	f.close()
	lines = [l.strip() for l in lines]
	lines = lines[1:] # skip header
	data = ddict(list)
	newdata = [] # list of tuples
	for l in lines:
		sent, label  = l.split('\t')
		data[label].append(sent)
	#import pdb; pdb.set_trace()
	for label in ['0', '1']:
		for i in range(len(data[label])):
			base_sent = data[label][i] # select base sentence
			num_sent = random.randint(1, 3) # 1 2 or 3
			for ns in range(num_sent):
				newsent = data[label][random.randint(0, len(data[label])-1)]
				if base_sent.strip().split()[-1] == '.':
					base_sent = base_sent + newsent
				else:
					base_sent = base_sent + ' . ' + newsent
			newdata.append((base_sent, label))
	f = open(path + file[:-4] + '_combined.tsv', 'w')
	random.shuffle(newdata)
	f.write('sentence\tlabel\n')
	for nd in newdata:
		f.write(nd[0]+'\t'+nd[1]+'\n')
	f.close()
