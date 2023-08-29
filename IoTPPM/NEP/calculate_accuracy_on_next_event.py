'''
this script takes as input the output of evaluate_suffix_and_remaining_time.py
therefore, the latter needs to be executed first

Author: Niek Tax
'''

from __future__ import division
import csv
import os
os.chdir(r'G:\My Drive\CurrentWork\NEP\IoTPPM')
eventlog = "event_data_alexander_rounded.csv"
with open('output_files/results/suffix_and_remaining_time_%s' % eventlog, 'r') as csvfile:
    r = csv.reader(csvfile)
    vals = dict()
    for row in r:
        next(r) # header
        l = list()
        if row[0] in vals.keys():
            l = vals.get(row[0])
        if len(row[1])==0 and len(row[2])==0:
            l.append(1)
        elif len(row[1])==0 and len(row[2])>0:
            l.append(0)
        elif len(row[1])>0 and len(row[2])==0:
            l.append(0)
        else:
            l.append(int(row[1][0]==row[2][0]))
        vals[row[0]] = l
        #print(vals)
    
l2 = list()
for k in vals.keys():
    #print('{}: {}'.format(k, vals[k]))
    l2.extend(vals[k])
    res = sum(vals[k])/len(vals[k])
    print('{}: {}'.format(k, res))

print('total: {}'.format(sum(l2)/len(l2)))
