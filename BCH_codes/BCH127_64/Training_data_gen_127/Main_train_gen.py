# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:31:50 2021

@author: Administrator
"""
import numpy as np
import sys
import os
# Run as follows:
np.set_printoptions(precision=3)
import fill_matrix_info as Fill_matrix
import globalmap as GL
import data_generating as Data_gen
# python main.py 1.5 3 16 1e4   wimax_1056_0.83.alist            
#command line arguments
sys.argv = "python 3.0 3.0 100 100 BCH_127_64_10_strip.alist".split()
GL.set_map('snr_lo', float(sys.argv[1]))
GL.set_map('snr_hi', float(sys.argv[2]))
GL.set_map('batch_size', int(sys.argv[3]))
GL.set_map('training_batch_number', int(sys.argv[4]))
GL.set_map('H_filename', sys.argv[5])

# setting global parameters
H_filename=GL.get_map('H_filename')
batch_size = GL.get_map('batch_size')

code = Fill_matrix.Code(H_filename)
GL.set_map('code_parameters', code)
GL.set_map('ALL_ZEROS_CODEWORD_TRAINING', False)
GL.set_map('extended_input', False)
GL.set_map('prefix_str', 'bch')

#training setting
#retrieving global paramters of the code
n = code.check_matrix_column
train_batch = GL.get_map('training_batch_number')

snr_lo = GL.get_map('snr_lo')
snr_hi = GL.get_map('snr_hi')
SNRs = [snr_lo,snr_hi]

#create directory if not existence      
file_dir = './data/snr'+str(round(snr_lo,2))+'-'+str(round(snr_hi,2))+'dB/'
if not os.path.exists(file_dir ):
  os.makedirs(file_dir) 
nDatas_train = train_batch*batch_size  
#generating training data
train_data,train_labels = Data_gen.training_data_generating(code, SNRs,nDatas_train)   
# make training set file
data = (train_data,train_labels)
prefix_str = GL.get_map('prefix_str')
if GL.get_map('ALL_ZEROS_CODEWORD_TRAINING'):  
    if GL.get_map('extended_input'):
        file_name = file_dir + f'{prefix_str}-train-allzero-extended.tfrecord'
    else:
        file_name = file_dir + f'{prefix_str}-train-allzero.tfrecord'
else:
    if GL.get_map('extended_input'):
        file_name = file_dir + f'{prefix_str}-train-nonzero-extended.tfrecord'
    else:
        file_name = file_dir + f'{prefix_str}-train-nonzero.tfrecord'

Data_gen.make_tfrecord(data, out_filename=file_name)
    
print("Data for training generated successfully!")