"""
Created on Thu Nov 11 23:58:09 2021

@author: Administrator
"""# dictionary operations including adding,deleting or retrieving
import fill_matrix_info as Fill_matrix
import read_TFdata as Reading
import numpy as np
import os
map = {}
def set_map(key, value):
    map[key] = value
def del_map(key):
    try:
        del map[key]
    except KeyError :
        print ("key:'"+str(key)+"' non-existence")
def get_map(key):
    try:
        if key in "all":
            return map
        return map[key]
    except KeyError :
        print ("key:'"+str(key)+"' non-existence")

#global parameters setting
def global_setting(argv):
    #command line arguments
    set_map('snr_lo', float(argv[1]))
    set_map('snr_hi', float(argv[2]))
    set_map('snr_num',int(argv[3]))
    set_map('unit_batch_size', int(argv[4]))
    set_map('num_batch_train', int(argv[5]))
    set_map('num_iterations', int(argv[6]))
    set_map('H_filename', argv[7])
    set_map('selected_decoder_type', argv[8])
    
    # the training/testing paramters setting for selected_decoder_type
    set_map('ALL_ZEROS_CODEWORD_TESTING', False) 
    set_map('epochs',100)

    #set_map('termination_step',100)
    set_map('regular_matrix',True)
    set_map('generate_extended_parity_check_matrix',False)

    set_map('print_interval',50)
    set_map('record_interval',50)
    set_map('plot_low_limit',1e-3)
    
    set_map('train_snr',3.0)
    set_map('DIA_deployment',True)
    set_map('intercept_length',1000)
    set_map('relax_factor',10)
    set_map('extended_input',False)
    set_map('regenerate_data_points',True)
    set_map('prefix_str','bch')
    #filling parity check matrix info
    H_filename = get_map('H_filename')
    code = Fill_matrix.Code(H_filename)
    #store it onto global space
    set_map('code_parameters', code)
    
def logistic_setting_model():
    prefix_str = get_map('prefix_str')
    n_iteration = get_map('num_iterations')
    #snr_lo = round(get_map('snr_lo'),2)
    #snr_hi = round(get_map('snr_hi'),2)
    training_snr = get_map('train_snr')
    snr_lo = training_snr
    snr_hi = training_snr
    snr_info = f'{str(snr_lo)}-{str(snr_hi)}dB'
    prefix = 'dia/'
    ckpt_nm = f'{prefix_str}-ckpt'
    ckpts_dir = f'../DL_Training/ckpts/{snr_info}/{str(n_iteration)}th/{prefix}'+ckpt_nm
    #create the directory if not existing
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)      
    restore_model_step = 'latest'
    restore_model_info = [ckpts_dir,ckpt_nm,restore_model_step]
    return restore_model_info 
def data_setting():
    code = get_map('code_parameters')
    n_dims = code.check_matrix_col
    batch_size = get_map('unit_batch_size')
    snr_num = get_map('snr_num')
    snr_lo = get_map('snr_lo')
    snr_hi = get_map('snr_hi')
    n_iteration = get_map('num_iterations')
    list_length = n_iteration+1
    decoder_type = get_map('selected_decoder_type')
    snr_list = np.linspace(snr_lo,snr_hi,snr_num)
    data_handler_list = []
    data_dir = f'../Testing_data_gen_{str(n_dims)}/data/snr{str(snr_lo)}-{str(snr_hi)}dB/{str(decoder_type)}/{str(n_iteration)}th/' 
    for i in range(snr_num):
        snr = str(round(snr_list[i],1))
        if get_map('ALL_ZEROS_CODEWORD_TESTING'):
            file_name = f'{snr}dB/retest-allzero.tfrecord'
        else:
            file_name = f'{snr}dB/retest-nonzero.tfrecord'
        # reading in training/validating data;make dataset iterator
        file_dir = data_dir+file_name
        dataset_test = Reading.data_handler(code.check_matrix_col,file_dir,batch_size=batch_size*list_length)
        #dataset_test = dataset_test.take(10)
        data_handler_list.append(dataset_test)
        
    return data_handler_list,snr_list   