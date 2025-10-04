"""
Created on Thu Nov 11 23:58:09 2021

@author: Administrator
"""# dictionary operations including adding,deleting or retrieving
import fill_matrix_info as Fill_matrix
import read_TFdata as Reading
import os,sys
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
    set_map('unit_batch_size', int(argv[3]))
    set_map('num_batch_train', int(argv[4]))
    set_map('num_iterations', int(argv[5]))
    set_map('H_filename', argv[6])
    set_map('selected_decoder_type', argv[7])
    
    # the training/testing paramters setting for selected_decoder_type
    set_map('ALL_ZEROS_CODEWORD_TRAINING', False)

    #filling parity check matrix info
    H_filename = get_map('H_filename')
    code = Fill_matrix.Code(H_filename)
    #store it onto global space
    set_map('code_parameters', code)
    
    set_map('print_interval',1000)
    set_map('record_interval',1000)
    
    set_map('DIA_deployment',False)
    set_map('intercept_length',1000)
    set_map('relax_factor',10)

    set_map('create_update_teps_ranking',True)

    set_map('draw_low_limit',1e-3)
    
    set_map('extended_input',False)
    set_map('prefix_str','rs')
    

def logistic_setting_model():
    prefix_str = get_map('prefix_str')
    nn_type = 'dia'
    n_iteration = get_map('num_iterations')
    snr_lo = round(get_map('snr_lo'),2)
    snr_hi = round(get_map('snr_hi'),2)
    ckpt_nm = f'{prefix_str}-ckpt' 
    snr_info = f'{snr_lo}-{snr_hi}dB/'
    ckpts_dir = f'../DL_training/ckpts/{snr_info}{n_iteration}th/{nn_type}/'+ckpt_nm
    #create the directory if not existing
    if not os.path.exists(ckpts_dir):
        print('Error, no model paras found!')  
        sys.exit(-1) 
    restore_step = 'latest'
    restore_info = [ckpts_dir,restore_step,ckpt_nm]
    return restore_info  

def data_iteration(code, unit_batch_size):
    prefix_str = get_map('prefix_str')
    # Training data directory
    code_length = code.n
    snr_lo = round(get_map('snr_lo'), 1)
    snr_hi = round(get_map('snr_hi'), 1)
    n_iteration = get_map('num_iterations')
    list_length = n_iteration+1
    macro_size = unit_batch_size*list_length
    basic_dir = '../Training_data_gen_' + str(code_length) + '/data/snr' + str(snr_lo) + '-' + str(snr_hi) + 'dB/'
    decoder_type = get_map('selected_decoder_type')
    data_dir = basic_dir + str(n_iteration) + 'th/' + decoder_type + '/'
    # Reading in training/validating data; make dataset iterator
    if get_map('ALL_ZEROS_CODEWORD_TRAINING'):  
        if get_map('extended_input'):
            file_name = f'{prefix_str}-retrain-allzero-extended.tfrecord'
        else:
            file_name = f'{prefix_str}-retrain-allzero.tfrecord'
    else:
        if get_map('extended_input'):
            file_name = f'{prefix_str}-retrain-nonzero-extended.tfrecord'
        else:
            file_name = f'{prefix_str}-retrain-nonzero.tfrecord'
    dir_file = data_dir + file_name
    dataset_train = Reading.data_handler(dir_file,'zero',macro_size )
    #dataset_train = dataset_train.take(5)
    return dataset_train.as_numpy_iterator()