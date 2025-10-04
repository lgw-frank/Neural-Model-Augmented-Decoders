"""
Created on Thu Nov 11 23:58:09 2021

@author: Administrator
"""# dictionary operations including adding,deleting or retrieving
import read_TFdata as Reading
import fill_matrix_info as Fill_matrix
import os
import numpy as np
import pickle

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
    set_map('unit_batch_size',int(argv[4]))
    set_map('num_iterations', int(argv[5]))
    set_map('H_filename', argv[6])
    set_map('selected_decoder_type',argv[7])
    
    # the training/testing paramters setting when selected_decoder_type= Combination of VD/VS/SL,HD/HS/SL
    set_map('ALL_ZEROS_CODEWORD_TESTING', False)      

    set_map('train_snr',3.5)
    set_map('termination_threshold',10)
    
    set_map('intercept_length',1000)
    set_map('relax_factor',10)
    set_map('block_size',100)
    set_map('DIA_deployment',True)
    
    set_map('extended_input',False)
    #store it onto global space
    set_map('print_interval',100)
    set_map('record_interval',100)     
    set_map('ordering_option',4)  # training, convention, ALMT, macro_conv, macro_ALMT, optimized_conv, optimized_ALMT
         
    set_map('soft_margin',0.5)
    set_map('win_width',5)  #ensure the setting is an odd number
    set_map('sliding_strategy',True)
    set_map('prefix_str','rs')
    #filling parity check matrix info 
    H_filename = get_map('H_filename')
    code = Fill_matrix.Code(H_filename)  
    set_map('code_parameters', code)

def logistic_dia_model():
    prefix_str = get_map('prefix_str')
    nn_type = 'dia'
    n_iteration = get_map('num_iterations')
    training_snr = get_map('train_snr')
    snr_lo = training_snr
    snr_hi = training_snr
    snr_info = str(snr_lo)+'-'+str(snr_hi)+'dB'
    ckpt_nm = f'{prefix_str}-ckpt'
    ckpts_dir = f'../DL_Training/ckpts/{snr_info}/{n_iteration}th/{nn_type}/'+ckpt_nm
    #create the directory if not existing
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)        
    restore_model_step = 'latest'
    restore_model_info = [ckpts_dir,ckpt_nm,restore_model_step]
    return restore_model_info
def logistic_swa_model(): 
    prefix_str = get_map('prefix_str')
    nn_type = 'swa'
    n_iteration = get_map('num_iterations')
    train_snr = get_map('train_snr')
    intercept_length = get_map('intercept_length')
    block_size = get_map('block_size')
    win_width = get_map('win_width')
    relax_factor = get_map('relax_factor')
    snr_lo = train_snr
    snr_hi = train_snr
    snr_info = str(snr_lo)+'-'+str(snr_hi)+'dB'
    ckpt_nm = f'{prefix_str}-ckpt'
    ckpts_dir = f'../DL_Training/ckpts/{snr_info}/{n_iteration}th/{nn_type}/intercept{intercept_length}-relax_factor{relax_factor}-block_size{block_size}-width{win_width}/'+ckpt_nm
    #create the directory if not existing
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)     
    restore_model_step = 'latest'
    restore_predict_model_info = [ckpts_dir,ckpt_nm,restore_model_step]
    return restore_predict_model_info  

def logisticis_setting():
    intercept_length = get_map('intercept_length')
    relax_factor = get_map('relax_factor')
    block_size = get_map('block_size')
    win_width  = get_map('win_width')
    ordering_option =  get_map('ordering_option')
    restore_dia_info = logistic_dia_model()
    restore_swa_info = logistic_swa_model()
    
    restore_list = [restore_dia_info,restore_swa_info]
    
    total_teps_list = acquire_ordered_teps()
    decoding_matrix = np.stack(total_teps_list[ordering_option]) #ALMT ordering
    print(f'Actual decoding path:{decoding_matrix} with shape:{decoding_matrix.shape}')
    logdir = './log/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log_filename = logdir+f'intercept{intercept_length}-relax{relax_factor}-block_size{block_size}-width{win_width}.txt'  
    logistics = [restore_list,decoding_matrix,log_filename]
    return logistics

def acquire_ordered_teps():
    intercept_length = get_map('intercept_length')
    relax_factor = get_map('relax_factor')
    if get_map('extended_input'):
        suffix = '-extended'
    else:
        suffix = ''
    if get_map('DIA_deployment'):
        ending = '-dia'
    else:
        ending = ''
    output_ranking_dir = '../Optimizing_decoding_path/ckpts/ranking_patterns/'
    output_order_file = f'{output_ranking_dir}intercept{intercept_length}-relax{relax_factor}-ranking_orders{suffix}{ending}.pkl'
    
    with open(output_order_file,'rb') as f:
        full_teps_ordering_list = pickle.load(f) 
    total_teps_list = [full_teps_ordering_list[i][:intercept_length] for i in range(len(full_teps_ordering_list))]
    return total_teps_list

def data_setting():
    code = get_map('code_parameters')
    n_dims = code.check_matrix_col
    batch_size = get_map('unit_batch_size')
    snr_num = get_map('snr_num')
    snr_lo = get_map('snr_lo')
    snr_hi = get_map('snr_hi')
    snr_list = np.linspace(snr_lo,snr_hi,snr_num)
    n_iteration = get_map('num_iterations')
    list_length = n_iteration+1
    data_handler_list = []
    data_dir = '../Testing_data_gen_'+str(n_dims)+'/data/snr'+str(snr_lo)+'-'+str(snr_hi)+'dB/'
    decoder_type = get_map('selected_decoder_type')
    if get_map('ALL_ZEROS_CODEWORD_TESTING'):
        file_name = 'retest-allzero.tfrecord'
    else:
        file_name = 'retest-nonzero.tfrecord'    
    for i in range(snr_num):
        snr = str(round(snr_list[i],2))
        input_dir = data_dir+decoder_type+'/'+str(n_iteration)+'th/'+snr+'dB/'
        # reading in training/validating data;make dataset iterator
        data_file_dir = input_dir+file_name
        dataset_test = Reading.data_handler(code.check_matrix_col,data_file_dir,batch_size*list_length)
        data_handler_list.append(dataset_test)
       
    return data_handler_list,snr_list               
                   