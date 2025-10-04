"""
Created on Thu Nov 11 23:58:09 2021

@author: Administrator
"""# dictionary operations including adding,deleting or retrieving
import read_TFdata as Reading
import fill_matrix_info as Fill_matrix
import numpy as np

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
    set_map('num_iterations',int(argv[4]))
    set_map('unit_batch_size',int(argv[5]))
    set_map('H_filename', argv[6])
    set_map('selected_decoder_type',argv[7])
    
    # the training/testing paramters setting when selected_decoder_type= Combination of VD/VS/SL,HD/HS/SL
    set_map('ALL_ZEROS_CODEWORD_TESTING', False)      
  
    #filling parity check matrix info
    H_filename = get_map('H_filename')
    code = Fill_matrix.Code(H_filename)
    #store it onto global space
    set_map('code_parameters', code)  
    
    set_map('termination_threshold',10)
    set_map('maximum_order',1)     
    

    set_map('print_interval',100)
    set_map('record_interval',100)  

def data_setting():
    code = get_map('code_parameters')
    n_dims = code.n
    batch_size = get_map('unit_batch_size')
    snr_num = get_map('snr_num')
    snr_lo = get_map('snr_lo')
    snr_hi = get_map('snr_hi')
    snr_list = np.linspace(snr_lo,snr_hi,snr_num)
    data_handler_list = []
    data_dir = '../Testing_data_gen_'+str(n_dims)+'/data/snr'+str(snr_lo)+'-'+str(snr_hi)+'dB/' 
    for i in range(snr_num):
        snr = str(round(snr_list[i],1))
        if get_map('ALL_ZEROS_CODEWORD_TESTING'):
            file_name = 'test-zero'+str(snr)+'dB-Awgn.tfrecord'
        else:
            file_name = 'test-nonzero'+str(snr)+'dB-Awgn.tfrecord'
        # reading in training/validating data;make dataset iterator
        file_dir = data_dir+file_name
        dataset_test = Reading.data_handler(code.n,file_dir,batch_size)
        data_handler_list.append(dataset_test)
        
    return data_handler_list,snr_list    
           
def post_data_setting():
    code = get_map('code_parameters')
    n_dims = code.n
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
            file_name = f'{snr}dB/retest-zero.tfrecord'
        else:
            file_name = f'{snr}dB/retest-nonzero.tfrecord'
        # reading in training/validating data;make dataset iterator
        file_dir = data_dir+file_name
        dataset_test = Reading.data_handler(code.n,file_dir,batch_size*list_length)
        data_handler_list.append(dataset_test)
        
    return data_handler_list,snr_list                    