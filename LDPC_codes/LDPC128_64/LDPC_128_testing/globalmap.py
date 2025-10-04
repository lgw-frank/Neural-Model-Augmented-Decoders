# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 23:58:09 2021

@author: Administrator
"""
import os
import tensorflow as tf
from tensorflow import keras
import fill_matrix_info as Fill_matrix
import read_TFdata as Reading
import ms_test as Decoder_module
# dictionary operations including adding,deleting or retrieving
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

def global_setting(argv):
    #command line arguments
    set_map('snr_lo', float(argv[1]))
    set_map('snr_hi', float(argv[2]))
    set_map('snr_num', int(argv[3]))
    set_map('unit_batch_size', int(argv[4]))
    set_map('num_batch_train', int(argv[5]))
    set_map('num_iterations', int(argv[6]))
    set_map('H_filename', argv[7])
    set_map('selected_decoder_type', argv[8])
    
    # the training/testing paramters setting for selected_decoder_type
    set_map('loss_process_indicator', True)
    set_map('ALL_ZEROS_CODEWORD_TESTING', False)
    set_map('loss_coefficient',5)
    
    set_map('epochs',100)
    #filling parity check matrix info
    H_filename = get_map('H_filename')
    code = Fill_matrix.Code(H_filename)
    #store it onto global space
    set_map('code_parameters', code)
    set_map('print_interval',50)
    set_map('record_interval',50)
    set_map('decoding_threshold',10)
    set_map('Rayleigh_fading', False)
    set_map('reacquire_data',True)
    if get_map('Rayleigh_fading'):
        set_map('duration', 1)
        suffix = 'Rayleigh_awgn_duration_'+str(get_map('duration'))
    else:
        suffix = 'Awgn'
    set_map('suffix',suffix)
    set_map('prefix_str','ldpc')
   
def logistic_setting():
    prefix_str = get_map('prefix_str')
    n_dims = get_map('code_parameters').check_matrix_column
    n_iteration = get_map('num_iterations')
    decoder_type = get_map('selected_decoder_type')  
    basic_dir = f'../{prefix_str.upper()}_'+str(n_dims)+'_training/ckpts/'+decoder_type+'/'+str(n_iteration)+'th'+'/'
    ckpts_dir_par = basic_dir+'par/'
    ckpt_nm = f'{prefix_str}-ckpt'        #for NMS 
    ckpts_dir = basic_dir+ckpt_nm
    #create the directory if not existing
    if not os.path.exists(ckpts_dir_par):
        os.makedirs(ckpts_dir_par)        
    restore_step = 'latest'  
    restore_info = [ckpts_dir,ckpt_nm,ckpts_dir_par,restore_step]
    return restore_info  

def log_setting():
    decoder_type = get_map('selected_decoder_type')
    n_iteration = get_map('num_iterations')
    logdir = './log/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)   
    log_filename = logdir+'FER-'+decoder_type+'-'+str(n_iteration)+'th'+'.txt'
    return log_filename

def data_setting(snr):
    snr = round(snr,2)
    # reading in training/validating data;make dataset iterator
    if get_map('ALL_ZEROS_CODEWORD_TESTING'):
        ending = 'allzero'
    else:
        ending = 'nonzero'
    unit_batch_size = get_map('unit_batch_size')
    decoder_type = get_map('selected_decoder_type')
    n_iteration = get_map('num_iterations')
    snr_lo = round(get_map('snr_lo'),2)
    snr_hi = round(get_map('snr_hi'),2)
    code = get_map('code_parameters')
    suffix = get_map('suffix')
    n_dims = code.check_matrix_column
    data_dir = '../Testing_data_gen_'+str(n_dims)+'/data/snr'+str(snr_lo)+'-'+str(snr_hi)+'dB/'
    iput_file =  data_dir +'test-'+ending+str(snr)+'dB-'+suffix+'.tfrecord'
    output_dir = data_dir+'/'+str(decoder_type)+'/'+str(n_iteration)+'th/'+str(snr)+'dB/'
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    dataset_test = Reading.data_handler(n_dims,iput_file,unit_batch_size)
    #preparing batch iterator of data file
    #dataset_test = dataset_test.take(10)
    return output_dir,dataset_test 
def retore_saved_model(restore_info):
    code = get_map('code_parameters')
    NMS_model = Decoder_module.Decoding_model()
    # Explicitly build the model with dummy input
    dummy_input_shape = (None, code.check_matrix_column)  # Replace with actual shape
    NMS_model.build(dummy_input_shape)  # ⚠️ This triggers build() in Decoder_Layer
    print("Pre-restoration weight:", NMS_model.decoder_layer.decoder_check_normalizor.numpy())
    # save restoring info
    checkpoint = tf.train.Checkpoint(myAwesomeModel=NMS_model)
    [ckpts_dir,ckpt_nm,ckpts_dir_par,restore_step] = restore_info 
    print("Ready to restore a saved latest or designated model!")
    ckpt = tf.train.get_checkpoint_state(ckpts_dir)
    if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
        if restore_step == 'latest':
          ckpt_f = tf.train.latest_checkpoint(ckpts_dir)
        else:
          ckpt_f = ckpts_dir+ckpt_nm+'-'+restore_step
        print('Loading wgt file: '+ ckpt_f)   
        status = checkpoint.restore(ckpt_f)
        print("Post-restoration weight:", NMS_model.decoder_layer.decoder_check_normalizor.numpy())
        #status.assert_existing_objects_matched() 
        status.expect_partial()
    else:
        print('Error, no qualified file found')
    return NMS_model

