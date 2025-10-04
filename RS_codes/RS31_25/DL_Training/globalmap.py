"""
Created on Thu Nov 11 23:58:09 2021

@author: Administrator
"""# dictionary operations including adding,deleting or retrieving
import tensorflow as tf
import read_TFdata as Reading
import fill_matrix_info as Fill_matrix
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
    set_map('unit_batch_size', int(argv[3]))
    set_map('num_iterations', int(argv[4]))
    set_map('H_filename', argv[5])
    set_map('selected_decoder_type',argv[6])  
    
    # the training/testing paramters setting when selected_decoder_type= Combination of VD/VS/SL,HD/HS/SL
    set_map('ALL_ZEROS_CODEWORD_TRAINING', False)   
    
    set_map('initial_learning_rate', 0.01)
    set_map('decay_rate', 0.95)
    set_map('decay_step', 500)  
    
    set_map('dia_termination_step',1500)
    set_map('swa_termination_step',3000)
    
    set_map('print_interval',100)
    set_map('record_interval',100) 

    set_map('dia_model_train',False) 
    set_map('swa_model_train',True) 

    set_map('regnerate_training_samples',False)
    set_map('win_width',5)  #ensure the setting is an odd number
    set_map('intercept_length',1000)
    set_map('relax_factor',10)
    set_map('block_size',100)    
    
    set_map('DIA_deployment',True)
    set_map('ALMLT_available',True)
    
    set_map('train_snr',3.5)
    
    #filling parity check matrix info
    H_filename = get_map('H_filename')
    code = Fill_matrix.Code(H_filename) 

    #store it onto global space
    set_map('code_parameters', code)
    set_map('nn_train',True)  
    set_map('regulation_weight',10.)      
    set_map('extended_input',False)
    set_map('option_without_dia',2)  # training 0, convention 1, ALMT 2, macro_conv 3, macro_ALMT 4, optimized_conv, optimized_ALMT
    set_map('option_with_dia',4)  # training 0, convention 1, ALMT 2, macro_conv 3, macro_ALMT 4, optimized_conv, optimized_ALMT
    set_map('original_NMS_indicator',False)
    set_map('prefix_str','rs')

def logistic_dia_model(original_NMS_indicator=False):
    prefix_str = get_map('prefix_str')
    nn_type = 'dia'
    n_iteration = get_map('num_iterations')
    snr_lo = round(get_map('snr_lo'),2)
    snr_hi = round(get_map('snr_hi'),2)
    ckpt_nm = f'{prefix_str}-ckpt' 
    snr_info = f'{snr_lo}-{snr_hi}dB/'
    if original_NMS_indicator:
        ckpts_dir = f'./convention_ckpts/{snr_info}{n_iteration}th/{nn_type}/'+ckpt_nm
    else:
        ckpts_dir = f'./ckpts/{snr_info}{n_iteration}th/{nn_type}/'+ckpt_nm
    #create the directory if not existing
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)      
    restore_model_step = 'latest'
    restore_dia_info = [ckpts_dir,ckpt_nm,restore_model_step]
    return restore_dia_info     

def logistic_swa_model(original_NMS_indicator=False):
    prefix_str = get_map('prefix_str')
    nn_type = 'swa'
    n_iteration = get_map('num_iterations')
    intercept_length = get_map('intercept_length')
    relax_factor = get_map('relax_factor')
    block_size = get_map('block_size')
    win_width = get_map('win_width')
    snr_lo = round(get_map('snr_lo'),2)
    snr_hi = round(get_map('snr_hi'),2)
    snr_info = f'{snr_lo}-{snr_hi}dB/'
    ckpt_nm = f'{prefix_str}-ckpt' 
    if original_NMS_indicator:
        ckpts_dir = f'./convention_ckpts/{snr_info}{n_iteration}th/{nn_type}/intercept{intercept_length}-relax_factor{relax_factor}-block_size{block_size}-width{win_width}/'+ckpt_nm
    else:
        ckpts_dir = f'./ckpts/{snr_info}{n_iteration}th/{nn_type}/intercept{intercept_length}-relax_factor{relax_factor}-block_size{block_size}-width{win_width}/'+ckpt_nm
    #create the directory if not existing
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)   
    restore_model_step = ''
    restore_swa_info = [ckpts_dir,ckpt_nm,restore_model_step]
    return restore_swa_info            


def save_model_dir(nn_type):
    n_iteration = get_map('num_iterations')
    ckpts_dir = './ckpts/'+nn_type+'/'+str(n_iteration)+'th'+'/'
    # checkpoint
    checkpoint_prefix = os.path.join(ckpts_dir,'ckpt')
    if not os.path.exists(os.path.dirname(checkpoint_prefix)):
        os.makedirs(os.path.dirname(checkpoint_prefix))
    return checkpoint_prefix
                 
def base_dataset(original_NMS_indicator=False):
    prefix_str = get_map('prefix_str')
    code = get_map('code_parameters')
    list_length =   get_map('num_iterations') + 1
    batch_size = get_map('unit_batch_size')*list_length
    code_length = code.n
    snr_lo = round(get_map('snr_lo'), 2)
    snr_hi = round(get_map('snr_hi'), 2)
    n_iteration = get_map('num_iterations')
    decoder_type = get_map('selected_decoder_type')
    data_dir = f'../Training_data_gen_{code_length}/data/snr{snr_lo}-{snr_hi}dB/{n_iteration}th/{decoder_type}/'
    file_name = f'{prefix_str}-retrain-allzero.tfrecord' if get_map('ALL_ZEROS_CODEWORD_TRAINING') else f'{prefix_str}-retrain-nonzero.tfrecord'
    if original_NMS_indicator:
        file_name = 'convention-'+file_name
    file_path = data_dir + file_name        
    dataset = Reading.data_handler(code_length, file_path, batch_size)
    dataset = dataset.take(100)
    return dataset.cache()  # no cache yet

def build_training_dataset(original_NMS_indicator=False):
    dataset = base_dataset(original_NMS_indicator)
    return dataset.shuffle(1000).repeat().prefetch(tf.data.AUTOTUNE)
                    
def optimizer_setting():
    #optimizing settings
    decay_rate = get_map('decay_rate')
    initial_learning_rate = get_map('initial_learning_rate')
    decay_steps = get_map('decay_step')
    exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps, decay_rate,staircase=True)
    return exponential_decay

def log_setting(restore_info,checkpoint):
    (ckpts_dir,ckpt_nm,_) = restore_info
    n_iteration = get_map('num_iterations')
    # summary recorder
    # Create the log directory
    log_dir = './tensorboard/'+str(n_iteration)+'th'+'/'
    os.makedirs(log_dir, exist_ok=True)
    # Set up TensorBoard writer
    summary_writer = tf.summary.create_file_writer(log_dir)     # the parameter is the log folder we created
    manager_current = tf.train.CheckpointManager(checkpoint, directory=ckpts_dir, checkpoint_name=ckpt_nm, max_to_keep=5)
    logger_info = (summary_writer,manager_current)
    return logger_info