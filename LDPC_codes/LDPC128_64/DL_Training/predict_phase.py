# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:46:41 2024

@author: zidonghua_30
"""
import globalmap as GL
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
import nn_net as CRNN_DEF
import interval_boundary as Boundary
from collections import Counter
#from collections import Counter,defaultdict,OrderedDict
#import numpy as np
import math
import numpy as np
import pickle
import  os
import sys
import matplotlib.pyplot as plt
import nn_net as NN_struct
from collections.abc import Iterable
#import tensorflow.keras.backend as K
#from tensorflow.keras.losses import BinaryCrossentropy
def retore_saved_model(restore_ckpts_dir,restore_step,ckpt_nm):
    print("\nReady to restore a saved latest or designated model!")
    ckpt = tf.train.get_checkpoint_state(restore_ckpts_dir)
    if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
      if restore_step == 'latest':
        ckpt_f = tf.train.latest_checkpoint(restore_ckpts_dir)
        start_step = int(ckpt_f.split('-')[-1]) + 1
      else:
        ckpt_f = restore_ckpts_dir+ckpt_nm+'-'+restore_step
        start_step = int(restore_step)+1
      print('Loading wgt file: '+ ckpt_f)   
    else:
      print('Error, no qualified file found')
      sys.exit(-1)
    return start_step,ckpt_f


class CustomCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='custom_categorical_accuracy', **kwargs):
        super(CustomCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.cat_acc_metric = tf.metrics.CategoricalAccuracy()
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.cat_acc_metric.update_state(y_true, y_pred, sample_weight)
    def result(self):
        return self.cat_acc_metric.result()
    def reset_state(self):
        self.cat_acc_metric.reset_state()
        
def format_weights(weights, decimals=4):
    formatted = []
    for layer in weights:
        if isinstance(layer, np.ndarray):
            # Convert array to list and format each element
            layer_list = layer.tolist()
            # Recursively format nested lists
            formatted.append(recursive_round(layer_list, decimals))
        else:
            # Handle non-array weights (unlikely in get_weights())
            formatted.append(round(float(layer), decimals))
    return formatted

def recursive_round(data, decimals):
    if isinstance(data, list):
        return [recursive_round(x, decimals) for x in data]
    elif isinstance(data, (float, np.floating)):
        return round(data, decimals)
    return data
       
def NN_gen(model,restore_info,observe_length=100,input_dim=3): 
    exponential_decay = GL.optimizer_setting()
    optimizer = tf.keras.optimizers.Adam(exponential_decay) 
    # Explicitly build the model with dummy input
    if input_dim == 3:
        dummy_input = tf.zeros([1, observe_length, 1])
    else:
        dummy_input = tf.zeros([1, observe_length])
    _ = model(dummy_input)  # Forces weight creation
    # Now weights exist and can be inspected/restored
    weights = model.get_weights()
    rounded_weights = format_weights(weights, decimals=2)
    print("Pre-restoration weights:\n",rounded_weights)  # Removes dtype and converts to native Python floats
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model, myAwesomeOptimizer=optimizer)
    logger_info = GL.log_setting(restore_info,checkpoint)
    start_step = 0
    #unpack related info for restoraging
    [ckpts_dir,ckpt_nm,restore_step] = restore_info  
    if restore_step:
        start_step,ckpt_f = retore_saved_model(ckpts_dir,restore_step,ckpt_nm)
        status = checkpoint.restore(ckpt_f)
        weights = model.get_weights()
        rounded_weights = format_weights(weights, decimals=2)
        print("\nPost-restoration weights:\n",rounded_weights)
        #status.assert_existing_objects_matched()
        status.expect_partial()
    return model,optimizer,logger_info,start_step

def choose_ordered_teps(option_without_dia,option_with_dia):
    intercept_length = GL.get_map('intercept_length')
    relax_factor = GL.get_map('relax_factor')
    if GL.get_map('extended_input'):
        suffix = '-extended'
    else:
        suffix = ''
    ending = ''
    output_ranking_dir = '../Optimizing_decoding_path/ckpts/ranking_patterns/'
    output_order_file = f'{output_ranking_dir}intercept{intercept_length}-relax{relax_factor}-ranking_orders{suffix}{ending}.pkl'  
    with open(output_order_file,'rb') as f:
        full_teps_ordering_list = pickle.load(f) 
    total_teps_list1 = [full_teps_ordering_list[i][:intercept_length] for i in range(len(full_teps_ordering_list))]
    ending = '-dia'
    output_order_file = f'{output_ranking_dir}intercept{intercept_length}-relax{relax_factor}-ranking_orders{suffix}{ending}.pkl'  
    with open(output_order_file,'rb') as f:
        full_teps_ordering_list = pickle.load(f) 
    total_teps_list2 = [full_teps_ordering_list[i][:intercept_length] for i in range(len(full_teps_ordering_list))]
    decoding_matrix1 = np.stack(total_teps_list1[option_without_dia]) #ALMT ordering
    decoding_matrix2 = np.stack(total_teps_list2[option_with_dia]) #ALMT_MS with DIA ordering
    return [decoding_matrix1,decoding_matrix2]

def plot_rows(records_matrix):
    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(2, 2)
    # Sample matrix and class labels
    matrix_data = records_matrix[:,:-1]
    class_labels = records_matrix[:,-1]  
    # Get unique class labels
    unique_classes = np.unique(class_labels)
    # Sort the items by keys in ascending order
    counter_instance = Counter(class_labels)
    sorted_items = sorted(counter_instance.items())   
    # Print the sorted dictionary
    for key, value in sorted_items:
        print(f"{key}: {value}")
    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(3, 2)
    # Plot rows belonging to each class separately
    
    axs[0, 0].set_title("Cases of Class '-1'")
    for i, row in enumerate(matrix_data[class_labels == unique_classes[0]]):
        axs[0, 0].plot(row)
        axs[0,0].set_xticks([])   # Hide x-axis ticks
    axs[0, 1].set_title("Cases of Class '0'")
    for i, row in enumerate(matrix_data[class_labels == unique_classes[1]]):
        axs[0, 1].plot(row)
        axs[0,1].set_xticks([])   # Hide x-axis ticks
    axs[1, 0].set_title("Cases of Class '2'")
    for i, row in enumerate(matrix_data[class_labels == unique_classes[3]]):
        axs[1, 0].plot(row)
        axs[1,0].set_xticks([])   # Hide x-axis ticks
    axs[1, 1].set_title("Cases of Class '4'")
    for i, row in enumerate(matrix_data[class_labels == unique_classes[5]]):
        axs[1, 1].plot(row)
        axs[1,1].set_xticks([])   # Hide x-axis ticks
    axs[2, 0].set_title("Cases of Class '9'")
    for i, row in enumerate(matrix_data[class_labels == unique_classes[10]]):
        axs[2, 0].plot(row)
        axs[2,0].set_xticks(range(0, 11, 2))  # Set x-axis ticks
    axs[2, 1].set_title("Cases of Class '11'")
    for i, row in enumerate(matrix_data[class_labels == unique_classes[12]]):
        axs[2, 1].plot(row)
        axs[2,1].set_xticks(range(0, 11, 2))  # Set x-axis ticks
    # Adjust layout to prevent overlap
    plt.tight_layout() 
    # Create the directory if it doesn't exist
    directory = './figs/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Save the figure to a file in the specified directory
    file_path = os.path.join(directory, 'subplots.png')
    plt.savefig(file_path, dpi=300)
    plt.show()


def shuffle_data(matrix, labels):
    # Generate a random permutation
    permutation = tf.random.shuffle(tf.range(len(labels)))   
    # Shuffle the matrix and labels according to the permutation
    shuffled_inputs = tf.gather(matrix, permutation)
    shuffled_labels = tf.gather(labels, permutation)  
    # acquire weights for imbalanced dataset 
    total_samples = len(shuffled_inputs)
    unique_classes = np.unique(shuffled_labels)  #actually binary choice
    num_classes = len(unique_classes)
    one_hot_matrix = tf.keras.utils.to_categorical(shuffled_labels, num_classes)
    count_class = tf.reduce_sum(one_hot_matrix,axis=0)
    class_weight = (total_samples) / (tf.cast(num_classes,tf.float32) * count_class)
    weight_samples = tf.constant(tf.reduce_sum(one_hot_matrix*class_weight,axis=-1))
    return shuffled_inputs,shuffled_labels,weight_samples,one_hot_matrix
                


#common sense definition taking into account all bits of codewords         
def calculation_loss(soft_output,one_hot_labels,batch_weight):
    penalty_coefficient = GL.get_map('regulation_weight')
    #cross entroy
    soft_output = tf.maximum(soft_output,1e-30)
    CE_vector = tf.reduce_sum(-tf.math.log(soft_output)*one_hot_labels,axis=-1)
    #normalize output
    indicator_vector = tf.where(soft_output[:,0]<soft_output[:,1],True,False)
    #penalty for premature ending
    #1 denotes ending of further sliding window. 0 cotinuous sliding.
    ground_vector = tf.where(one_hot_labels[:,0]==1,True,False)
    penalty_vector = tf.where(tf.logical_and(indicator_vector==True,ground_vector==True),penalty_coefficient,1.)
    CE_loss = tf.reduce_sum(CE_vector*penalty_vector*batch_weight)
    return CE_loss 

def normalize_window(window_block):
    mean = np.mean(window_block,axis=1,keepdims=True)
    std_dev = np.std(window_block,axis=1,keepdims=True)
    normalized_window = (window_block - mean) / std_dev
    return normalized_window

def train_sliding_window(DIA_model, SWA_model,restore_list):
    nn_type = 'SWA_model'
    n_iteration = GL.get_map('num_iterations')
    print_interval = GL.get_map('print_interval')
    record_interval = GL.get_map('record_interval')
    DIA_indicator = GL.get_map('DIA_deployment')
    batch_size = GL.get_map('unit_batch_size')
    snr_lo = round(GL.get_map('snr_lo'),2)
    snr_hi = round(GL.get_map('snr_hi'),2)
    snr_info = f'{snr_lo}-{snr_hi}dB/'
    #query partition of MRB
    block_size = GL.get_map('block_size')
    win_width = GL.get_map('win_width')
    intercept_length = GL.get_map('intercept_length')
    relax_factor = GL.get_map('relax_factor')
    option_without_dia =  GL.get_map('option_without_dia')
    option_with_dia =  GL.get_map('option_with_dia')
    
    restore_dia_info,restore_swa_info = restore_list
    if not intercept_length%block_size:
        if DIA_indicator:
            DIA_model,_,_,_= NN_gen(DIA_model,restore_dia_info,DIA_model.actual_length)  
            decoding_matrix = choose_ordered_teps(option_without_dia, option_with_dia)[1]
            suffix = '-dia'
        else:
            decoding_matrix = choose_ordered_teps(option_without_dia, option_with_dia)[0]
            suffix = ''
    else:      
        print('Block size has to be dividable by intercept_length')
        sys.exit(-1)
    

    model_dir = f'./data/{nn_type}/{snr_info}{n_iteration}th/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    nominal_path =  f'intercept{intercept_length}-relax{relax_factor}{suffix}.pkl'
    data_file_name = model_dir+nominal_path
    if GL.get_map('regenerate_swa_samples'):
        selected_ds = GL.base_dataset()
        saved_summary, records_matrix = Boundary.query_samples(selected_ds,DIA_model,decoding_matrix,data_file_name)
    else:
        if not os.path.exists(data_file_name):
            print('Data file not found, please check again!')
            sys.exit(-1)
        with open(data_file_name, "rb") as fh:
            saved_summary = pickle.load(fh)
            records_matrix = pickle.load(fh)   
    volume_inputs,volume_labels = Boundary.reformat_inputs(records_matrix)        
    #preparing training neural model for soft_max output of possible index of order patterns, or equivalently, guessting at the probabilities of the i-th doecoding path
    print(f'Summary:(actual_size,correct,failed,undeteced)={saved_summary}')

    #load sliding window aid model
    SWA_model,optimizer,log_info,step= NN_gen(SWA_model,restore_swa_info,SWA_model.input_width,input_dim=2) 
    summary_writer,manager_current = log_info
    # Instantiate the metric
    custom_accuracy = CustomCategoricalAccuracy()
    # Instantiate the loss function
    #unpack related info for restoraging    
    #initialize starting point
    num_counter = math.ceil(volume_inputs.shape[0]/batch_size)
    wrapped_info = shuffle_data(volume_inputs,volume_labels)  
    volume_inputs,volume_labels,weight_samples,one_hot_matrix = wrapped_info
    custom_accuracy.reset_state() 
    while step < GL.get_map('swa_termination_step') and GL.get_map('nn_train'):           
        i = step%num_counter
        inputs = volume_inputs[i*batch_size:(i+1)*batch_size]
        batch_weight = weight_samples[i*batch_size:(i+1)*batch_size]
        one_hot_labels = one_hot_matrix[i*batch_size:(i+1)*batch_size]
        with tf.GradientTape() as tape:
            outputs = SWA_model(inputs)
            #fcn.print_model()
            loss = calculation_loss(outputs,one_hot_labels,batch_weight)
        total_variables = SWA_model.trainable_variables         
        grads = tape.gradient(loss,total_variables)
        grads_and_vars=zip(grads, total_variables)
        capped_gradients = [(tf.clip_by_norm(grad,5e2), var) for grad, var in grads_and_vars if grad is not None]
        optimizer.apply_gradients(capped_gradients) 
        # print("Gradients:", grads)
        # print("Loss:", loss)
        custom_accuracy.update_state(y_true=one_hot_labels,y_pred=outputs,sample_weight=batch_weight)
        step = step + 1
        if step % print_interval == 0:   
            total_metric_value = custom_accuracy.result()
            print('Step:%d  Loss:%.3f Error rate:%.3f'%(step,loss.numpy(),1.0-total_metric_value))                                                            
        if step % record_interval == 0:
            manager_current.save(checkpoint_number=step)                   
    
    #verifying trained pars from start to end    
    success_sum = 0   # True positive and True negative cases
    fail_sum1 = 0     # False positive, partially beneficial
    fail_sum2 = 0    #False negative, partially beneficial
    actual_size = 0
    
    wrapped_info = shuffle_data(volume_inputs,volume_labels)  
    volume_inputs,volume_labels,weight_samples,_ = wrapped_info
    dist_samples = Counter(list(weight_samples.numpy()))
    dist_labels = Counter(list(volume_labels.numpy()))
    print(f'dist_samples:{dist_samples}')
    print(f'dist_labels:{dist_labels}')
    for i in range(num_counter):
        inputs = volume_inputs[i*batch_size:(i+1)*batch_size]
        labels = (volume_labels[i*batch_size:(i+1)*batch_size].numpy()).astype(int)
        batch_count = len(labels)
        actual_size += batch_count
        outputs = SWA_model(inputs)
        hard_prediction_index = tf.cast(tf.argmax(outputs,axis=-1),tf.int32)
        success_counter = tf.reduce_sum(tf.where(labels==hard_prediction_index,1,0))
        fail_counter1 = tf.reduce_sum(tf.where(labels>hard_prediction_index,1,0))
        fail_counter2 = tf.reduce_sum(tf.where(labels<hard_prediction_index,1,0))
        success_sum += success_counter
        fail_sum1 += fail_counter1
        fail_sum2 += fail_counter2
        if (i+1)%10 == 0:
            print(f'S:{success_sum} F1:{fail_sum1} F2:{fail_sum2}') 

    success_rate = success_sum/actual_size
    fail_rate1 = fail_sum1/actual_size    
    fail_rate2 = fail_sum2/actual_size    
    print('Summary of validation:')
    print(f'Total counts:{actual_size}')
    print(f'dist_samples:{dist_samples}')
    print(f'dist_labels:{dist_labels}')
    print(f'S:{success_sum} F1:{fail_sum1} F2:{fail_sum2}') 
    print(f'Success rate:{success_rate:.4f} FR1:{fail_rate1:.4f} FR2:{fail_rate2:.4f}') 
    #save on disk files
    log_dir = f'./log/{nn_type}/{snr_info}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  
    with open(log_dir+f'intercept{intercept_length}-relax{relax_factor}-block_size{block_size}-width{win_width}.txt', "a+") as f:
        f.write(f'\nFor Summery of {snr_info[:-1]}dB:\n')
        f.write(f'dist_samples:{dist_samples}\n')
        f.write(f'dist_labels:{dist_labels}\n')
        f.write(f'S:{success_sum} F1:{fail_sum1} F2:{fail_sum2}\n') 
        f.write(f'Success rate:{success_rate:.4f} FR1:{fail_rate1:.4f} FR2:{fail_rate2:.4f}\n') 