# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:53:46 2022

@author: zidonghua_30
"""
#from hyperts.toolbox import from_3d_array_to_nested_df
import globalmap as GL
import tensorflow as tf
import nn_net as NN_struct
import ordered_statistics_decoding as OSD_mod
import re
import pickle
import os
from collections import Counter,defaultdict,OrderedDict
import numpy as  np
import ast
import time,sys

def retore_file_retrieval(restore_ckpts_dir,restore_step,ckpt_nm):
    print("\nReady to restore a saved latest or designated model!")
    ckpt = tf.train.get_checkpoint_state(restore_ckpts_dir)
    if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
      if restore_step == 'latest':
        ckpt_f = tf.train.latest_checkpoint(restore_ckpts_dir)
      else:
        ckpt_f = restore_ckpts_dir+ckpt_nm+'-'+restore_step
      print('Loading wgt file: '+ ckpt_f)   
    else:
      print('Error, no qualified file found')
      sys.exit(-1)
    return ckpt_f

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

def NNs_gen(restore_list):
    win_width = GL.get_map('win_width')
    DIA_model = NN_struct.conv_bitwise()
    DIA_model = model_gen(DIA_model,restore_list[0],DIA_model.actual_length,input_dim=3)
    print('\n')
    SWA_model = NN_struct.osd_arbitrator(win_width)
    SWA_model = model_gen(SWA_model,restore_list[1],SWA_model.input_width,input_dim=2)
    return DIA_model,SWA_model
 
def model_gen(model,restore_info,observe_length=1,input_dim=3): 
    if input_dim == 3:
        dummy_input = tf.zeros([1, observe_length, 1])
    else:
        dummy_input = tf.zeros([1, observe_length])
    _ = model(dummy_input)  # Forces weight creation
    # Now weights exist and can be inspected/restored
    weights = model.get_weights()
    rounded_weights = format_weights(weights, decimals=2)
    print("Pre-restoration weights:\n",rounded_weights)  # Removes dtype and converts to native Python floats
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
    #unpack related info for restoraging
    [ckpts_dir,ckpt_nm,restore_step] = restore_info 
    if restore_step:
        ckpt_f = retore_file_retrieval(ckpts_dir,restore_step,ckpt_nm)
        status = checkpoint.restore(ckpt_f)
        weights = model.get_weights()
        rounded_weights = format_weights(weights, decimals=2)
        print("\nPost-restoration weights:\n",rounded_weights)
        #status.assert_existing_objects_matched()
        status.expect_partial()
    return model

def filter_order_patterns(decoding_path):
    residual_path = [] 
    threshold_sum = GL.get_map('threshold_sum')
    nomial_path_length = GL.get_map('decoding_length')  
    for order_pattern in decoding_path:
        if sum(order_pattern) <= threshold_sum:
            residual_path.append(order_pattern) 
    return residual_path[:nomial_path_length]
      
def calculate_loss(inputs,labels):
    labels = tf.cast(labels,tf.float32)  
    #measure discprepancy via cross entropy metric which acts as the loss definition for deep learning per batch         
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-inputs, labels=labels))
    return  loss

def calculate_list_cross_entropy_ber(input_list,labels):
    cross_entropy_list = []
    ber_list = []
    for i in range(len(input_list)):
        cross_entropy_element = calculate_loss(input_list[i],labels).numpy()
        cross_entropy_list.append(cross_entropy_element)
        current_hard_decision = tf.where(input_list[i]>0,0,1)
        compare_result = tf.where(current_hard_decision!=labels,1,0)
        num_errors = tf.reduce_sum(compare_result)
        ber_list.append(num_errors)
    return cross_entropy_list,ber_list

#fixed scheduling for decoding path    
def generate_teps(osd,residual_path):
    num_segments = GL.get_map('num_blocks')
    #segmentation of MRB
    _,boundary_MRB = GL.secure_segment_threshold()
    range_list = [range(boundary_MRB[i],boundary_MRB[i+1]) for i in range(num_segments)]
    #generate all possible error patterns of mrb
    error_pattern_list = []
    erro_pattern_size_list = []
    for j in range(len(residual_path)):
        element = osd.error_pattern_gen(residual_path[j],range_list)
        error_pattern_list.append(element)
        erro_pattern_size_list.append(element.shape[0])
    acc_block_size = np.insert(np.cumsum(erro_pattern_size_list),0,0)  
    return error_pattern_list,acc_block_size       

def Testing_OSD(OSD_instance, snr, selected_ds, model_list, logistics):
    start_time = time.time()
    DIA_model, SWA_model = model_list
    _, decoding_matrix, log_filename = logistics
    
    # Configuration parameters
    config = {
        'list_length': GL.get_map('num_iterations') + 1,
        'code': GL.get_map('code_parameters'),
        'soft_margin': GL.get_map('soft_margin'),
        'intercept_length': GL.get_map('intercept_length'),
        'relax_factor': GL.get_map('relax_factor'),
        'block_size': GL.get_map('block_size'),
        'win_width': GL.get_map('win_width'),
        'DIA_indicator': GL.get_map('DIA_deployment'),
        'termination_threshold': GL.get_map('termination_threshold'),
        'sliding_strategy': GL.get_map('sliding_strategy')
    }
    
    # Initialize counters
    counters = {
        'fail': 0,
        'correct': 0,
        'windows': 0,
        'complexity': 0,
        'actual_size': 0,
        'cross_entropy_sum': [0.] * (config['list_length'] + 1),
        'ber_sum': [0] * (config['list_length'] + 1)
    }
    
    input_list = list(selected_ds.as_numpy_iterator())
    
    for i, input_data in enumerate(input_list):
        # Preprocess inputs based on DIA indicator
        if config['DIA_indicator']:
            squashed_inputs, labels, iterative_list = DIA_model.preprocessing_inputs(input_data)
            forged_output = tf.reshape(DIA_model(squashed_inputs), [-1, config['code'].n])
            forged_output += iterative_list[0]
        else:
            labels = input_data[1][0::config['list_length']]
            forged_output = input_data[0][0::config['list_length']]
        
        counters['actual_size'] += labels.shape[0]
        
        # Calculate metrics
        input_data_list = [input_data[0][j::config['list_length']] for j in range(config['list_length'])]
        input_data_list.append(forged_output)
        cross_entropy_list, ber_list = calculate_list_cross_entropy_ber(input_data_list, labels)
        
        # Update counters
        counters['cross_entropy_sum'] = [a + b for a, b in zip(cross_entropy_list, counters['cross_entropy_sum'])]
        counters['ber_sum'] = [a + b for a, b in zip(ber_list, counters['ber_sum'])]
        
        # OSD processing
        correct, fail, windows, complexity = OSD_instance.choose_decoding_strategy(
            SWA_model, input_data[0], forged_output, labels, decoding_matrix
        )
        counters.update({
            'correct': counters['correct'] + correct,
            'fail': counters['fail'] + fail,
            'windows': counters['windows'] + windows,
            'complexity': counters['complexity'] + complexity
        })
        
        # Periodic reporting
        if (i + 1) % 10 == 0:
            _report_progress(snr, counters, config, start_time)
        
        # Early termination
        if i == len(input_list) - 1 or counters['fail'] >= config['termination_threshold']:
            break
    
    # Final reporting
    return _final_report(snr, counters, config, start_time, log_filename)

def _get_display_parameters(config):
    """Helper to determine what parameters to display based on configuration"""
    if config.get('sliding_strategy', False) == False:
        return "under max_decoding"
    else:
        return f"under soft_margin:{config['soft_margin']}"

def _report_progress(snr, counters, config, start_time):
    """Helper function for periodic progress reporting."""
    avg_size = round(counters['complexity'] / counters['actual_size'], 4)
    wins_size = round(counters['windows'] / counters['actual_size'], 4)
    
    display_param = _get_display_parameters(config)
    
    print(f'\nFor {snr:.1f}dB intercept{config["intercept_length"]}-'
          f'relax_factor{config["relax_factor"]}-block{config["block_size"]}-'
          f'width{config["win_width"]} {display_param}:')
    print(f'--> S/F:{counters["correct"]} /{counters["fail"]} '
          f'Avr TEPs:{avg_size} Wins:{wins_size}')
    
    avg_loss = [x / counters['actual_size'] for x in counters['cross_entropy_sum']]
    avg_ber = [x / (counters['actual_size'] * config['code'].check_matrix_col) for x in counters['ber_sum']]
    
    print('avr CE per itr:\n' + ' '.join(f'{x:.3f}' for x in avg_loss))
    print('BER: ' + ' '.join(f'{x:.3f}' for x in avg_ber))
    print(f'Running time:{time.time() - start_time} seconds '
          f'with mean time {(time.time() - start_time)/counters["actual_size"]:.4f}!')


def _final_report(snr, counters, config, start_time, log_filename):
    """Helper method for final reporting and log writing."""
    display_param = _get_display_parameters(config)
    T2 = time.time()
    FER = round(counters['fail'] / counters['actual_size'], 5)
    avg_size = round(counters['complexity'] / counters['actual_size'], 4)
    wins_size = round(counters['windows'] / counters['actual_size'], 4)
    
    avg_loss = [x / counters['actual_size'] for x in counters['cross_entropy_sum']]
    avg_ber = [x / (counters['actual_size'] * config['code'].check_matrix_col) for x in counters['ber_sum']]
    
    # Prepare report content
    report_lines = [
        f'\nFor {snr:.1f}dB intercept:{config["intercept_length"]} '
        f'block:{config["block_size"]} {display_param}:',
        f'----> S:{counters["correct"]} F:{counters["fail"]}\n',
        f'FER:{FER}--> S/F:{counters["correct"]}/{counters["fail"]} '
        f'Avr TEPs:{avg_size} Wins:{wins_size}',
        'avr CE per itr:\n' + ' '.join(map('{:.3f}'.format, avg_loss)),
        'BER: ' + ' '.join(map('{:.3f}'.format, avg_ber)),
        f'Running time:{T2 - start_time} seconds '
        f'with mean time {(T2 - start_time)/counters["actual_size"]:.4f}!'
    ]
    
    # Print to console
    print('\n'.join(report_lines))
    
    # Write to log file
    with open(log_filename, 'a+') as f:
        f.write('\n'.join(report_lines) + '\n')
    
    return FER, log_filename

  
def evaluate_MRB_bit(updated_inputs,labels):
    inputs_abs = tf.abs(updated_inputs)
    code = GL.get_map('code_parameters')
    order_index = tf.argsort(inputs_abs,axis=-1,direction='ASCENDING')
    order_inputs = tf.gather(updated_inputs,order_index,batch_dims=1)
    order_inputs_hard = tf.where(order_inputs>0,0,1)
    order_labels = tf.cast(tf.gather(labels,order_index,batch_dims=1),tf.int32)
    cmp_result = tf.reduce_sum(tf.where(order_inputs_hard[:,-code.k:] == order_labels[:,-code.k:],0,1),axis=-1).numpy()
    Demo_result=Counter(cmp_result) 
    #print(Demo_result)
    return Demo_result
def dic_union(dicA,dicB):
    for key,value in dicB.items():
        if key in dicA:
            dicA[key] += value
        else:
            dicA[key] = value
    return dict(sorted(dicA.items(), key=lambda d:d[0])) 
    

def stat_pro_osd(inputs,labels):
    #initialize of mask
    code = GL.get_map('code_parameters')
    order_H_list,order_inputs,order_labels = check_matrix_reorder(inputs,labels)
    updated_MRB_list = []
    swap_len_list = []
    for i in range(inputs.shape[0]):
        # H assumed to be full row rank to obtain its systematic form
        tmp_H = np.copy(order_H_list[i])
        #reducing into row-echelon form and record column 
        #indices involved in pre-swapping
        M,record_col_index = code.gf2elim(tmp_H) 
        index_length = len(record_col_index)
        #update all swapping index
        index_order = np.array(range(code.check_matrix_col))
        for j in range(index_length):
            tmpa = record_col_index[j][0]
            tmpb = record_col_index[j][1]
            index_order[tmpa],index_order[tmpb] = index_order[tmpb],index_order[tmpa]   
        #udpated mrb indices
        updated_MRB = index_order[-code.k:]
        #print(Updated_MRB)
        updated_MRB_list.append(updated_MRB) 
        swap_indicator = np.where(updated_MRB>=code.check_matrix_col-code.k,0,1)
        swap_sum = sum(swap_indicator)
        swap_len_list.append(swap_sum)
    return updated_MRB_list,swap_len_list 

def check_matrix_reorder(inputs,labels):
    code = GL.get_map('code_parameters')
    expanded_H = tf.expand_dims(code.original_H,axis=0)
    #query the least reliable independent positions
    lri_p = tf.argsort(abs(inputs),axis=-1,direction='ASCENDING')
    order_inputs = tf.gather(inputs,lri_p,batch_dims=1)
    order_labels = tf.gather(labels,lri_p,batch_dims=1)
    batched_H = tf.transpose(tf.tile(expanded_H,[inputs.shape[0],1,1]),perm=[0,2,1])
    tmp_H_list = tf.gather(batched_H,lri_p,batch_dims=1)
    order_H_list = tf.transpose(tmp_H_list,perm=[0,2,1])
    return order_H_list,order_inputs,order_labels

def gf2elim(M):
      m,n = M.shape
      i=0
      j=0
      record_col_exchange_index = []
      while i < m and j < n:
          #print(M)
          # find value and index of largest element in remainder of column j
          if np.max(M[i:, j]):
              k = np.argmax(M[i:, j]) +i
        # swap rows
              #M[[k, i]] = M[[i, k]] this doesn't work with numba
              if k !=i:
                  temp = np.copy(M[k])
                  M[k] = M[i]
                  M[i] = temp              
          else:
              if not np.max(M[i, j:]):
                  M = np.delete(M,i,axis=0) #delete a all-zero row which is redundant
                  m = m-1  #update according info
                  continue
              else:
                  column_k = np.argmax(M[i, j:]) +j
                  temp = np.copy(M[:,column_k])
                  M[:,column_k] = M[:,j]
                  M[:,j] = temp
                  record_col_exchange_index.append((j,column_k))
      
          aijn = M[i, j:]
          col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected
          col[i] = 0 #avoid xoring pivot row with itself
          flip = np.outer(col, aijn)
          M[:, j:] = M[:, j:] ^ flip
          i += 1
          j +=1
      return M,record_col_exchange_index 
  
