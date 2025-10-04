# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 22:33:54 2024

@author: lgw
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:53:46 2022

@author: zidonghua_30
"""
#from hyperts.toolbox import from_3d_array_to_nested_df
import globalmap as GL
import tensorflow as tf
#from tensorflow.keras import  metrics
import nn_net as NN_struct
import predict_phase as Predict
#from collections import Counter,defaultdict,OrderedDict
import numpy as np
import pickle,re
import  os,sys
from itertools import combinations,chain
import itertools as it
#from typing import Any, Dict,Optional, Union
#from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
#from tensorflow.python.trackable.data_structures import NoDependency
#from sympy.utilities.iterables import multiset_permutations
# from itertools import combinations
# import math
# import ast
def check_matrix_reorder(original_inputs,inputs,labels):
    code = GL.get_map('code_parameters')
    expanded_H = tf.expand_dims(code.H,axis=0)
    #query the least reliable independent positions
    lri_p = tf.argsort(abs(inputs),axis=-1,direction='ASCENDING')
    order_inputs = tf.gather(inputs,lri_p,batch_dims=1)
    order_original_inputs = tf.gather(original_inputs,lri_p,batch_dims=1)
    order_labels = tf.gather(labels,lri_p,batch_dims=1)
    batched_H = tf.transpose(tf.tile(expanded_H,[inputs.shape[0],1,1]),perm=[0,2,1])
    tmp_H_list = tf.gather(batched_H,lri_p,batch_dims=1)
    order_H_list = tf.transpose(tmp_H_list,perm=[0,2,1])
    return order_H_list,order_original_inputs,order_inputs,order_labels

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

def query_convention_path(indicator_list,prefix_list,DIA):
    nn_type = 'benchmark'
    order_sum = GL.get_map('threshold_sum')+1
    if DIA: 
        for i,element in enumerate(indicator_list):
            if element == True:
                nn_type = prefix_list[i]
                break
    decoding_path = []
    for i in range(order_sum):
        for j1 in range(order_sum):
            for j2 in range(order_sum):                
                for j3 in range(order_sum):
                    if j1+j2+j3<=i:
                        decoding_path.append([j1,j2,j3])
    decoding_path,_ = tf.raw_ops.UniqueV2(x=decoding_path,axis=[0])
    return decoding_path,nn_type
#fixed scheduling for decoding path    
def query_decoding_path(indicator_list,prefix_list,DIA):
    snr_lo = round(GL.get_map('snr_lo'),2)
    snr_hi = round(GL.get_map('snr_hi'),2)
    snr_info = str(snr_lo)+"-"+str(snr_hi)
    #query decoding path
    nn_type = 'benchmark'
    if DIA: 
        for i,element in enumerate(indicator_list):
            if element == True:
                nn_type = prefix_list[i]
                break
    decoder_type = GL.get_map('selected_decoder_type')
    log_dir = './log/'+decoder_type+'/'+snr_info+'dB/'
    file_name = log_dir+"dist-error-pattern-"+nn_type+".pkl"
    with open(file_name, "rb") as fh:
        _ = pickle.load(fh)
        _ = pickle.load(fh)    
        _ = pickle.load(fh)
        _ = pickle.load(fh)    
        _ = pickle.load(fh)
        pattern_dict = pickle.load(fh)
    num_blocks = GL.get_map('num_blocks')
    tep_blocks,acc_block_size = partition_counter(pattern_dict, num_blocks) 
    return tep_blocks,acc_block_size,nn_type  

def string2digits(source_str):
    distilled_digits = re.findall(r"\w+",source_str)
    num_group = [int(element) for element in distilled_digits]
    return num_group

def error_pattern_gen(direction, range_list):
    code = GL.get_map('code_parameters')

    def function1_inline(i, value):
        if value:
            tmp = combinations(range_list[i], value)
        else:
            tmp = [[]]  # Use an empty list instead of -1
        return tmp

    itering_list = [function1_inline(i, value) for i, value in enumerate(direction)]

    # Adjusting the Cartesian product to properly handle the iterables
    combination_join = list(it.product(*itering_list))  # Unpack the list of iterables

    # Filtering and flattening the combinations
    filtered_comb = [list(filter(lambda x: x, combination_element)) for combination_element in combination_join]
    error_patterns = np.zeros(shape=[len(combination_join), code.k], dtype=int)

    def function2_inline(i, sequence):
        indices = list(chain.from_iterable(sequence))
        error_patterns[i, indices] = 1

    [function2_inline(i, sequence) for i, sequence in enumerate(filtered_comb)]

    return error_patterns   
                  
    
def filter_and_sort_counter(counter, threshold):
    num_blocks = GL.get_map('num_blocks')
    # Sort the filtered items by count in descending order (you can change this to ascending if needed)
    sorted_items = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    pattern_list = [string2digits(sorted_items[i][0]) for i in range(len(sorted_items))]
    order_pattern_matrix = tf.reshape(pattern_list,[-1,num_blocks])
    # Calculate the sum of elements in each row
    row_sums = np.sum(order_pattern_matrix, axis=1)
    filtered_patterns = order_pattern_matrix[row_sums <= threshold]
    error_pattern_list = generate_teps(filtered_patterns)
    size_list = [element.shape[0] for element in error_pattern_list]
    return error_pattern_list,size_list


def partition_counter(error_patterns_counter, num_blocks): 
    threshold_sum = GL.get_map('threshold_sum')
    error_pattern_list,size_list = filter_and_sort_counter(error_patterns_counter,threshold_sum)
    acc_block_size = np.insert(np.cumsum(size_list),0,0)
    return error_pattern_list,acc_block_size


def binary_sequences_within_hamming_weight(length, max_hamming_weight):
    def next_combination(x):
        u = x & -x
        v = u + x
        return v + (((v ^ x) // u) >> 2)
    
    all_sequences = []
    
    for hamming_weight in range(max_hamming_weight + 1):
        if hamming_weight == 0:
            # Add the all-zeros sequence
            all_sequences.append([0] * length)
            continue
        if hamming_weight > length:
            continue
        
        start = (1 << hamming_weight) - 1
        end = start << (length - hamming_weight)
        
        x = start
        while x <= end:
            binary_str = bin(x)[2:].zfill(length)
            all_sequences.append([int(bit) for bit in binary_str])
            x = next_combination(x)   
    # Convert the list of lists to a NumPy array
    TEP_tensor = np.array(all_sequences, dtype=int) 
    return TEP_tensor

def hamming_weight(row):
    """Compute the Hamming weight (number of 1s) of a binary row."""
    return np.sum(row)

def index_sum(row):
    """Calculate the sum of indices of non-zero bits, starting from the leftmost bit (index 0)."""
    return np.sum(np.nonzero(row)[0]) if np.any(row) else 0

def subtract_and_sort_matrices(A, B):
    # Convert rows to sets of tuples for fast set operations
    set_A = {tuple(row) for row in A}
    set_B = {tuple(row) for row in B}
    
    # Subtract sets to get rows in B but not in A
    set_diff = set_B - set_A
    
    # Convert the set difference back to a numpy array
    diff_matrix = np.array([list(row) for row in set_diff])
    
    # Sort the resulting matrix first by Hamming weight, then by index sum of non-zero bits
    sorted_matrix = sorted(diff_matrix, key=lambda row: (hamming_weight(row), index_sum(row)))  
    return np.array(sorted_matrix)


def filter_order_patterns(decoding_path):
    residual_path = []
    code = GL.get_map('code_parameters') 
    threshold_sum = GL.get_map('threshold_sum')
    nomial_path_length = GL.get_map('decoding_length')
    for order_pattern in decoding_path:
        if sum(order_pattern) <= threshold_sum:
           residual_path.append(order_pattern) 
    #for one case one order pattern, complement all other quailifed TEPs within given order-p
    stacked_path = np.stack(residual_path)
    if len(residual_path) < nomial_path_length:
        sequences = binary_sequences_within_hamming_weight(code.k,threshold_sum)
        updated_residual_path_matrix = subtract_and_sort_matrices(stacked_path,sequences)
        #reconstruct residual path
        residual_path_matrix = np.vstack((stacked_path,updated_residual_path_matrix))
        decoding_path = residual_path_matrix[:nomial_path_length]
    else:
        decoding_path = stacked_path[:nomial_path_length]
    return  decoding_path

def retore_saved_model(restore_info):
    print("Ready to restore a saved latest or designated model!")
    [ckpts_dir,ckpt_nm,restore_step] = restore_info 
    ckpt = tf.train.get_checkpoint_state(ckpts_dir)
    if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
        if restore_step == 'latest':
            ckpt_f = tf.train.latest_checkpoint(ckpts_dir)
            start_step = int(ckpt_f.split('-')[-1]) 
        else:
            ckpt_f = ckpts_dir+ckpt_nm+'-'+restore_step
            start_step = int(restore_step)  
    else:
        print('Error, no qualified file found')
    return start_step,ckpt_f

#fixed scheduling for decoding path    
def generate_teps(residual_path):
    num_seg = GL.get_map('num_blocks')
    #segmentation of MRB
    _,boundary_MRB= GL.secure_segment_threshold()
    range_list = [range(boundary_MRB[i],boundary_MRB[i+1]) for i in range(num_seg)]
    #generate all possible error patterns of mrb
    error_pattern_list = [error_pattern_gen(residual_path[j],range_list) for j in range(len(residual_path))]
    return error_pattern_list  
    
def query_samples(selected_ds,DIA_model,decoding_matrix,saved_file_name):
    code = GL.get_map('code_parameters')
    print('Selectd decoding path:\n',decoding_matrix)
    #acquire decoding     cnn = CRNN_DEF.conv_bitwise()
    DIA_indicator = GL.get_map('DIA_deployment')
    list_length = GL.get_map('num_iterations')+1
    try:
        num_counter = sum(1 for _ in selected_ds)
    except tf.errors.OutOfRangeError:
        pass  # Silence the warning     
    fail_sum = 0
    correct_sum = 0
    undetect_sum = 0
    # validating the effect by finding the consumed number of TEPs before hitting the authentic EP
    validate_size = 0
    record_list = []
    input_tuple_list = list(selected_ds.as_numpy_iterator())
    for i in range(num_counter):
        inputs = input_tuple_list[i]
    #for i in range(10):       
        if DIA_indicator:
            squashed_inputs,labels,inputs_list = DIA_model.preprocessing_inputs(inputs)
            updated_inputs = tf.reshape(DIA_model(squashed_inputs),[-1,code.n])+inputs_list[0]
            original_inputs = inputs_list[0]
        else:
            updated_inputs = inputs[0][0::list_length] 
            labels = inputs[1][0::list_length]
            original_inputs = inputs[0][0::list_length]
        validate_size += updated_inputs.shape[0]
        #preparing training samples
        records,cf_counter = query_teps_dist(original_inputs,updated_inputs,labels,decoding_matrix) 
        record_list.append(records)
        correct_sum += cf_counter[0]
        fail_sum += cf_counter[1]
        undetect_sum += cf_counter[2]
        if (i+1)%2 == 0: 
            print(correct_sum,fail_sum,undetect_sum)
    #save aquired data in disk
    saved_summary = (validate_size,correct_sum,fail_sum,undetect_sum)
    records_matrix = np.concatenate(record_list,axis=0)
    print(f'Correct:{correct_sum},Failed:{fail_sum},Undetected:{undetect_sum}')
    with open(saved_file_name, "wb") as fh:
        pickle.dump(saved_summary,fh)
        pickle.dump(records_matrix,fh)
    return saved_summary, records_matrix

def reformat_inputs(records_matrix):
    # Extract success class indicator (1 for positive cases, -1 for negative)
    suc_class_indicator = tf.cast(records_matrix[:, -1:] != -1, tf.float32)
    input_list = records_matrix[:, :-1]  
    # Calculate minimum values once
    min_value = tf.reduce_min(input_list, axis=-1, keepdims=True)
    window_width = GL.get_map('win_width')
    # Generate all windows in one operation using tf.extract_rolling_window (or similar)
    # For TensorFlow versions without extract_rolling_window, we'll keep the loop but make it more efficient
    windows = tf.stack([
        input_list[:, i:i+window_width] 
        for i in range(input_list.shape[1] - window_width + 1)
    ], axis=1)   
    # Vectorized operations for all windows
    # Reshape min_value to [199, 1, 1] for proper broadcasting
    min_value_reshaped = tf.reshape(min_value, [-1, 1, 1])
    window_mins = tf.reduce_min(windows, axis=2, keepdims=True)
    indicator_min = tf.cast(min_value_reshaped == window_mins, tf.float32)
    label_indicator = indicator_min * tf.reshape(suc_class_indicator,[-1,1,1]) 
    # Generate positions in correct shape
    positions = tf.tile(
        tf.range(input_list.shape[1] - window_width + 1, dtype=tf.float32)[None, :, None],
        [tf.shape(records_matrix)[0], 1, 1]
    )      
    # Final concatenation
    windows = tf.cast(windows, tf.float32)
    combined = tf.concat([windows, positions, label_indicator], axis=2)    
    # Reshape and sort
    windows_matrix = tf.reshape(combined, [-1, window_width + 2])
    volume_data = tf.sort(windows_matrix[:, :-2], axis=1, direction='ASCENDING')  
    return (
        tf.concat([volume_data, windows_matrix[:, -2:-1]], axis=1),
        windows_matrix[:, -1]
    )


def query_teps_dist(original_inputs,updated_inputs,labels,decoding_matrix):
    code = GL.get_map('code_parameters')
    block_size = GL.get_map('block_size')
    intercept_length = GL.get_map('intercept_length')
    block_length = intercept_length//block_size
    #OSD processing
    #first arrangment of H
    swap_info = check_matrix_reorder(original_inputs,updated_inputs,labels)
    order_H_list,order_original_inputs_list,order_updated_inputs_list,order_labels_list = swap_info
    record_list = []
    success_counter = 0
    fail_counter = 0
    undetect_counter = 0
    actual_size = updated_inputs.shape[0]
    for i in range(actual_size):
        # H assumed to be full row rank to obtain its systematic form
        tmp_H = np.copy(order_H_list[i])
        #reducing into row-echelon form and record column 
        #indices involved in pre-swapping
        #second arrangement of H
        reduce_H,record_col_index = gf2elim(tmp_H) 
        index_length = len(record_col_index)
        #update all swapping index
        index_order = np.array(range(code.n))
        for j in range(index_length):
            tmpa = record_col_index[j][0]
            tmpb = record_col_index[j][1]
            index_order[tmpa],index_order[tmpb] = index_order[tmpb],index_order[tmpa]   
        #udpated mrb indices
        updated_MRB = index_order[-code.k:]
        mrb_swapping_index = tf.argsort(updated_MRB,axis=0,direction='ASCENDING')
        mrb_order = tf.sort(updated_MRB,axis=0,direction='ASCENDING')
        updated_index_order = tf.concat([index_order[:(code.n-code.k)],mrb_order],axis=0)
        #third arrangement of H
        final_H2 = tf.gather(reduce_H[:,-code.k:],mrb_swapping_index,axis=1)   
        renewed_original_inputs = tf.gather(order_original_inputs_list[i],updated_index_order)
        renewed_updated_inputs = tf.gather(order_updated_inputs_list[i],updated_index_order)    
        renewed_labels  = tf.cast(tf.gather(order_labels_list[i],updated_index_order),dtype=tf.int32)
        # setting anchoring point    
        renewed_updated_hard = tf.where(renewed_updated_inputs>0,0,1)
        baseline_mrb = tf.reshape(renewed_updated_hard[-code.k:],[1,-1]) 
        #estimations of codeword candidate
        error_pattern_matrix = decoding_matrix
        estimated_mrb_matrix = tf.transpose((tf.cast(error_pattern_matrix,tf.int32)+baseline_mrb)%2)
        estimated_lrb_matrix = tf.matmul(tf.cast(final_H2,tf.int32),estimated_mrb_matrix)%2
        codeword_candidate_matrix = tf.transpose(tf.concat([estimated_lrb_matrix,estimated_mrb_matrix],axis=0))
        expand_codeword_candidate_matrix = tf.concat([codeword_candidate_matrix,tf.reshape(renewed_labels,[1,-1])],axis=0)
        order_hard = tf.where(renewed_original_inputs>0,0,1)  
        discrepancy_matrix = tf.cast((expand_codeword_candidate_matrix + order_hard)%2,dtype=tf.float32)
        soft_discrepancy_sum = tf.reduce_sum(discrepancy_matrix*abs(renewed_original_inputs),axis=-1)
        temp_min_index = tf.argmin(soft_discrepancy_sum[:-1])
        locate_phase = -1
        if soft_discrepancy_sum[temp_min_index] < soft_discrepancy_sum[-1]:
            #tf.print(soft_discrepancy_sum[temp_min_index],soft_discrepancy_sum[-1])
            undetect_counter += 1
            continue
        if soft_discrepancy_sum[temp_min_index] == soft_discrepancy_sum[-1]:
            locate_phase = 1
        block_min_sum = [np.min(soft_discrepancy_sum[k*block_size:(k+1)*block_size]) for k in range(block_length)]
        record_unit = np.append(block_min_sum,locate_phase)
        record_list.append(record_unit)
        success_decoding_indicator = (locate_phase != -1)
        if success_decoding_indicator:
            success_counter += 1
        else:
            fail_counter += 1   
    cf_counter = (success_counter,fail_counter,undetect_counter)
    record_matrix = np.vstack(record_list)
    return record_matrix,cf_counter
