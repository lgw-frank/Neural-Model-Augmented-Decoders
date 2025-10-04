# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:53:46 2022

@author: zidonghua_30
"""
#from hyperts.toolbox import from_3d_array_to_nested_df
import globalmap as GL
import tensorflow as tf
from tensorflow.keras import  metrics
import nn_net as NN_struct
from collections import Counter,defaultdict,OrderedDict
import numpy as np
import pickle,re
import  os,sys
from typing import Any, Dict,Optional, Union
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
#from tensorflow.python.trackable.data_structures import NoDependency
#from sympy.utilities.iterables import multiset_permutations
from itertools import combinations
import math
import ast
from scipy.special import comb
import predict_phase as Predict


def check_matrix_reorder(inputs,labels):
    code = GL.get_map('code_parameters')
    expanded_H = tf.expand_dims(code.H,axis=0)
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
def identify_mrb(order_H_list):
    #initialize of mask
    code = GL.get_map('code_parameters')
    updated_index_list = []
    updated_M_list = []
    swap_len_list = []
    swap_lrb_position_list = []
    for i in range(order_H_list.shape[0]):
        # H assumed to be full row rank to obtain its systematic form
        tmp_H = np.copy(order_H_list[i])
        #reducing into row-echelon form and record column 
        #indices involved in pre-swapping
        swapped_H,record_col_index = gf2elim(tmp_H) 
        index_length = len(record_col_index)
        #update all swapping index
        index_order = np.array(range(code.check_matrix_column))
        for j in range(index_length):
            tmpa = record_col_index[j][0]
            tmpb = record_col_index[j][1]
            index_order[tmpa],index_order[tmpb] =  index_order[tmpb],index_order[tmpa]   
        #udpated mrb indices
        updated_MRB = index_order[-code.k:]
        updated_LRB = index_order[:code.k]
        mrb_swapping_index = tf.argsort(updated_MRB,axis=0,direction='ASCENDING')
        mrb_order = tf.sort(updated_MRB,axis=0,direction='ASCENDING')
        updated_index_order = tf.concat([index_order[:(code.check_matrix_column-code.k)],mrb_order],axis=0)
        updated_M = tf.gather(swapped_H[:,-code.k:],mrb_swapping_index,axis=1)    
        updated_index_list.append(updated_index_order)  
        updated_M_list.append(updated_M)
        swap_indicator = np.where(updated_MRB>=code.check_matrix_column-code.k,0,1)
        # focus of rear part of positions in LRB plus those positions swapped from MRB      
        discrinating_size = GL.get_map('sense_region')
        jump_point = (code.check_matrix_column-code.k)-discrinating_size
        swap_lrb_indicator = np.where(updated_LRB>=jump_point,1,0)
        swap_sum = sum(swap_indicator)
        swap_len_list.append(swap_sum)
        swap_lrb_position_list.append(swap_lrb_indicator)
    return updated_index_list,updated_M_list,swap_len_list,swap_lrb_position_list  

# def place_ones(size,order):
#   total_list = []
#   for i in range(order+1):
#       tmp_point = size*[0,]
#       if i:
#           tmp_point[:i] = i*(1,)
#       perm_a = list(multiset_permutations(tmp_point))
#       total_list += perm_a  
#   return np.reshape(total_list,[-1,size])

def generate_binary_arrays(length, hamming_weight):
    if hamming_weight < 0 or hamming_weight > length:
        return []
    all_arrays = []
    # Generate all combinations of indices with the specified hamming weight
    index_combinations = list(combinations(range(length), hamming_weight))
    for indices in index_combinations:
        binary_array = np.zeros(length, dtype=int)
        binary_array[list(indices)] = 1
        all_arrays.append((np.sum(np.nonzero(binary_array)[0]), tf.constant(binary_array, dtype=tf.int32)))
    # Sort by the sum of indices of nonzero elements in ascending order
    sorted_arrays = sorted(all_arrays, key=lambda x: x[0]) 
    return [arr[1] for arr in sorted_arrays]

def group_binary_arrays(arrays, batch_block_size):
    num_groups = math.ceil(len(arrays)/batch_block_size)
    grouped_arrays = [[] for _ in range(num_groups)]  
    for i, arr in enumerate(arrays):
        group_index = i // batch_block_size
        grouped_arrays[group_index].append(arr) 
    return grouped_arrays,num_groups

def analyze_distribution(original_counter, cluster_sequence,cluster_group):
    # Convert string keys to lists of integers
    new_list = [(list(map(int, key.split())), value) for key, value in original_counter.items()]
    # Get items in descending order of values
    items_descending = sorted(new_list, key=lambda x: x[1], reverse=True)
    group_occurrences = [0] * len(cluster_sequence)  
    # Generate a list of cumulative sums
    cumulative_sums = [0]+[sum(cluster_group[:i+1]) for i in range(len(cluster_group))]
    for item_tuple in items_descending:
        #print(item_tuple)
        #print(group_occurrences)
        indices_list = item_tuple[0]
        if indices_list == [-1]:
            start_index = cumulative_sums[0]
            end_index = cumulative_sums[1]
        else:
            shift = len(indices_list)
            if shift >= len(cumulative_sums)-1:
                continue
            start_index = cumulative_sums[shift]
            end_index = cumulative_sums[shift+1]
        increment = item_tuple[1]
        jump_indicator = False
        for i, group in enumerate(cluster_sequence[start_index:end_index]):
            for binary_array in group:
                if indices_list == [-1] or tf.reduce_all(tf.gather(binary_array, indices_list)!=0):
                    group_occurrences[start_index+i] += increment
                    jump_indicator = True
                    break
            if jump_indicator == True:
                break
    # Print the occurrences for each group
    print(group_occurrences)
    #for i, occurrences in enumerate(group_occurrences):
        #print(f"Group {i + 1}: {occurrences} occurrences")
    return group_occurrences

def execute_osd(inputs,labels):
    code = GL.get_map('code_parameters')
    order_list= check_matrix_reorder(inputs,labels)
    order_H_list,order_inputs,order_labels = order_list
    updated_index_list,updated_M_list,swap_len_list,swap_lrb_position_list = identify_mrb(order_H_list)
    input_size = inputs.shape[0]
    order_input_list = [tf.gather(order_inputs[i],updated_index_list[i]) for i in range(input_size)]
    order_label_list = [tf.cast(tf.gather(order_labels[i],updated_index_list[i]),dtype=tf.int32) for i in range(input_size)]
    order_input_matrix = tf.reshape(order_input_list,[-1,code.check_matrix_column])
    order_label_matrix = tf.reshape(order_label_list,[-1,code.check_matrix_column])
    order_hard_matrix = tf.where(order_input_matrix>0,0,1)
    #record error bit positions of MRB
    bit_records = tf.where(order_hard_matrix[:,-code.k:]==order_label_matrix[:,-code.k:],0,1)
    # Find the indices of nonzero elements
    nonzero_indices = tf.where(bit_records != 0)
    # Group the indices by row using a dictionary
    count_dict = {}
    for row_index, col_index in nonzero_indices.numpy():
        if row_index in count_dict:
            count_dict[row_index].append(col_index)
        else:
            count_dict[row_index] = [col_index]
    # Convert the lists of column indices to strings
    count_dict_str = {row_index: ' '.join(map(str, col_indices)) for row_index, col_indices in count_dict.items()}
    # Use Counter to count occurrences
    Demo_result = Counter(count_dict_str.values().tolist())
    # Manually set count for a specific item of all zero TEP
    item_to_add = '-1'
    count_to_set = inputs.shape[0]-sum(Demo_result.values())
    Demo_result[item_to_add] = count_to_set
    print(Demo_result)
    return Demo_result
    
        
def try_count_flops(model: Union[tf.Module, tf.keras.Model],
                    inputs_kwargs: Optional[Dict[str, Any]] = None,
                    output_path: Optional[str] = None):
    """Counts and returns model FLOPs.
  Args:
    model: A model instance.
    inputs_kwargs: An optional dictionary of argument pairs specifying inputs'
      shape specifications to getting corresponding concrete function.
    output_path: A file path to write the profiling results to.
  Returns:
    The model's FLOPs.
  """
    if hasattr(model, 'inputs'):
        try:
            # Get input shape and set batch size to 1.
            if model.inputs:
                inputs = [
                    tf.TensorSpec([1] + input.shape[1:], input.dtype)
                    for input in model.inputs
                ]
                concrete_func = tf.function(model).get_concrete_function(inputs)
            # If model.inputs is invalid, try to use the input to get concrete
            # function for model.call (subclass model).
            else:
                concrete_func = tf.function(model.call).get_concrete_function(
                    **inputs_kwargs)
            frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)

            # Calculate FLOPs.
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            if output_path is not None:
                opts['output'] = f'file:outfile={output_path}'
            else:
                opts['output'] = 'none'
            flops = tf.compat.v1.profiler.profile(
                graph=frozen_func.graph, run_meta=run_meta, options=opts)
            return flops.total_float_ops
        except Exception as e:  # pylint: disable=broad-except
            print('Failed to count model FLOPs with error %s, because the build() '
                 'methods in keras layers were not called. This is probably because '
                 'the model was not feed any input, e.g., the max train step already '
                 'reached before this run.', e)
            return None
    return None
def print_flops(model):
    flops = try_count_flops(model)
    print(flops/1e3,"K Flops")
    return None
def print_model_summary(model):
    # Create an instance of the model
    #model = ResNet(num_blocks=3, filters=64, kernel_size=3)
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Print model summary
    list_length = GL.get_map('num_iterations')+1
    stripe = GL.get_map('stripe')
    model.build(input_shape=(None, list_length,stripe))  # Assuming input shape is (batch_size, sequence_length, input_dim)
    model.summary()
    return None


def calculate_loss(updated_inputs,labels):
    labels = tf.cast(labels,tf.float32)  
    #measure discprepancy via cross entropy metric which acts as the loss definition for deep learning per batch  
    # weight_matrix = tf.where(tf.sign(updated_inputs)==1-2*labels,1.,5. )
    # loss_matrix = tf.nn.sigmoid_cross_entropy_with_logits(logits=-updated_inputs, labels=labels)*weight_matrix
    # loss = tf.reduce_sum(loss_matrix)
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-updated_inputs, labels=labels))
    #tf.print(loss_list)
    return  loss

def evaluate_MRB_bit(updated_inputs,labels):
    inputs_abs = tf.abs(updated_inputs)
    code = GL.get_map('code_parameters')
    order_index = tf.argsort(inputs_abs,axis=-1,direction='ASCENDING')
    order_inputs = tf.gather(updated_inputs,order_index,batch_dims=1)
    order_inputs_hard = tf.where(order_inputs>0,0,1)
    order_labels = tf.cast(tf.gather(labels,order_index,batch_dims=1),tf.int32)
    cmp_result = tf.reduce_sum(tf.where(order_inputs_hard[:,-code.k:] == order_labels[:,-code.k:],0,1),axis=-1).numpy()
    Demo_result=Counter(cmp_result.tolist()) 
    #print(Demo_result)
    return Demo_result

def dic_union(dicA,dicB):
    for key,value in dicB.items():
        if key in dicA:
            dicA[key] += value
        else:
            dicA[key] = value
    return dict(sorted(dicA.items(), key=lambda d:d[0])) 
      
def Training_dia_model(DIA_model,base_ds,restore_info,original_NMS_indicator=False):
    print_interval = GL.get_map('print_interval')
    record_interval = GL.get_map('record_interval')
    dia_termination_step = GL.get_map('dia_termination_step')
    
    #query of size of input feedings
    selected_ds = GL.build_training_dataset(original_NMS_indicator)
    iterator = iter(selected_ds) 
    #unpack related info for restoraging
    DIA_model,optimizer,logger_info,batch_index = Predict.NN_gen(DIA_model,restore_info,DIA_model.actual_length,input_dim=3)
    [summary_writer,manager_current] = logger_info
    loss_meter = metrics.Mean()
    if GL.get_map('nn_train'):
        while True:
            if batch_index >= dia_termination_step:
                break
            try:
                inputs = next(iterator)
            except StopIteration:
                # In case dataset is finite and exhausted
                print("⚠️ Dataset exhausted. Consider adding `.repeat()` in dataset pipeline.")
                break
            squashed_inputs,labels,_ = DIA_model.preprocessing_inputs(inputs)
            with tf.GradientTape() as tape:
                refined_inputs = tf.reshape(DIA_model(squashed_inputs),[-1,DIA_model.n_dims])
                loss = calculate_loss(refined_inputs,labels)
                loss_meter.update_state(loss) 
            grads = tape.gradient(loss, DIA_model.trainable_variables)
            grads_and_vars = [(tf.clip_by_norm(grad, 5e2), var)
                              for grad, var in zip(grads, DIA_model.trainable_variables)
                              if grad is not None]
            optimizer.apply_gradients(grads_and_vars)    
            batch_index += 1
            if batch_index % print_interval == 0:   
                with summary_writer.as_default():
                    tf.summary.scalar("loss", loss, step=batch_index)
                tf.print(f'Step:{batch_index} Loss:{loss:.3f}')
                #_ = evaluate_MRB_bit(inputs,labels)
                #_ = evaluate_MRB_bit(refined_inputs,labels)                                                               
            if batch_index % record_interval == 0:
                manager_current.save(checkpoint_number=batch_index)                   
            loss_meter.reset_state()  
    print("\nFinal selected weights:")
    weights = DIA_model.get_weights()
    rounded_weights = Predict.format_weights(weights, decimals=2)
    print(rounded_weights)  # Removes dtype and converts to native Python floats
    return DIA_model


def Testing_dia_model(DIA_model,base_ds): 
    
    list_length = GL.get_map('num_iterations')+1
    snr_lo = round(GL.get_map('snr_lo'),2)
    snr_hi = round(GL.get_map('snr_hi'),2)
    snr_info = f'{snr_lo}-{snr_hi}dB/'

    block_size = GL.get_map('block_size')
    intercept_length = GL.get_map('intercept_length')
    relax_factor = GL.get_map('relax_factor')
    option_without_dia =  GL.get_map('option_without_dia')
    option_with_dia =  GL.get_map('option_with_dia')
    
    #verifying trained pars from start to end    
    dic_sum_initial = {}
    dic_sum = {}
    pattern_cnt_initial = Counter()
    pattern_cnt = Counter()
    actual_size = 0

    if not intercept_length%block_size:
        decoding_matrix_list = Predict.choose_ordered_teps(option_without_dia,option_with_dia)
    else:      
        print('Block size has to be dividable by intercept_length')
        sys.exit(-1)     
    try:
        num_counter = sum(1 for _ in base_ds)
    except tf.errors.OutOfRangeError:
        pass  # Silence the warning
    input_list  = list(base_ds.as_numpy_iterator())      
    for i in range(num_counter):
        labels = input_list[i][1][0::list_length]
        squashed_inputs,_,inputs = DIA_model.preprocessing_inputs(input_list[i])       
        updated_inputs = tf.reshape(DIA_model(squashed_inputs),[-1,DIA_model.n_dims])
        updated_inputs += inputs[0]

        actual_size += labels.shape[0]
        #query pattern distribution
        cmp_results = evaluate_MRB_bit(inputs[0],labels)
        dic_sum_initial = dic_union(dic_sum_initial,cmp_results)     
        pattern_cnt_initial,_ = generate_pattern_dist(inputs[0],labels,pattern_cnt_initial,decoding_matrix_list[0]) 
        
        cmp_results = evaluate_MRB_bit(updated_inputs,labels)
        dic_sum = dic_union(dic_sum,cmp_results)          
        pattern_cnt,_ = generate_pattern_dist(updated_inputs,labels,pattern_cnt,decoding_matrix_list[1]) 
        
        if (i+1)%100 == 0:
            # Convert keys to int and regenerate the Counter
            clean_pattern_cnt_initial = Counter({int(k): v for k, v in pattern_cnt_initial.items()})
            clean_pattern_cnt = Counter({int(k): v for k, v in pattern_cnt.items()})
            print(dic_sum_initial) 
            print(dic_sum) 
            print(clean_pattern_cnt_initial) 
            print(clean_pattern_cnt) 
                      
    print(f'Total counts:{actual_size}')
    print(f'Summary for initial and aggregated inputs:\n{dic_sum_initial}\n{dic_sum}')
    # # Get items sorted by counts in descending order
    sorted_items_initial = pattern_cnt_initial.most_common()
    sorted_items = pattern_cnt.most_common()
    # # Print the items in descending order of counts
    if GL.get_map('extended_input'):
        suffix = '-extended'
    else:
        suffix = ''
    print(f'\nSummary of ALMT(option:{option_without_dia} no DIA) decoding_pth:')
    for key, value in sorted_items_initial:
         print(f'{key}: {value}',end=' ')
    print(f'\nSummary of ALMT(option:{option_with_dia},DIA) decoding_pth:')
    for key, value in sorted_items:
         print(f'{key}: {value}',end=' ')
    #save on disk files
    saved_data_dir = f'./data/DIA_model/{snr_info}'
    if not os.path.exists(saved_data_dir):
        os.makedirs(saved_data_dir)  

    file_name = saved_data_dir+f"teps_dist-intercept{intercept_length}-relax{relax_factor}-block_size{block_size}{suffix}.pkl"
    with open(file_name, "wb") as fh:
        pickle.dump(actual_size,fh)
        pickle.dump(dic_sum_initial,fh)        
        pickle.dump(dic_sum,fh)
        pickle.dump(pattern_cnt_initial,fh)          
        pickle.dump(pattern_cnt,fh)     

def retore_saved_model(restore_ckpts_dir,restore_step,ckpt_nm):
    print("Ready to restore a saved latest or designated model!")
    ckpt = tf.train.get_checkpoint_state(restore_ckpts_dir)
    if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
      if restore_step == 'latest':
        ckpt_f = tf.train.latest_checkpoint(restore_ckpts_dir)
        start_step = int(ckpt_f.split('-')[-1]) 
      else:
        ckpt_f = restore_ckpts_dir+ckpt_nm+'-'+restore_step
        start_step = int(restore_step)
      print('Loading wgt file: '+ ckpt_f)   
    else:
      print('Error, no qualified file found')
    return start_step,ckpt_f
    
def generate_pattern_dist(inputs,labels,pattern_cnt,decoding_matrix):
    Updated_MRB_list,swap_len_list = stat_pro_osd(inputs,labels)    
    pattern_cnt = evaluate_MRB_pattern(inputs,labels,Updated_MRB_list,swap_len_list,pattern_cnt,decoding_matrix) 
    return pattern_cnt,swap_len_list


def evaluate_MRB_pattern(inputs,labels, updated_mrb_list,swap_len_list,pattern_cnt,decoding_matrix):
    block_size = GL.get_map('block_size')
    inputs_abs = tf.abs(inputs)
    order_index = tf.argsort(inputs_abs,axis=-1,direction='ASCENDING')
    order_inputs = tf.gather(inputs,order_index,batch_dims=1)
    order_labels = tf.cast(tf.gather(labels,order_index,batch_dims=1),tf.int32) 
    updated_mrb_list = [np.array(list(element)) for element in updated_mrb_list]
    reorder_index = tf.sort(tf.reshape(updated_mrb_list,[inputs.shape[0],-1]),axis=-1,direction='ASCENDING')
    reorder_inputs = tf.gather(order_inputs,reorder_index,batch_dims=1)
    reorder_inputs_hard = tf.where(reorder_inputs>0,0,1)
    reorder_labels = tf.cast(tf.gather(order_labels,reorder_index,batch_dims=1),tf.int32)
    #difference matrix
    diff_matrix = np.where(reorder_inputs_hard == reorder_labels,0,1)
    indices = find_row_indices(diff_matrix,decoding_matrix)
    for index in indices:
        located_pocket = index//block_size
        pattern_cnt[located_pocket] +=1  
    return pattern_cnt

def find_row_indices(A, B):
    """
    Finds the index of each row of A in B. Returns -1 if not found. 
    Args:
        A (np.ndarray): Query matrix (shape: (m, d))
        B (np.ndarray): Reference matrix (shape: (n, d)) 
    Returns:
        np.ndarray: Indices of A's rows in B (shape: (m,))
    """
    # Convert rows to tuples for hashing
    B_rows = {tuple(row): idx for idx, row in enumerate(B)}
    # Find indices using dictionary lookup (O(1) per query)
    indices = np.array([B_rows.get(tuple(row), -1) for row in A])   
    return indices

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
        index_order = np.array(range(code.n))
        for j in range(index_length):
            tmpa = record_col_index[j][0]
            tmpb = record_col_index[j][1]
            index_order[tmpa],index_order[tmpb] = index_order[tmpb],index_order[tmpa]   
        #udpated mrb indices
        updated_MRB = index_order[-code.k:]
        #print(Updated_MRB)
        updated_MRB_list.append(updated_MRB) 
        swap_indicator = np.where(updated_MRB>=code.n-code.k,0,1)
        swap_sum = sum(swap_indicator)
        swap_len_list.append(swap_sum)
    return updated_MRB_list,swap_len_list 

