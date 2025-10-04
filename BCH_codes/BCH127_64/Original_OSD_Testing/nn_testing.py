# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:53:46 2022

@author: zidonghua_30
"""
#from hyperts.toolbox import from_3d_array_to_nested_df
import globalmap as GL
import tensorflow as tf
import ordered_statistics_decoding as OSD_mod
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from collections import Counter
import numpy as  np
import time,os
from itertools import combinations
import matplotlib.pyplot as plt

def plot_curve(data_lists, label_list, snr, x_vertical_line,vline_height):
    """Plot curves with optional vertical reference line.
    
    Args:
        data_lists: List of data series to plot
        label_list: List of legend labels for each series
        snr: Signal-to-noise ratio value (for title/filename)
        x_vertical_line: x-position for vertical reference line (optional)
    """
    # Get code parameters from global configuration
    code = GL.get_map('code_parameters')
    
    # Set global font settings (supporting Chinese characters)
    chinese_fonts = ['Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'STXihei', 'STKaiti']
    plt.rcParams['font.sans-serif'] = chinese_fonts  # Specify Chinese-compatible fonts
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issues
    
    # Create figure with specified size
    plt.figure(figsize=(10,7))
    
    # Plot each data series
    x_values = range(1, len(data_lists[0]) + 1)
    for i, data_list in enumerate(data_lists):
        plt.plot(x_values, data_list, label=label_list[i])
    
    # Add vertical dotted reference line if position specified
    ymin = min([data_list[i] for i in range(len(data_list))])
    ymax = max([data_list[i] for i in range(len(data_list))])
    plt.ylim(ymin,ymax)
    plt.vlines(x=x_vertical_line,
               ymin=ymin,
               ymax=ymin + vline_height * (ymax - ymin),
               colors='magenta',
               linestyles=':',
               linewidth=2)
        # Optional: Add text annotation at the top of the line
        # plt.text(
        #     x_vertical_line, plt.ylim()[1], 
        #     'Reference', 
        #     ha='center', va='bottom', 
        #     rotation=90, 
        #     backgroundcolor='white'
        # )
    
    # Configure legend
    plt.legend(
        fontsize=16, 
        loc='upper left'  # Automatic optimal positioning
    )
    
    # Set axis labels
    plt.xlabel('Indices of ordered bits', fontsize=18)
    plt.ylabel('Drifting delta', fontsize=18)
    
    # Set tick label sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Enable grid with custom style
    plt.grid(
        True, 
        linestyle='--',  # Dashed lines
        alpha=0.6  # Semi-transparent
    )
    xmin = 0
    xmax = code.n
    plt.xlim(xmin,xmax)
    # Create output directory if needed
    gen_data_dir = './figs/'
    if not os.path.exists(gen_data_dir):
        os.makedirs(gen_data_dir)
    
    # Save high-quality image
    plt.savefig(
        f'{gen_data_dir}mean_reliability_{snr}dB_bch_{code.n}_{code.k}_code.jpg',
        dpi=1000,  # High resolution
        bbox_inches='tight'  # Prevent cropping
    )
    
    # Display plot
    plt.show()

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

def generate_binary_sequences(n, p):
    sequences = []   
    # Generate sequences for Hamming weights from 0 to p
    for weight in range(p + 1):
        for positions in combinations(range(n), weight):
            seq = np.zeros(n, dtype=int)
            seq[list(positions)] = 1
            sequences.append(seq)  
    return np.array(sequences)      

def Testing_pro_bits(snr,selected_ds):
    start_time = time.process_time()
    code = GL.get_map('code_parameters')
    maximum_order = GL.get_map('maximum_order')
    osd_instance = OSD_mod.osd(code)
    n_iteration = GL.get_map('num_iterations')
    list_length = n_iteration+1
    input_list = list(selected_ds.as_numpy_iterator())
    num_counter = len(input_list) 
    actual_size = 0 
    sorted_sum = np.zeros([1,code.n],dtype=float)
    swapped_sum = np.zeros([1,code.n],dtype=float)
    sorted_sum = np.zeros([1,code.n],dtype=float)
    swapped_sum = np.zeros([1,code.n],dtype=float)
    sum_list = [sorted_sum,swapped_sum]
    for i in range(num_counter): 
        inputs = input_list[i][0][::list_length]
        labels = input_list[i][1][::list_length]  
        actual_size += labels.shape[0]
        sum_list = osd_instance.convention_osd_preprocess(inputs, labels,sum_list)
        if (i+1)%10 == 0:
            pass
            #average_sorted_mean = tf.reduce_sum(sum_list[0],axis=0)/actual_size
            #print(f'\nFor {snr:.1f}dB maximum_order:{maximum_order}')
            #print(f'--> mean:{average_sorted_mean}')      
            #T2 =time.process_time()
            #print(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/actual_size:.4f}!')
    average_sorted_mean = tf.reduce_sum(sum_list[0],axis=0)/actual_size
    average_swapped_mean = tf.reduce_sum(sum_list[1],axis=0)/actual_size
    mean_list = [average_sorted_mean,average_swapped_mean]
    return mean_list
    
def Testing_OSD(snr,selected_ds):
    start_time = time.process_time()
    code = GL.get_map('code_parameters')
    maximum_order = GL.get_map('maximum_order')
    osd_instance = OSD_mod.osd(code) 
    teps_matrix = generate_binary_sequences(code.k,maximum_order)
    logdir = './log/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log_filename = logdir+'OSD-'+str(maximum_order)+'-original_osd.txt' 
    
    input_list = list(selected_ds.as_numpy_iterator())
    num_counter = len(input_list)      
    
    fail_sum = 0
    correct_sum = 0
    complexity_sum = 0
    actual_size = 0    
    sorted_sum = np.zeros([1,code.n],dtype=float)
    swapped_sum = np.zeros([1,code.n],dtype=float)
    sum_list = [sorted_sum,swapped_sum]
    for i in range(num_counter): 
        inputs = input_list[i][0]
        labels = input_list[i][1]      
        actual_size += labels.shape[0]
        #OSD processing
        correct_counter,fail_counter,complexity_size,sum_list = osd_instance.convention_osd(inputs,labels,teps_matrix,sum_list)
        correct_sum += correct_counter
        fail_sum += fail_counter  
        complexity_sum += complexity_size
        if (i+1)%10 == 0:
            average_size = round(complexity_sum/actual_size,4)
            average_sorted_mean = tf.reduce_sum(sum_list[0],axis=0)/actual_size
            print(f'\nFor {snr:.1f}dB maximum_order:{maximum_order}')
            print(f'--> S/F:{correct_sum} /{fail_sum} Avr TEPs:{average_size} mean:{average_sorted_mean}')      
            T2 =time.process_time()
            print(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/actual_size:.4f}!')
        if i == num_counter-1 or fail_sum >= GL.get_map('termination_threshold'):
            break
    T2 =time.process_time()
    FER = round(fail_sum/actual_size,5)  
    average_size = round(complexity_sum/actual_size,4)
    average_sorted_mean = tf.reduce_sum(sum_list[0],axis=0)/actual_size
    average_swapped_mean = tf.reduce_sum(sum_list[1],axis=0)/actual_size
    mean_list = [average_sorted_mean,average_swapped_mean]
    print('\nFor %.1fdB (maximum_order:%d) '%(snr,maximum_order)+':\n')
    print('----> S:'+str(correct_sum)+' F:'+str(fail_sum)+'\n')          
    print(f'FER:{FER}--> S/F:{correct_sum} /{fail_sum} Avr TEPs:{average_size}')
    print(f'sorted_mean:{average_sorted_mean} swapped_mean:{average_swapped_mean}')
    print(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/actual_size:.4f}!')
    with open(log_filename,'a+') as f:
        f.write(f'For {snr:.1f}dB maximum_order:{maximum_order}\n')
        f.write('----> S:'+str(correct_sum)+' F:'+str(fail_sum)+'\n')         
        f.write(f'FER:{FER}--> S/F:{correct_sum} /{fail_sum} Avr TEPs:{average_size}')
        f.write(f' mean:{average_sorted_mean} swapped_mean:{average_swapped_mean} \n')
        f.write(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/actual_size:.4f}!\n')
    return FER,log_filename,mean_list

  
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
  
