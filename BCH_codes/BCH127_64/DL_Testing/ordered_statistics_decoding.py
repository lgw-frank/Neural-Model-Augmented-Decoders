# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:34:01 2023

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import globalmap as GL
from itertools import combinations,chain
import itertools as it
#from scipy.special import comb
#from sympy.utilities.iterables import multiset_permutations

# input operation:1)magnitude swapping 2)Gaussian elimination swapping 3)MRB magnitude swapping
class osd:
    def __init__(self,code):
        self.original_H = code.original_H   
        self.n_dims = code.check_matrix_col
        self.k = code.k
        self.m = self.n_dims - self.k         
    #magnitude of signals
    def mag_input_gen(self,inputs):
        inputs_abs = abs(inputs)
        reorder_index_batch = tf.argsort(inputs_abs,axis=-1,direction='ASCENDING')
        return reorder_index_batch         
   
    def check_matrix_reorder(self,iteration_inputs,inputs,labels):
        expanded_H = tf.expand_dims(self.original_H,axis=0)
        list_length = GL.get_map('num_iterations')+1
        #query the least reliable independent positions
        lri_p = self.mag_input_gen(inputs)
        order_inputs = tf.gather(inputs,lri_p,batch_dims=1)
        order_original_list = [tf.gather(iteration_inputs[i::list_length],lri_p,batch_dims=1)  for i in range(list_length)]
        order_labels = tf.gather(labels,lri_p,batch_dims=1)
        batched_H = tf.transpose(tf.tile(expanded_H,[inputs.shape[0],1,1]),perm=[0,2,1])
        tmp_H_list = tf.gather(batched_H,lri_p,batch_dims=1)
        order_H_list = tf.transpose(tmp_H_list,perm=[0,2,1])
        return order_H_list,order_inputs,order_original_list,order_labels 

    def identify_mrb(self,order_H_list):
       #initialize of mask
       code = GL.get_map('code_parameters')
       updated_index_list = []
       updated_M_list = []
       for i in range(order_H_list.shape[0]):
           # H assumed to be full row rank to obtain its systematic form
           tmp_H = np.copy(order_H_list[i])
           #reducing into row-echelon form and record column 
           #indices involved in pre-swapping
           swapped_H,record_col_index = self.full_gf2elim(tmp_H) 
           index_length = len(record_col_index)
           #update all swapping index
           index_order = np.array(range(code.check_matrix_col))
           for j in range(index_length):
               tmpa = record_col_index[j][0]
               tmpb = record_col_index[j][1]
               index_order[tmpa],index_order[tmpb] =  index_order[tmpb],index_order[tmpa]   
           #udpated mrb indices
           mrb_swapping_index = tf.argsort(index_order[-code.k:],axis=0,direction='ASCENDING')
           mrb_order = tf.sort(index_order[-code.k:],axis=0,direction='ASCENDING')
           updated_index_order = tf.concat([index_order[:(code.check_matrix_col-code.k)],mrb_order],axis=0)
           updated_M = tf.gather(swapped_H[:,-code.k:],mrb_swapping_index,axis=1)    
           updated_index_list.append(updated_index_order)  
           updated_M_list.append(updated_M)
       return updated_index_list,updated_M_list
     
    def error_pattern_gen(self,direction,range_list):
        code = GL.get_map('code_parameters')
        def function1_inline(i,value):
            if value:        
                tmp = combinations(range_list[i],value)
            else:
                tmp = [-1]
            return tmp
        itering_list = [ function1_inline(i,value) for i,value in enumerate(direction)]
        combination_join = list(it.product(*itering_list))
        filtered_comb = [list(filter(lambda x:x!=-1,combination_element)) for combination_element in combination_join]
        error_patterns = np.zeros(shape=[len(combination_join),code.k],dtype=int)      
        def function2_inline(i,sequence):
            indices = list(chain.from_iterable(sequence))
            error_patterns[i, indices] = 1
        [ function2_inline(i,sequence) for i, sequence in enumerate(filtered_comb)]    
                      
        return error_patterns 

    def collect_tep(self,decoding_path):
        Convention_path_indicator = GL.get_map('convention_path')
        code = GL.get_map('code_parameters')
        factor_gap = GL.get_map('delimiter')
        #segmentation of MRB
        delimiter1 = code.k//factor_gap
        delimiter2 = 3*delimiter1
        LR_part = range(delimiter1)
        MR_part = range(delimiter1,delimiter2)
        HR_part = range(delimiter2,code.k)
        range_seg = [LR_part,MR_part,HR_part]
        #trim decoding path if requested
        if Convention_path_indicator:          
            valid_length = len(decoding_path)
        else:
            valid_length = GL.get_map('decoding_length')         
        proper_error_pattern_list = [self.error_pattern_gen(decoding_path[j],range_seg) for j in range(valid_length)]
        proper_error_pattern_matrix = tf.concat(proper_error_pattern_list,axis=0)
        return proper_error_pattern_matrix

    def is_row_in_matrix(self,v, M):
        # Ensure v and M are TensorFlow tensors
        v = tf.convert_to_tensor(v, dtype=tf.int32)
        M = tf.convert_to_tensor(M, dtype=tf.int32)        
        # Compare v with every row in M (broadcasting)
        match = tf.reduce_all(tf.equal(M, v), axis=1)      
        # Check if any row matches
        return tf.reduce_any(match).numpy()

    def best_estimating(self,order_list,decoding_matrix):
        code = GL.get_map('code_parameters')
        order_H_list,order_inputs,order_original_list,order_labels = order_list
        updated_index_list,updated_M_list = self.identify_mrb(order_H_list)
        input_size = order_labels.shape[0]              
        order_input_list = [tf.gather(order_inputs[i],updated_index_list[i]) for i in range(input_size)]
        order_original_input_list = [tf.gather(order_original_list[0][i],updated_index_list[i]) for i in range(input_size)]  
        order_label_list = [tf.cast(tf.gather(order_labels[i],updated_index_list[i]),dtype=tf.int32) for i in range(input_size)]                  
       
        correct_counter = 0
        Bound_fail_counter = 0 
        ML_fail_counter = 0

        for i in range(input_size):
            print('.',end='')
            #ground truth
            order_hard_original = tf.where(order_original_input_list[i]>0,0,1)
            order_hard = tf.where(order_input_list[i]>0,0,1)
            initial_mrb = order_hard[-code.k:]           
            M_matrix = updated_M_list[i] 
            mag_metric = abs(order_original_input_list[i]) 
            sorted_list,sorted_indices,estiamted_codeword = self.acquire_min_estimate(decoding_matrix,initial_mrb,M_matrix,order_hard_original,mag_metric) #minimum of teps belonging to some order pattern
            discrepancy_vector = tf.math.mod(estiamted_codeword + order_label_list[i],2)
            discrepancy_origin = tf.math.mod(order_hard_original + order_label_list[i],2)
            ground_discrepancy_sum = tf.reduce_sum(tf.cast(discrepancy_origin,tf.float32)*mag_metric)
            #shortened_list = [np.round(tensor, decimals=5) for tensor in sorted_list[:5]] 
            if tf.reduce_sum(discrepancy_vector)==0:
                correct_counter += 1
            else:
                if sorted_list[0] > ground_discrepancy_sum:
                    Bound_fail_counter += 1
                else:
                    ML_fail_counter += 1               
        return correct_counter,Bound_fail_counter,ML_fail_counter
    
    def sliding_window_ops(self,SWA_model,window,global_min,k): 
        early_termination = False
        # Sort by the first element of each tuple
        sorted_window = sorted(window, key=lambda x: x[0])
        expanded_list = [sorted_window[i][0] for i in range(len(sorted_window))]+[float(k)]
        output_prb = tf.squeeze(SWA_model(tf.reshape(expanded_list,[1,-1])))
        win_min = sorted_window[0]
        #if output_prb[0]-output_prb[1] > 0.95: 
        if output_prb[1] - output_prb[0] > GL.get_map('soft_margin'): 
            early_termination = True        
        min_tuple = min(global_min, win_min, key=lambda x: x[0])
        return early_termination,min_tuple

    def acquire_min_estimate(self,error_pattern_matrix,initial_mrb,M_matrix,original_hard_decision,mag_metric): #minimum of teps belonging to some order pattern
            estimated_mrb_matrix = (error_pattern_matrix + tf.expand_dims(initial_mrb,axis=0))%2     
            estimated_lrb_matrix = tf.math.mod(tf.matmul(estimated_mrb_matrix, tf.transpose(M_matrix)), 2)            
            codeword_candidate_matrix = tf.concat([estimated_lrb_matrix,estimated_mrb_matrix],axis=1)
            #order pattern min-value
            #selection best estimation 
            discrepancy_matrix = tf.math.mod(codeword_candidate_matrix + tf.expand_dims(original_hard_decision,0),2)   
            row_sums = tf.reduce_sum(tf.cast(discrepancy_matrix,tf.float32)*tf.expand_dims(mag_metric,axis=0),axis=-1) 
            sorted_min_sums = tf.sort(row_sums)
            sorted_indices = tf.argsort(row_sums)
            #min_sum = tf.reduce_min(row_sums)
            min_index = tf.argmin(row_sums)
            return sorted_min_sums,sorted_indices,codeword_candidate_matrix[min_index]

    def choose_decoding_strategy(self,SWA_model,input_list,inputs,labels,decoding_matrix):
        order_list= self.check_matrix_reorder(input_list,inputs,labels)
        if GL.get_map('sliding_strategy'):
            correct_counter,fail_counter,windows_sum,swa_counter,complexity_sum = self.sliding_osd(SWA_model,order_list,decoding_matrix)
        else:            
            correct_counter,Bound_fail_counter,ML_fail_counter = self.best_estimating(order_list,decoding_matrix)
            fail_counter = Bound_fail_counter+ML_fail_counter
            windows_sum = 0
            swa_counter = 0
            complexity_sum = GL.get_map('intercept_length')*(correct_counter+fail_counter)
        return correct_counter,fail_counter,windows_sum,swa_counter,complexity_sum
    
    def sliding_osd(self,SWA_model,order_list,decoding_matrix):
        code = GL.get_map('code_parameters')  
        intercept_length = GL.get_map('intercept_length')
        block_size = GL.get_map('block_size')
        sliding_win_width = GL.get_map('win_width')
        
        block_length = intercept_length//block_size
        success_dec = 0
        failure_dec = 0
        
        order_H_list,order_inputs,order_original_list,order_labels = order_list
        updated_index_list,updated_M_list = self.identify_mrb(order_H_list)
        input_size = order_labels.shape[0]              
        teps_list = [decoding_matrix[i*block_size:(i+1)*block_size] for i in range(block_length)]
        order_input_list = [tf.gather(order_inputs[i],updated_index_list[i]) for i in range(input_size)]
        order_original_input_list = [tf.gather(order_original_list[0][i],updated_index_list[i]) for i in range(input_size)]  
        order_label_list = [tf.cast(tf.gather(order_labels[i],updated_index_list[i]),dtype=tf.int32) for i in range(input_size)]                  
        complexity_sum = 0
        windows_sum = 0
        swa_counter = 0
        for i in range(input_size):
            print('.',end='')
            #ground truth
            order_hard_original = tf.where(order_original_input_list[i]>0,0,1)
            mag_metric = abs(order_original_input_list[i])
            M_matrix = updated_M_list[i]  
            #serial processing each codeword            
            order_hard = tf.where(order_input_list[i]>0,0,1)
            initial_mrb = order_hard[-code.k:]
            #initialize the window content
            window = []    
            for error_pattern_matrix in teps_list[:sliding_win_width]:
                sorted_min_sums,_,codeword_candidate = self.acquire_min_estimate(error_pattern_matrix,initial_mrb,M_matrix,order_hard_original,mag_metric)
                min_sum = (sorted_min_sums[0],codeword_candidate)
                window.append(min_sum)  
            # Find the tuple with the smallest first component
            global_min = min(window, key=lambda x: x[0])
            for k in range(len(teps_list)-sliding_win_width+1):    
                deep_limit = k+sliding_win_width
                if k != 0:
                    #update window content
                    #estimations of codeword candidate
                    error_pattern_matrix = teps_list[sliding_win_width+k-1]
                    sorted_min_sums,_,codeword_candidate  = self.acquire_min_estimate(error_pattern_matrix,initial_mrb,M_matrix,order_hard_original,mag_metric)
                    min_sum = (sorted_min_sums[0],codeword_candidate)
                    window.append(min_sum) 
                    # Keep the last four elements plus the new one
                    window = window[-sliding_win_width:]
                    if min_sum[0] > global_min[0]:
                        continue
                #tf.print(window)   
                early_termination,global_min = self.sliding_window_ops(SWA_model,window,global_min,k)
                swa_counter += 1  #record number of swa model callings
                if early_termination:
                    break   
            window_num = deep_limit - sliding_win_width+1
            #tf.print(window_num)
            complexity_sum += deep_limit*block_size
            windows_sum += window_num
            # Check equality
            are_equal = tf.reduce_all(tf.equal(global_min[1], order_label_list[i]))
            if are_equal:
                success_dec += 1
            else:
                failure_dec += 1
        return success_dec,failure_dec,windows_sum,swa_counter,complexity_sum
     
    def full_gf2elim(self,M):
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

    def execute_osd(self,input_list,inputs,labels,proper_error_pattern_matrix):
        threshold_indicator = GL.get_map('threshold_indicator')
        threshold_sum = GL.get_map('threshold_sum')
        code = GL.get_map('code_parameters')   
        list_length = GL.get_map('num_iterations')+1
        order_list= self.check_matrix_reorder(input_list,inputs,labels)
        order_H_list,order_inputs,order_original_list,order_labels = order_list
        updated_index_list,updated_M_list,swap_len_list,swap_lrb_position_list = self.identify_mrb(order_H_list)
        input_size = inputs.shape[0]

        
        order_input_list = [tf.gather(order_inputs[i],updated_index_list[i]) for i in range(input_size)]
        order_label_list = [tf.cast(tf.gather(order_labels[i],updated_index_list[i]),dtype=tf.int32) for i in range(input_size)]
        order_original_input_list = [tf.gather(order_original_list[i][j],updated_index_list[j]) for i in range(list_length) for j in range(input_size)]                

        def function_inline(i):
            print('.',end='')
            #serial processing each codeword
            order_hard = tf.where(order_input_list[i]>0,0,1)
            M_matrix = updated_M_list[i]
            #generate all possible error patterns of mrb            
            error_pattern_matrix = proper_error_pattern_matrix       
            # setting starting point                                              
            initial_mrb = order_hard[-code.k:]
            initial_lrb = tf.reshape(order_hard[:code.check_matrix_col-code.k],[-1,1])
            codeword_lrb = tf.matmul(tf.reshape(initial_mrb,[1,-1]),tf.cast(M_matrix,tf.int32),transpose_b=True)%2
            codeword_candidate_matrix = tf.concat([codeword_lrb,tf.reshape(initial_mrb,[1,-1])],axis=1)
            #estimations of codeword candidate
            estimated_mrb_matrix = tf.transpose((tf.cast(error_pattern_matrix,tf.int32)+initial_mrb)%2)
            estimated_lrb_matrix = tf.matmul(tf.cast(M_matrix,tf.int32),estimated_mrb_matrix)%2        
            #new branch to exclude some low probability test error patterns
            if threshold_indicator:
                swap_lrb_position_vector = tf.reshape(swap_lrb_position_list[i],[-1,1])
                binary_lrb_sum= (initial_lrb+estimated_lrb_matrix)%2
                focus_position_diff_sum = tf.reduce_sum(binary_lrb_sum*swap_lrb_position_vector,axis=0)
                residual_index = tf.squeeze(tf.where(focus_position_diff_sum <= 2*threshold_sum))
                if len(residual_index) < 2:
                    residual_index = [0,1,2,3]
                estimated_mrb_matrix = tf.gather(estimated_mrb_matrix,residual_index,axis=1)
                estimated_lrb_matrix = tf.gather(estimated_lrb_matrix,residual_index,axis=1)
            candidate_size = estimated_mrb_matrix.shape[1]
            #print('candidate_size:',candidate_size)
            codeword_candidate_matrix = tf.transpose(tf.concat([estimated_lrb_matrix,estimated_mrb_matrix],axis=0))
            return codeword_candidate_matrix,candidate_size
        tuple_list = [function_inline(i) for i in range(input_size)]
        candidate_list = [tuple_list[i][0] for i in range(input_size)]
        candidate_size_list = [tuple_list[i][1] for i in range(input_size)]
        print('\n')
        candiate_sum_size = tf.reduce_sum(candidate_size_list)      
        return order_original_input_list,order_label_list,candidate_list,candiate_sum_size