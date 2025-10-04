# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:34:01 2023

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import globalmap as GL
#from sympy.utilities.iterables import multiset_permutations

# input operation:1)magnitude swapping 2)Gaussian elimination swapping 3)MRB magnitude swapping
class osd_light:
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
   
    def check_matrix_reorder(self,inputs,labels):
        expanded_H = tf.expand_dims(self.original_H,axis=0)
        #query the least reliable independent positions
        lri_p = self.mag_input_gen(inputs)
        order_inputs = tf.gather(inputs,lri_p,batch_dims=1)
        order_labels = tf.gather(labels,lri_p,batch_dims=1)
        batched_H = tf.transpose(tf.tile(expanded_H,[inputs.shape[0],1,1]),perm=[0,2,1])
        tmp_H_list = tf.gather(batched_H,lri_p,batch_dims=1)
        order_H_list = tf.transpose(tmp_H_list,perm=[0,2,1])
        return order_H_list,order_inputs,order_labels

    def identify_mrb(self,order_H_list):
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
            swapped_H,record_col_index = self.full_gf2elim(tmp_H) 
            index_length = len(record_col_index)
            #update all swapping index
            index_order = np.array(range(code.check_matrix_col))
            for j in range(index_length):
                tmpa = record_col_index[j][0]
                tmpb = record_col_index[j][1]
                index_order[tmpa],index_order[tmpb] =  index_order[tmpb],index_order[tmpa]   
            #udpated mrb indices
            updated_MRB = index_order[-code.k:]
            mrb_swapping_index = tf.argsort(updated_MRB,axis=0,direction='ASCENDING')
            mrb_order = tf.sort(updated_MRB,axis=0,direction='ASCENDING')
            updated_index_order = tf.concat([index_order[:(code.n-code.k)],mrb_order],axis=0)
            updated_M = tf.gather(swapped_H[:,-code.k:],mrb_swapping_index,axis=1)    
            updated_index_list.append(updated_index_order)  
            updated_M_list.append(updated_M)
            swap_indicator = np.where(updated_MRB>=code.check_matrix_col-code.k,0,1)
            # focus of rear part of positions in LRB plus those positions swapped from MRB      
            swap_sum = sum(swap_indicator)
            swap_len_list.append(swap_sum)
        return updated_index_list,updated_M_list,swap_len_list,swap_lrb_position_list      
        
    def convention_osd_preprocess(self,inputs,labels):
        code = GL.get_map('code_parameters')   
        order_list= self.check_matrix_reorder(inputs,labels)
        order_H_list,order_inputs,order_labels = order_list
        updated_index_list,updated_M_list,swap_len_list,swap_lrb_position_list = self.identify_mrb(order_H_list)
        input_size = inputs.shape[0]        
        order_input_list = [tf.gather(order_inputs[i],updated_index_list[i]) for i in range(input_size)]
        order_label_list = [tf.cast(tf.gather(order_labels[i],updated_index_list[i]),dtype=tf.int32) for i in range(input_size)]                  
        authentic_error_pattern_list = []
        for i in range(input_size):
            #ground truth
            order_hard = tf.where(order_input_list[i][-code.k:]>0,0,1)
            authentic_error_pattern = (order_label_list[i][-code.k:] + order_hard)%2
            authentic_error_pattern_list.append(authentic_error_pattern)
        return np.reshape(authentic_error_pattern_list,[-1,code.k])
            
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
