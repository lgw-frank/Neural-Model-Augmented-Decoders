# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:41:33 2022

@author: Administrator
"""
import tensorflow as tf
import globalmap as GL
import data_generating as Data_gen
import numpy as np
import math,os
from collections import Counter
import matplotlib.pyplot as plt


#from distfit import distfit    
np.random.seed(0)

class Decoding_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.decoder_layer = Decoder_Layer()  # Explicitly track the layer
    def build(self, input_shape):
        # Convert TensorShape to plain Python dimensions for your layer
        if hasattr(input_shape, 'as_list'):
            processed_shape = input_shape.as_list()
        else:
            processed_shape = list(input_shape)       
        # Build your decoder layer with concrete dimensions
        self.decoder_layer.build(processed_shape)       
        # Skip super().build() entirely - it's often not needed
        self.built = True  # Manually mark as built
    def call(self,inputs): 
        soft_output_list,labels = self.decoder_layer(inputs)
        return soft_output_list,labels
    
    def collect_failed_input_output(self,soft_output_list,labels,indices):
        num_iterations = GL.get_map('num_iterations')
        list_length = num_iterations + 1
        buffer_inputs = []
        buffer_labels = []
        #indices = tf.squeeze(index,1).numpy()
        for i in indices:
            for j in range(list_length):
                buffer_inputs.append(soft_output_list[j][i])    
                buffer_labels.append(labels[i])
        return buffer_inputs,buffer_labels     

    def get_eval_fer(self,soft_output_list,labels):
        soft_output = soft_output_list[-1]
        tmp = tf.cast(tf.where(soft_output>0,0,1),tf.int64)
        err_batch = tf.where(tmp == labels,0,1)
        err_sum = tf.reduce_sum(err_batch,-1)
        FER_data = tf.where(err_sum!=0,1,0)     
        FER_num = tf.math.count_nonzero(FER_data)
        #identify the indices of undected decoding errors        
        return FER_num.numpy() 

    def new_get_eval(self,Model_undetected,soft_output_list,labels):
        code = GL.get_map('code_parameters')
        num_iterations = GL.get_map('num_iterations')
        soft_margin = GL.get_map('soft_margin')
        H = code.original_H
        soft_output = soft_output_list[-1]
        final_hard_decision = tf.cast(tf.where(soft_output>0,0,1),tf.int64)
        syndrome = tf.matmul(final_hard_decision,H,transpose_b=True)%2
        nms_declare_n_index = np.where(np.sum(syndrome,-1)!=0)[0]
        nms_declare_p_index = np.where(np.sum(syndrome,-1)==0)[0]
        delta_matrix = self.transform_samples(soft_output_list)
        passed_list = [delta_matrix[i] for i in nms_declare_p_index]
        p_matrix = tf.reshape(passed_list,[-1,num_iterations,1])
        outputs = Model_undetected(p_matrix)
        #partition the outputs into two categories
        compact_index = np.where(outputs[:,0]-outputs[:,1]>soft_margin)[0]
        #passed index after model
        ude_declare_p_index = [nms_declare_p_index[i] for i in compact_index]
        #failed index after model
        ude_declare_n_index = set(nms_declare_p_index)-set(ude_declare_p_index)
        declare_n_index = list(ude_declare_n_index | set(nms_declare_n_index))
        #index_fail = list(set(index_fail))
        #ground truth comparison to determine FER and BER
        correct_counter = 0
        err_batch = tf.where(final_hard_decision == labels,0,1)
        err_word_indicator = tf.reduce_sum(err_batch,-1)
        err_bit_sum = tf.reduce_sum(err_word_indicator)
        false_p_index = []
        false_n_index = []
        for i in ude_declare_p_index:
            correct_sign = tf.where(err_word_indicator[i]==0,1,0)
            correct_counter += correct_sign 
            if not correct_sign:
                false_p_index.append(i)
        for i in ude_declare_n_index:
            correct_sign = tf.where(err_word_indicator[i]==0,1,0)
            if correct_sign:
                false_n_index.append(i)
        FER_count = soft_output.shape[0]- correct_counter              
        BER_count = err_bit_sum   
        return FER_count,BER_count,false_p_index,false_n_index,declare_n_index  
    
    def get_eval(self,soft_output_list,labels):
        code = GL.get_map('code_parameters')
        labels = tf.cast(labels,tf.int64)
        H = code.original_H
        soft_output = soft_output_list[-1]
        tmp = tf.cast(tf.where(soft_output>0,0,1),tf.int64)
        syndrome = tf.matmul(tmp,H,transpose_b=True)%2
        index1 = np.nonzero(tf.reduce_sum(syndrome,-1))[0]
        err_batch = tf.where(tmp == labels,0,1)
        err_sum = tf.reduce_sum(err_batch,-1)
        BER_count = tf.reduce_sum(err_sum)
        FER_data = tf.where(err_sum!=0,1,0)
        FER_count = tf.math.count_nonzero(FER_data)
        #identify the indices of undected decoding errors        
        return FER_count, BER_count,index1   
    
    def get_eval_expanded(self,soft_output_list,labels, iteration_counter):
        code = GL.get_map('code_parameters')
        labels = tf.cast(labels,tf.int64)
        H = code.H
        history_index_set = set()
        for i in range(soft_output_list.shape[0]):
            tmp_post_LLRs = soft_output_list[i]
            tmp_hard_decision = tf.cast(tf.where(tmp_post_LLRs>0,0,1),tf.int64)
            syndrome = tf.matmul(tmp_hard_decision,H,transpose_b=True)%2
            sums_row = np.sum(syndrome, axis=-1)
            success_index_set = set(np.where(sums_row==0)[0])
            record_index_set = success_index_set - history_index_set
            iteration_counter.update({i:len(record_index_set)})
            history_index_set = success_index_set           
        fail_index = np.nonzero(sums_row)[0]    
        iteration_counter.update({-1:len(fail_index)})
        err_batch = tf.where(tmp_hard_decision == labels,0,1)
        err_sum = tf.reduce_sum(err_batch,-1)
        BER_count = tf.reduce_sum(err_sum)
        FER_data = tf.where(err_sum!=0,1,0)
        FER_count = tf.math.count_nonzero(FER_data)
        #identify the indices of undected decoding errors        
        return FER_count, BER_count,fail_index,iteration_counter   
    
    def create_samples(self,soft_output_list, labels):
        print('.',end=' ')
        label_bool = tf.cast(labels, tf.bool)
        code = GL.get_map('code_parameters')
        H = code.original_H
        soft_output = soft_output_list[-1]         
        final_hard = tf.cast(tf.where(soft_output>0,0,1),tf.int64)
        delta_list = self.transform_samples(soft_output_list)
        syndrome = tf.matmul(final_hard,H,transpose_b=True)%2
        #indices of discarded pair
        index1 = np.nonzero(tf.reduce_sum(syndrome,-1))[0] 
        #cared indices to be classifed as positive or negative pair
        index2 = np.where(tf.reduce_sum(syndrome,-1) == 0)[0]
        pairs, pair_labels = [], []
        ground_labels = []
        for i in index2:
            output_hard_decision = tf.cast((soft_output[i] < 0),tf.bool)       
            err_indicator = tf.math.logical_xor(output_hard_decision, label_bool[i])
            Find_FER_sign = tf.reduce_any(err_indicator)
            pairs.append(delta_list[i])
            if Find_FER_sign:
                pair_labels.append(1)               
            else:
                pair_labels.append(0)
            ground_labels.append(labels[i])
        return tf.convert_to_tensor(pairs), tf.convert_to_tensor(pair_labels),tf.convert_to_tensor(ground_labels),index1  

    def transform_samples(self,soft_output_list):
        list_length = len(soft_output_list)
        initial_input = soft_output_list[0]
        final_output = soft_output_list[-1]         
        final_hard = tf.where(final_output>0,0,1)
        delta_list = []
        for i in range(list_length-1): 
            current_iteration_hard =  tf.where(soft_output_list[i]>0,0,1)
            difference_indicator = tf.cast((final_hard+current_iteration_hard)%2,tf.float32)
            differed_distance = tf.reduce_sum(difference_indicator*abs(initial_input),1,keepdims=True)
            delta_list.append(differed_distance)
        delta_matrix = tf.concat(delta_list,1)
        return delta_matrix
    
class Decoder_Layer(tf.keras.layers.Layer):
    def __init__(self,initial_value = -2.):
        super().__init__()
        self.decoder_type = GL.get_map('selected_decoder_type')
        self.num_iterations = GL.get_map('num_iterations')
        self.code = GL.get_map('code_parameters')
        self.H = self.code.H
        self.initials = initial_value
        self.supplement_matrix =  tf.expand_dims(tf.cast(1-self.H,dtype=tf.float32),0)
        # SOLUTION: Force immediate weight creation with explicit name
        with tf.init_scope():  # <-- CRITICAL FIX
            self.decoder_check_normalizor = self.add_weight(
                name='decoder_check_normalizor',
                shape=[1],
                trainable=True,
                initializer=tf.keras.initializers.Constant(initial_value),
                dtype=tf.float32
            )
        self.built = True
    #V:vertical H:Horizontal D:dynamic S:Static  /  VSSL: Vertical Static/Dynamic Shared Layer
    def build(self, input_shape):   
        pass
    def call(self,inputs):
        soft_input = inputs[0]
        labels = inputs[1]    
        outputs = self.belief_propagation_op(soft_input, labels)
        soft_output_array, label = outputs
        return soft_output_array.stack(), label  # ✅ stack before returning
         

    def get_evenly_shifted_integers(self,limit,num_count):
        start = np.random.randint(limit)
        step = limit//num_count
        sequence = (start + step * np.arange(num_count)) %limit      
        return list(sequence)
      
    def interleave_columns(self,tensor1, tensor2):
        """
        Interleave the columns of two tensors into one tensor.
        
        Args:
            tensor1 (tf.Tensor): First tensor of shape (m, n).
            tensor2 (tf.Tensor): Second tensor of shape (m, n) or (m, n-1).
        
        Returns:
            tf.Tensor: Tensor with interleaved columns.
        """
        # Ensure tensors have compatible shapes
        if tensor1.shape[0] != tensor2.shape[0]:
            raise ValueError("Tensors must have the same number of rows")
        
        # Get shapes
        m, n1 = tensor1.shape
        n2 = tensor2.shape[1]
        
        # Interleave columns using stack and reshape
        if n1 == n2:
            # If the number of columns is the same, stack along the third axis and then reshape
            interleaved = tf.reshape(tf.stack([tensor1, tensor2], axis=2), (m, n1 + n2))
        elif n1 == n2 + 1:
            # If tensor1 has one more column than tensor2
            tensor1_split = tf.split(tensor1, [n2, 1], axis=1)
            interleaved = tf.reshape(tf.stack([tensor1_split[0], tensor2], axis=2), (m, n1 + n2 - 1))
            interleaved = tf.concat([interleaved, tensor1_split[1]], axis=1)
        else:
            raise ValueError("tensor1 must have the same number of columns or one more column than tensor2")
        
        return interleaved
    
    def frobenius_automorphism(self,codeword, n):
        """
        Apply the Frobenius automorphism to a binary codeword using the formula (2 * i) % n.
        
        Args:
            codeword (tf.Tensor): Binary codeword of shape (1, n).
            n (int): Length of the codeword.
        
        Returns:
            tf.Tensor: Permuted codeword.
        """
        # Generate positions using the formula (2 * i) % n
        positions = [(2 * i) % n for i in range(n)]
        
        # Create the permutation matrix using the generated positions
        perm_matrix = tf.eye(n, dtype=codeword.dtype)
        perm_matrix = tf.gather(perm_matrix, positions, axis=1)
        
        # Apply the permutation
        permuted_codeword = tf.matmul(codeword, perm_matrix)
        
        return permuted_codeword
    
    def inverse_frobenius_automorphism(self,codeword, n):
        """
        Apply the inverse Frobenius automorphism to a binary codeword using the inverse of (2 * i) % n.
        
        Args:
            codeword (tf.Tensor): Binary codeword of shape (1, n).
            n (int): Length of the codeword.
        
        Returns:
            tf.Tensor: Permuted codeword (original codeword before applying the Frobenius automorphism).
        """
        # Generate positions using the formula (2 * i) % n
        positions = [(2 * i) % n for i in range(n)]
        
        # Generate the inverse positions by reversing the permutation
        inverse_positions = [positions.index(i) for i in range(n)]
        
        # Create the permutation matrix for the inverse positions
        perm_matrix = tf.eye(n, dtype=codeword.dtype)
        perm_matrix = tf.gather(perm_matrix, inverse_positions, axis=1)
        
        # Apply the inverse permutation
        recovered_codeword = tf.matmul(codeword, perm_matrix)
        
        return recovered_codeword

    
    def aggregate_cyclic_words(self,soft_input):
        #cycling the input from start to end
        num_shifts = GL.get_map('num_shifts')
        if num_shifts > 0:
            soft_input_split = tf.concat([soft_input[:,0::2],soft_input[:,1::2]],axis=1)
            permutated_codewords = self.frobenius_automorphism(soft_input,soft_input.shape[1])
            soft_input = tf.concat([soft_input,soft_input_split,permutated_codewords],0)            
            shifted_integers = self.get_evenly_shifted_integers(soft_input.shape[1],num_shifts)
            shifted_input_list = [tf.roll(soft_input,shifted_integers[i],axis=1)  for i in range(num_shifts)]
            super_inputs = tf.concat(shifted_input_list,axis=0)
        else:
            super_inputs = soft_input
            shifted_integers = [-1]
        return super_inputs,shifted_integers                  

    def belief_propagation_op(self, soft_input, labels):
        soft_output_array = tf.TensorArray(
            dtype=tf.float32,
            size=self.num_iterations + 1,
            clear_after_read=False  # <-- Required if you want to read an index multiple times
        )
        # Write initial value
        soft_output_array = soft_output_array.write(0, soft_input)
    
        def condition(soft_output_array, labels, iteration):
            return iteration < self.num_iterations
        def body(soft_output_array,labels, iteration):
            super_input,shifted_integers = self.aggregate_cyclic_words(soft_output_array.read(iteration))
            vc_matrix = self.compute_vc(super_input)
            # compute cv
            cv_matrix = self.compute_cv(vc_matrix)      
            # get output for this iteration
            soft_output_array = self.marginalize(cv_matrix, super_input,shifted_integers,iteration,soft_output_array)  
            iteration += 1   
            return soft_output_array,labels, iteration
    
        soft_output_array, labels, iteration = tf.while_loop(
            condition,
            body,
            loop_vars=[soft_output_array, labels, 0]
        )   
        return soft_output_array, labels
            
    # compute messages from variable nodes to check nodes

    def compute_vc(self, soft_input):
        check_matrix_H = tf.cast(self.code.H,tf.float32)   
        vc_matrix = tf.expand_dims(soft_input,axis=1)*check_matrix_H   
        return vc_matrix 
       
    def optimized_topk(self,matrix_3d, mask_2d):
        # Flatten and mask
        a, b, c = tf.shape(matrix_3d)[0], tf.shape(matrix_3d)[1], tf.shape(matrix_3d)[2]
        masked = matrix_3d * tf.tile(tf.expand_dims(mask_2d, 0), [a, 1, 1])      
        # Replace zeros with large value
        processed = tf.where(masked == 0, 1e10, masked)    
        # Reshape and find top-2
        flattened = tf.reshape(processed, [a * b, c])
        topk = -tf.math.top_k(-flattened, k=2)[0]   
        # Extract results
        smallest = tf.reshape(topk[:, 0], [a, b, 1])
        second_smallest = tf.reshape(topk[:, 1], [a, b, 1])
        # Apply your update rule
        updated = tf.where(
            matrix_3d == smallest,
            second_smallest,
            smallest
        ) * mask_2d        
        return updated        
       
    def compute_cv(self,vc_matrix):
        check_matrix_H = self.H
        #operands sign processing 
        sign_info = self.supplement_matrix + vc_matrix
        vc_matrix_sign = tf.sign(sign_info)
        temp1 = tf.reduce_prod(vc_matrix_sign,2,keepdims=True)
        transition_sign_matrix = temp1*check_matrix_H
        result_sign_matrix = transition_sign_matrix*vc_matrix_sign 
        # Get the number of rows and columns in the matrix
        batches, rows, cols = vc_matrix.shape
        updated_matrix = self.optimized_topk(tf.abs(vc_matrix), tf.cast(check_matrix_H,tf.float32))     
        if GL.get_map('selected_decoder_type') in ['NMS-1']:
          normalized_tensor = tf.nn.softplus(self.decoder_check_normalizor)
        cv_matrix = normalized_tensor*updated_matrix*tf.stop_gradient(result_sign_matrix) 
        return cv_matrix  
 
    #combine messages to get posterior LLRs
    def marginalize(self,cv_matrix, soft_input,shifted_integers,iteration,soft_output_array):
        temp = tf.reduce_sum(cv_matrix,1)
        #aligning with cycled input
        num_shifts = GL.get_map('num_shifts')
        if num_shifts > 0:
            batch_size = soft_input.shape[0]//num_shifts
            basic_batch_size = batch_size//3 #three kinds of permutations
            # Use vectorized operations
            shifted_temps = [tf.roll(temp[i*batch_size:(i+1)*batch_size], shift=-shifted_integers[i], axis=1) for i in range(num_shifts)]
            #shift permutation
            shift_list = [shifted_temps[i][:basic_batch_size] for i in range(num_shifts)]
            tensor1 = tf.add_n(shift_list)
            shift2_list = [shifted_temps[i][basic_batch_size:2*basic_batch_size] for i in range(num_shifts)]
            tensor2 = tf.add_n(shift2_list)
            width = math.ceil(tensor2.shape[1]/2)
            interleaved = self.interleave_columns(tensor2[:,:width],tensor2[:,width:]) 
            shift3_list = [shifted_temps[i][2*basic_batch_size:] for i in range(num_shifts)]
            tensor3 = tf.add_n(shift3_list)
            permutated = self.inverse_frobenius_automorphism(tensor3, tensor3.shape[1])
            soft_output = soft_output_array.read(iteration)+(tensor1+interleaved+permutated)
        else:
            soft_output = soft_output_array.read(iteration)+temp
        soft_output_array = soft_output_array.write(iteration + 1, soft_output)
        return soft_output_array    

def retore_saved_model():
    prefix_str = GL.get_map('prefix_str')
    decoder_type = GL.get_map('selected_decoder_type')
    n_iteration = GL.get_map('num_iterations')
    reduction_iteration = GL.get_map('reduction_iteration')
    redundancy_factor = GL.get_map('redundancy_factor')
    n_dims = GL.get_map('code_parameters').check_matrix_col
    ckpt_nm = f'{prefix_str}-ckpt'
    restore_ckpts_dir = f'../{prefix_str.upper}_{str(n_dims)}_training/ckpts/{decoder_type}/{str(n_iteration)}th/IF{str(reduction_iteration)}-{str(redundancy_factor)}/'
    #instance of Model creation
    test_Model = Decoding_model()
    # saved restoring info
    restore_step = 'latest'
    checkpoint = tf.train.Checkpoint(myAwesomeModel=test_Model)
    ckpt_f = retore_saved_file(restore_ckpts_dir,restore_step,ckpt_nm)
    status = checkpoint.restore(ckpt_f)
    print(f'Loading saved parameters from {ckpt_f}')
    status.expect_partial()
    return test_Model

def retore_saved_file(restore_ckpts_dir,restore_step,ckpt_nm):
    print("Ready to restore a saved latest or designated model!")
    ckpt = tf.train.get_checkpoint_state(restore_ckpts_dir)
    if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
      if restore_step == 'latest':
        ckpt_f = tf.train.latest_checkpoint(restore_ckpts_dir)
      else:
        ckpt_f = restore_ckpts_dir+ckpt_nm+'-'+restore_step
      print('Loading wgt file: '+ ckpt_f)   
    else:
      print('Error, no qualified file found')
    return ckpt_f

def calculate_loss(inputs,labels):
    labels = tf.cast(labels,tf.float32)  
    #measure discprepancy via cross entropy metric which acts as the loss definition for deep learning per batch         
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-inputs, labels=labels))
    return  loss.numpy()
 
#save modified data for postprocessing
def save_decoded_data2(snr,updated_buffer,dir_file,log_filename):
    list_length = GL.get_map('num_iterations')+1
    stacked_buffer_info = tf.stack(updated_buffer[0])
    stacked_buffer_label = tf.stack(updated_buffer[1])
    CE_loss_list = []
    for i in range(list_length):
        cross_entropy_bits = stacked_buffer_info[i::list_length]
        cross_entropy_labels = stacked_buffer_label[i::list_length]
        CE_loss = calculate_loss(cross_entropy_bits,cross_entropy_labels)
        CE_loss_list.append(round((CE_loss/cross_entropy_bits.shape[0]),3))
    # Notation text
    notation_text = "CElist:"
    with open(log_filename,'a+') as f:
      f.write(str(cross_entropy_bits.shape[0])+'tested:\n')
      f.write("# " + notation_text + '\n')
      f.write(' '.join(map(str, CE_loss_list)) + '\n')  # Join list elements with space as delimiter
    print(f'{cross_entropy_bits.shape[0]} tested \nCE_list:{CE_loss_list}')
    print(f'Data for retraining  with {stacked_buffer_info.shape[0]} cases to be stored!')
    data = (stacked_buffer_info.numpy(),stacked_buffer_label.numpy())
    Data_gen.make_tfrecord(data, out_filename=dir_file)    
    print(f'For{snr}dB:Data storing finished!')

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

# Function to normalize a vector while preserving signs
def normalize_with_signs(v):
    return np.sign(v) * (np.abs(v) / np.linalg.norm(v))

#postprocessing after first stage training
def generate_parity_pass_samples(Model,iterator):
    #collecting erroneous decoding info
    buffer_inputs = []
    buffer_labels = []
    data_labels = []
    #query of size of input feedings
    input_list = list(iterator.as_numpy_iterator())
    num_counter = len(input_list) 
    for i in range(num_counter):
        if not (i+1) % 100:
            print("Total ",i+1," batches are processed!")
        inputs = input_list[i]
        soft_output_list,_,label = Model(inputs)
        pair,pair_label,data_label,_ = Model.create_samples(soft_output_list,label)
        buffer_inputs.append( pair)
        buffer_labels.append(pair_label)
        data_labels.append(data_label)
    sample_matrix = tf.concat(buffer_inputs,axis=0)
    data_labels_matrix = tf.concat(data_labels,axis=0)
    labels_vector =tf.cast(tf.concat(buffer_labels,axis=0),tf.float32)
    feature_matrix = tf.concat([sample_matrix,labels_vector],1)
    return feature_matrix,data_labels_matrix
#postprocessing after first stage training
def generate_distance_model_samples(Model,iterator):
    #collecting erroneous decoding info
    buffer_inputs = []
    buffer_labels = []
    data_labels = []
    #query of size of input feedings
    input_list = list(iterator.as_numpy_iterator())
    num_counter = len(input_list) 
    for i in range(num_counter):
        if not (i+1) % 100:
            print("Total ",i+1," batches are processed!")
        inputs = input_list[i]
        soft_output_list,_,label = Model(inputs)
        pair,pair_label,data_label,_ = Model.create_samples(soft_output_list,label)
        buffer_inputs.append( pair)
        buffer_labels.append(pair_label)
        data_labels.append(data_label)
    sample_matrix = tf.concat(buffer_inputs,axis=0)
    data_labels_matrix = tf.concat(data_labels,axis=0)
    labels_vector = tf.concat(buffer_labels,axis=0) 
    return sample_matrix,labels_vector,data_labels_matrix

def postprocess_statistics(Model,iterator):
    #query of size of input feedings
    input_list = list(iterator.as_numpy_iterator())
    num_counter = len(input_list) 
    FER_sum = 0.
    for i in range(num_counter):
        print('.',end='')
        inputs = input_list[i]
        soft_output_list,_,label = Model(inputs)
        #print(Model.trainable_variables)
        FER = Model.get_eval(soft_output_list,label)
        FER_sum += FER
        #tf.print(FER,Model.trainable_variables)
        if not (i+1) % 10:
            print("Total ",i+1," batches are processed!")
            print(f'FER:{FER_sum/(i+1):.4f}')
    return FER_sum/num_counter,num_counter
#postprocessing after first stage training
def save_decoded_data(buffer_inputs,buffer_labels,file_dir,snr):
    #code = GL.get_map('code_parameters')
    stacked_buffer_info = tf.stack(buffer_inputs)
    stacked_buffer_label = tf.stack(buffer_labels)
    print("Data for retraining  with %d cases to be stored " % stacked_buffer_info.shape[0])
    (features, labels) = (stacked_buffer_info.numpy(),stacked_buffer_label.numpy()) 
    data = (features,labels)
    Data_gen.make_tfrecord(data,out_filename=file_dir)

def postprocess_testing(Model_NMS,iterator):
    code = GL.get_map('code_parameters')
    #collecting erroneous decoding info
    buffer_inputs = []
    buffer_labels = []
    #query of size of input feedings
    input_list = list(iterator.as_numpy_iterator())
    num_counter = len(input_list) 
    FER_sum = 0
    BER_sum = 0
    num_samples = 0
    iteration_counter = Counter()
    for i in range(num_counter):
        if (i+1) % 100 == 0:
            print("Total ",i+1," batches are processed!")
            print(f'FER:{FER_sum/num_samples:.4f} BER:{BER_sum/(num_samples*code.n):.4f}')
        inputs = input_list[i]
        soft_output_list,label = Model_NMS(inputs)
        #FER_count,BER_count,delare_n_index = Model_NMS.get_eval(soft_output_list,label)     
        FER_count,BER_count,delare_n_index,iteration_counter = Model_NMS.get_eval_expanded(soft_output_list,label,iteration_counter)     
        buffer_inputs_tmp,buffer_labels_tmp = Model_NMS.collect_failed_input_output(soft_output_list,label,delare_n_index)   
        buffer_inputs.append(buffer_inputs_tmp)
        buffer_labels.append(buffer_labels_tmp)
        num_samples += label.shape[0]
        FER_sum += FER_count
        BER_sum += BER_count
        if FER_sum > GL.get_map('decoding_threshold'):
          break
    buffer_inputs = [j for i in buffer_inputs for j in i]
    buffer_labels = [j for i in buffer_labels for j in i]
    return buffer_inputs,buffer_labels,FER_sum/num_samples,BER_sum/(num_samples*code.n),num_samples,iteration_counter

def print_iterations_ratios(iteration_counter,snr):
    num_iterations = GL.get_map('num_iterations')
    # Calculate what proportion of total items each iteration handled
    total_items = sum(iteration_counter.values())
    iteration_proportions = {it_index: count / total_items 
                             for it_index, count in iteration_counter.items()} 
    total_iterations = 0
    for it_index, count in iteration_counter.items():
        if it_index == -1:
            it_cost = num_iterations
        else:
            it_cost = it_index
        total_iterations += it_cost*count 
    average_iterations = total_iterations/total_items
    print(f"Iteration proportions at {snr}dB:")
    ratio_list = []
    for it_index in sorted(iteration_proportions.keys(), key=lambda x: (x == -1, x)):
        prop = iteration_proportions[it_index]    
        print(f"  {it_index}: {prop:.4f} ({prop*100:.1f}%)")
        if it_index == -1:
            ratio_list[-1][1] = round(ratio_list[-1][1]+prop,4)
        else:
            ratio_list.append([it_index,round(prop,4)])
        
    return ratio_list,average_iterations

def plot_and_save_multiple_snr_proportions(SNRs,average_iterations_list,proportion_list, title="Iteration Proportions Comparison", 
                                          save_path=None, filename="iteration_proportions_comparison.png"):
    # Create the plot
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    fig, ax1 = plt.subplots(figsize=(10, 7))  
    ax2 = ax1.twinx()  # Create secondary y-axis for average iterations
    # Get all unique iteration keys across all SNR data
    sorted_keys = [proportion_list[0][i][0] for i in range(len( proportion_list[0]))]
 
    labels = [f'Iter {k}' for k in sorted_keys]
    
    # Create colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_keys)))
    
    # Plot each SNR
    snr_values = [round(SNRs[i],2) for i in range(len(SNRs))]
    x_positions = np.arange(len(snr_values))/4
    width = 0.1
    
    for i, snr in enumerate(snr_values):
        counter = proportion_list[i]
        bottom = 0      
        for j,key in enumerate(sorted_keys):
            prop = counter[key][1] 
            ax2.bar(x_positions[i], prop, width, bottom=bottom, 
                  color=colors[j], label=labels[j] if i == 0 else "", 
                  alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add percentage labels for significant segments
            if prop > 0.1:  # Only label segments > 5%
                ax2.text(x_positions[i], bottom + prop/2, f'{prop:.0%}', 
                       ha='center', va='center', fontsize=14,
                       color='black')
            bottom += prop
    # Plot average iterations line on secondary axis
    ax1.plot(x_positions, average_iterations_list, 'ro-', linewidth=3, markersize=8, 
                   label='Avg Iterations', markerfacecolor='red', markeredgecolor='darkred')    
    # Customize the primary axis (proportions)
    ax1.set_xlabel('$E_b/N_0$ (dB)', fontsize=16)
    #ax2.set_ylabel('Proportion', fontsize=16, color='blue')
    #ax2.set_title(title, fontsize=16, fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f'{snr}' for snr in snr_values], fontsize=16)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelcolor='blue')
    # Customize the secondary axis (average iterations)
    ax1.set_ylabel('Average Number of Iterations', fontsize=16, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Set appropriate y-limits for average iterations
    ax1.set_ylim(sorted_keys[0],sorted_keys[-1])    
    # Add legend
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Iterations', fontsize=14)
    # Add grid for better readability
    ax1.grid(True, alpha=0.5, axis='y', linestyle='-')
    
    # Adjust layout to make room for legend
    plt.tight_layout()
    
    # Save the figure
    if save_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, filename)
    else:
        full_path = filename
    
    # Save in high resolution for publications/reports
    plt.savefig(full_path, dpi=1000, bbox_inches='tight', facecolor='white')
    print(f"Chart saved to: {full_path}")
    print(f"Average iterations per SNR: {dict(zip(snr_values, average_iterations_list))}")
    plt.show()
    
    return full_path
        
def post_process4_dia_samples(snr,NMS_model):
    output_dir,test_dataset = GL.data_setting(snr) 
    if GL.get_map('ALL_ZEROS_CODEWORD_TESTING'):
        output_file_name = 'retest-allzero.tfrecord'
    else:
        output_file_name = 'retest-nonzero.tfrecord'
    # Define the file path
    retest_dir_file = output_dir+output_file_name
    print('\n')
    #save trajectories of NMS failed cases
    buffer_list_data,buffer_list_label,average_FER,average_BER,num_samples,iteration_counter = postprocess_testing(NMS_model,test_dataset)
    ratio_list,average_iterations = print_iterations_ratios(iteration_counter,snr)

    metric_list = [average_FER,average_BER,num_samples,ratio_list,average_iterations]
    print(f'FER:{average_FER:.4f} BER:{average_BER:.4f} Num_samples:{num_samples}')   
    #save training samples which pass parity-checks of related code
    save_decoded_data(buffer_list_data,buffer_list_label,retest_dir_file,snr)  
    print(f"Collecting at {snr}dB is finished!")
    return metric_list
  