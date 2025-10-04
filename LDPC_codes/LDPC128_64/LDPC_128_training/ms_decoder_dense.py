# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:41:33 2022

@author: Administrator
"""
import tensorflow as tf
import globalmap as GL
from tensorflow.keras.layers import Dense     # for the hidden layer
import data_generating as Data_gen
from keras.constraints import non_neg
import numpy as np
import math
#from distfit import distfit

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
        soft_output_list,label,loss = self.decoder_layer(inputs)
        return soft_output_list,label,loss 
    
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
    
    def get_eval(self,soft_output_list,labels):
        code = GL.get_map('code_parameters')
        labels = tf.cast(labels,tf.int64)
        H = code.H
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
    def call(self, inputs):
        # VERIFICATION: Ensure weight persists
        if not hasattr(self, 'decoder_check_normalizor'):
            raise RuntimeError("Weight lost during execution!")
        soft_input = inputs[0]
        labels = inputs[1]    
        outputs = self.belief_propagation_op(soft_input, labels)
        soft_output_array, label, loss = outputs
        return soft_output_array.stack(), label, loss  # ✅ stack before returning
             
    def belief_propagation_op(self, soft_input, labels):
        soft_output_array = tf.TensorArray(
            dtype=tf.float32,
            size=self.num_iterations + 1,
            clear_after_read=False  # <-- Required if you want to read an index multiple times
        )
        # Write initial value
        soft_output_array = soft_output_array.write(0, soft_input)
        cv_matrix = tf.zeros([soft_input.shape[0],self.code.check_matrix_row,self.code.check_matrix_column],dtype=tf.float32)# cv_matrix
    
        def condition(soft_output_array, cv_matrix,labels,loss, iteration):
            return iteration < self.num_iterations
        def body(soft_output_array, cv_matrix,labels,loss, iteration):
            vc_matrix = self.compute_vc(soft_output_array, cv_matrix)
            # compute cv
            cv_matrix = self.compute_cv(vc_matrix)      
            # get output for this iteration
            soft_output_array = self.marginalize(soft_output_array,cv_matrix,iteration)  
            iteration += 1   
            soft_output = soft_output_array.read(iteration)
            loss = self.calculation_loss(soft_output,labels,loss)
            return soft_output_array,cv_matrix,labels,loss,iteration
    
        soft_output_array, cv_matrix,labels,loss,iteration = tf.while_loop(
            condition,
            body,
            loop_vars=[soft_output_array, cv_matrix,labels,0., 0]
        )   
        return soft_output_array, labels,loss
            
    # compute messages from variable nodes to check nodes
    def compute_vc(self,soft_output_array, cv_matrix):
        soft_input = soft_output_array.read(0)
        check_matrix_H = tf.cast(self.code.H,tf.float32)                    
        temp = tf.reduce_sum(cv_matrix,1)                        
        temp += soft_input
        temp = tf.expand_dims(temp,1)
        temp = temp*check_matrix_H
        vc_matrix = temp - cv_matrix
        return vc_matrix 
       
    # compute messages from check nodes to variable nodes
    def compute_cv(self,vc_matrix):
        normalized_tensor = 1.0
        check_matrix_H = self.code.H
        #operands sign processing 
        supplement_matrix = tf.cast(1-check_matrix_H,dtype=tf.float32)
        supplement_matrix = tf.expand_dims(supplement_matrix,0)
        sign_info = supplement_matrix + vc_matrix
        vc_matrix_sign = tf.sign(sign_info)
        temp1 = tf.reduce_prod(vc_matrix_sign,2)
        temp1 = tf.expand_dims(temp1,2)
        transition_sign_matrix = temp1*check_matrix_H
        result_sign_matrix = transition_sign_matrix*vc_matrix_sign 
        #preprocessing data for later calling of top k=2 largest items
        back_matrix = tf.where(check_matrix_H==0,-1e30-1,0.)
        back_matrix = tf.expand_dims(back_matrix,0)
        vc_matrix_abs = tf.abs(vc_matrix)
        vc_matrix_abs_clip = tf.clip_by_value(vc_matrix_abs, 0, 1e30)
        vc_matrix_abs_minus = -tf.abs(vc_matrix_abs_clip)
        vc_decision_matrix = vc_matrix_abs_minus+back_matrix
        min_submin_info = tf.nn.top_k(vc_decision_matrix,k=2)
        min_column_matrix = -min_submin_info[0][:,:,0]
        min_column_matrix = tf.expand_dims(min_column_matrix,2)
        min_column_matrix = min_column_matrix*check_matrix_H
        second_column_matrix = -min_submin_info[0][:,:,1]
        second_column_matrix = tf.expand_dims(second_column_matrix,2)
        second_column_matrix = second_column_matrix*check_matrix_H  
        result_matrix = tf.where(vc_matrix_abs_clip>min_column_matrix,min_column_matrix,second_column_matrix)
        if GL.get_map('selected_decoder_type') in ['NMS-1']:
            normalized_tensor = tf.nn.softplus(self.decoder_check_normalizor)       
        cv_matrix = normalized_tensor *result_matrix*tf.stop_gradient(result_sign_matrix)         
        return cv_matrix
    
    def calculation_loss(self,soft_output,labels,loss):
         #cross entroy
        labels = tf.cast(labels,tf.float32)
        CE_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels)) 
        return CE_loss+loss   
    
    #combine messages to get posterior LLRs
    def marginalize(self,soft_output_array,cv_matrix,iteration):
        soft_input = soft_output_array.read(0)
        temp = tf.reduce_sum(cv_matrix,1)
        soft_output = soft_input+temp
        soft_output_array = soft_output_array.write(iteration + 1, soft_output)
        return soft_output_array   
                  

def retore_saved_model(restore_ckpts_dir,restore_step,ckpt_nm):
    print("Ready to restore a saved latest or designated model!")
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
    return start_step,ckpt_f

#save modified data for postprocessing
def save_decoded_data(buffer_inputs,buffer_labels,dir_file):
    #code = GL.get_map('code_parameters')
    stacked_buffer_info = tf.stack(buffer_inputs)
    stacked_buffer_label = tf.stack(buffer_labels)
    print(" Data for retraining  with %d cases to be stored " % stacked_buffer_info.shape[0])
    data = (stacked_buffer_info.numpy(),stacked_buffer_label.numpy())
    Data_gen.make_tfrecord(data, out_filename=dir_file)    
    print("Data storing finished!")


#postprocessing after first stage training
def postprocess_training(Model_NMS, iterator):
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
    for i in range(num_counter):
        if (i+1) % 100 == 0:
            print("Total ",i+1," batches are processed!")
            print(f'FER:{FER_sum/num_samples:.4f} BER:{BER_sum/(num_samples*code.check_matrix_column):.4f}')
        inputs = input_list[i]
        soft_output_list,label,_ = Model_NMS(inputs)
        FER_count,BER_count,delare_n_index = Model_NMS.get_eval(soft_output_list,label)     
        buffer_inputs_tmp,buffer_labels_tmp = Model_NMS.collect_failed_input_output(soft_output_list,label,delare_n_index)   
        buffer_inputs.append(buffer_inputs_tmp)
        buffer_labels.append(buffer_labels_tmp)
        num_samples += label.shape[0]
        FER_sum += FER_count
        BER_sum += BER_count
    buffer_inputs = [j for i in buffer_inputs for j in i]
    buffer_labels = [j for i in buffer_labels for j in i]
    return buffer_inputs,buffer_labels,FER_sum/num_samples,BER_sum/(num_samples*code.check_matrix_column),num_samples
    
#main training process
def training_block(batch_index, Model, optimizer, exponential_decay, selected_ds, log_info, restore_info):
    print_interval = GL.get_map('print_interval')
    record_interval = GL.get_map('record_interval')
    termination_step = GL.get_map('nms_termination_step')
    summary_writer, manager_current = log_info
    ckpts_dir, ckpt_nm, ckpts_dir_par, restore_step = restore_info
    ds_iter = iter(selected_ds)
    while True:
        if batch_index >= termination_step:
            break
        try:
            inputs = next(ds_iter)
        except StopIteration:
            # In case dataset is finite and exhausted
            print("⚠️ Dataset exhausted. Consider adding `.repeat()` in dataset pipeline.")
            break
        with tf.GradientTape() as tape:
            soft_output_list, label, loss = Model(inputs)
        fer, ber, _ = Model.get_eval(soft_output_list, label)
        grads = tape.gradient(loss, Model.trainable_variables)
        grads_and_vars = [(tf.clip_by_norm(grad, 5e2), var)
                          for grad, var in zip(grads, Model.trainable_variables)
                          if grad is not None]
        optimizer.apply_gradients(grads_and_vars)     
        batch_index += 1
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=batch_index)
            tf.summary.scalar("FER", fer, step=batch_index)
            tf.summary.scalar("BER", ber, step=batch_index)
        if batch_index % print_interval == 0:
            manager_current.save(checkpoint_number=batch_index)
            print("Parameter evaluation:")
            for weight in  Model.get_weights():
              print(weight)
            tf.print("Step%4d: Lr:%.3f Ls:%.1f FER:%.3f BER:%.4f"%\
            (batch_index,exponential_decay(batch_index),loss, fer/label.shape[0], ber/(label.shape[0]*label.shape[1]) )) 
        if batch_index % record_interval == 0:
            with open(ckpts_dir_par + 'values.txt', 'a+') as f:
                f.write("For all layers at %4d-th step:\n" % batch_index)
                for variable in Model.variables:
                    f.write(variable.name + ' ' + str(variable.numpy()) + '\n') 
    inputs = next(ds_iter)
    _ = Model(inputs)        #in case of loading model from file, to activate model.
    print("Final selected parameters:")
    for weight in  Model.get_weights():
      print(weight)
    return Model        