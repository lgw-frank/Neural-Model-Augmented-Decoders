# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 22:47:53 2025

@author: lgw
"""
# -*- coding: utf-8 -*-
import time
T1 = time.time()
import numpy as np
np.set_printoptions(precision=3)
#import matplotlib
import sys
import globalmap as GL
import optimized_decoding_path as Opt_path
import pickle,os
#from datetime import datetime

sys.argv = "python 3.0 3.0 100 100 10 CCSDS_ldpc_n128_k64.alist NMS-1".split()
GL.global_setting(sys.argv)
#initial setting for restoring model
unit_batch_size = GL.get_map('unit_batch_size')
code = GL.get_map('code_parameters')
draw_low_limit = GL.get_map('draw_low_limit')
# Example usage
print_interval = GL.get_map('print_interval')
record_interval = GL.get_map('record_interval')
decoding_length = GL.get_map('num_iterations')+1
intercept_length = GL.get_map('intercept_length')
relax_factor = GL.get_map('relax_factor')
macro_length = intercept_length*relax_factor
#create the directory if not existing
output_ranking_dir = './ckpts/ranking_patterns/'
if not os.path.exists(output_ranking_dir):
    os.makedirs(output_ranking_dir)
os.makedirs("./figs", exist_ok=True) #outputed figures location
if GL.get_map('extended_input'):
    suffix = '-extended'
else:
    suffix = ''
if GL.get_map('DIA_deployment'):
    ending = '-dia'
else:
    ending = ''
output_order_file = f'{output_ranking_dir}intercept{intercept_length}-relax{relax_factor}-ranking_orders{ending}{suffix}.pkl'
print(f'ordered_file:{output_order_file}')
snr_lo = round(GL.get_map('snr_lo'), 1)
snr_hi = round(GL.get_map('snr_hi'), 1)

if GL.get_map('create_update_teps_ranking'):                 
    #initializing test error patterns ordering
    macro_convention_teps_ordering = Opt_path.generate_error_patterns(code.k)
    convention_teps_ordering = macro_convention_teps_ordering[:intercept_length] #truncate the outbounded length 
    macro_ALMLT_teps_ordering = Opt_path.ALMLT_ranking(snr_lo,macro_length)
    ALMLT_teps_ordering = macro_ALMLT_teps_ordering[:intercept_length]
    #training data
    training_data_iterator = GL.data_iteration(code, unit_batch_size)
    error_pattern_batch_list = Opt_path.find_error_pattern(training_data_iterator)
    sorted_train_teps = Opt_path.sort_rows_by_occurrences(error_pattern_batch_list)
    print(f'Shape:{len(sorted_train_teps)}')
    #updating by optmization
    # manager_convention = Opt_path.ErrorPatternManager(convention_teps_ordering)
    # manager_convention.iterate_optimize(error_pattern_batch_list)
    
    # manager_ALMLT = Opt_path.ErrorPatternManager(ALMLT_teps_ordering) 
    # manager_ALMLT.iterate_optimize(error_pattern_batch_list)
    
    macro_manager_convention = Opt_path.ErrorPatternManager(macro_convention_teps_ordering)
    macro_manager_convention.one_step_optimize(error_pattern_batch_list)
    
    macro_manager_ALMLT = Opt_path.ErrorPatternManager(macro_ALMLT_teps_ordering)
    macro_manager_ALMLT.one_step_optimize(error_pattern_batch_list)
    ordering_list = [sorted_train_teps,convention_teps_ordering,ALMLT_teps_ordering,\
                     macro_manager_convention.sorted_patterns[:intercept_length],macro_manager_ALMLT.sorted_patterns[:intercept_length]]    
    # ordering_list = [sorted_train_teps,convention_teps_ordering,ALMLT_teps_ordering,\
    #                  macro_manager_convention.sorted_patterns[:intercept_length],macro_manager_ALMLT.sorted_patterns[:intercept_length],\
    #                  manager_convention.sorted_patterns,manager_ALMLT.sorted_patterns]
    manager_list = [macro_manager_convention,macro_manager_ALMLT]
    #manager_list = [macro_manager_convention,macro_manager_ALMLT,manager_convention,manager_ALMLT]
    with open(output_order_file, 'wb') as f:
        pickle.dump(ordering_list, f)
        pickle.dump(manager_list,f)
        pickle.dump(error_pattern_batch_list,f)
        
else:
    with open(output_order_file, 'rb') as f:     
        ordering_list = pickle.load(f)
        manager_list = pickle.load(f)
        error_pattern_batch_list = pickle.load(f)
    #extension processing
    if GL.get_map('extended_input'):
        training_data_iterator = GL.data_iteration(code, unit_batch_size)
        error_pattern_batch_list_extended = Opt_path.find_error_pattern(training_data_iterator)
        #merged training data by squashing them
        error_pattern_batch_list = [ [row] for matrix in error_pattern_batch_list_extended + error_pattern_batch_list for row in matrix ]
        sorted_train_teps = Opt_path.sort_rows_by_occurrences(error_pattern_batch_list)
        manager_list[0].one_step_optimize(error_pattern_batch_list)
        manager_list[1].one_step_optimize(error_pattern_batch_list)
        manager_list[2].iterate_optimize(error_pattern_batch_list)
        manager_list[3].iterate_optimize(error_pattern_batch_list)      
        ordering_list = [sorted_train_teps,ordering_list[1],ordering_list[2],\
                         manager_list[0].sorted_patterns[:intercept_length],manager_list[1].sorted_patterns[:intercept_length],\
                         manager_list[2].sorted_patterns,manager_list[3].sorted_patterns]
        print([len(ordering_list[i]) for i in range(len(ordering_list))])
        with open(output_order_file, 'wb') as f:
            pickle.dump(ordering_list, f)
            pickle.dump(manager_list,f)
            pickle.dump(error_pattern_batch_list,f)  

data_counter_list = []
for sorted_order in ordering_list:
    point_tuple = Opt_path.summary_ranking(sorted_order,error_pattern_batch_list)
    data_counter_list.append(point_tuple)
# Create timestamp: YYYYMMDD_HHMM

#timestamp = datetime.now().strftime("%Y%m%d_%H%M") 
Opt_path.drawing_plot(snr_lo,data_counter_list,draw_low_limit,save_path=f"./figs/ror_intercept{intercept_length}{suffix}{ending}_{snr_lo}dB.png")
    
T2 =time.time()
print('Running time:%s seconds!'%(T2 - T1))