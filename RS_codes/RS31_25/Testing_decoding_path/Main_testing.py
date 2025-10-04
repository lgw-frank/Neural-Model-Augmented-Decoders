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
import sys,os
import globalmap as GL
import optimized_decoding_path as Opt_path
import pickle
import nn_net as NN_struct


sys.argv = "python 2.0 4.5 6 100 100 8 Hopt_RS_31_25_3_1440ones.alist NMS-1".split()
#sys.argv = "python 2.8 4.0 25 8000 10 wimax_1056_0.83.alist ANMS".split()
#setting a batch of global parameters
GL.global_setting(sys.argv)
#initial setting for restoring model
unit_batch_size = GL.get_map('unit_batch_size')
code = GL.get_map('code_parameters')
# Example usage
print_interval = GL.get_map('print_interval')
record_interval = GL.get_map('record_interval')
#train_snr = GL.get_map('train_snr')
plot_low_limit = GL.get_map('plot_low_limit')


os.makedirs("./figs", exist_ok=True) #outputed figures location
log_dir = './log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

intercept_length = GL.get_map('intercept_length')
relax_factor = GL.get_map('relax_factor')

if GL.get_map('extended_input'):
    suffix = '-extended'
else:
    suffix = ''
if GL.get_map('DIA_deployment'):
    ending = '-dia'
else:
    ending = ''
output_ranking_dir = '../Optimizing_decoding_path/ckpts/ranking_patterns/'
output_order_file = f'{output_ranking_dir}intercept{intercept_length}-relax{relax_factor}-ranking_orders{suffix}{ending}.pkl'

with open(output_order_file,'rb') as f:
    full_teps_ordering_list = pickle.load(f)

total_teps_list = [full_teps_ordering_list[i][:intercept_length] for i in range(len(full_teps_ordering_list))]
if not os.path.exists('./ckpts'):
    os.makedirs('./ckpts')
output_saved_data_points = f'./ckpts/varied_snrs_data{suffix}{ending}'
DIA_model = NN_struct.conv_bitwise()
if GL.get_map('DIA_deployment'):
    #load DIA model
    restore_info = GL.logistic_setting_model()
    DIA_model = Opt_path.NN_gen(DIA_model, restore_info,DIA_model.actual_length) 
if GL.get_map('regenerate_data_points'):
    selected_ds_list,snr_list = GL.data_setting()
    snr_data_list = []
    for i in range(len(snr_list)):
          error_pattern_batch_list = Opt_path.find_error_pattern(DIA_model,selected_ds_list[i])
          data_counter_list = []
          for sorted_order in total_teps_list:
              point_tuple = Opt_path.summary_ranking(sorted_order,error_pattern_batch_list)
              data_counter_list.append(point_tuple)
          snr_data_list.append(data_counter_list)
    #save acquired data in file
    with open(output_saved_data_points, 'wb') as f:
          pickle.dump(snr_list, f)
          pickle.dump(snr_data_list, f)
          print("Saved data with varied SNRs")
else:
    #open acquired data in file
    with open(output_saved_data_points, 'rb') as f:
          snr_list = pickle.load(f)
          snr_data_list = pickle.load(f)
          print("Load data with varied SNRs")
        
for i in range(len(snr_list)):
    snr = round(snr_list[i],2)
    data_counter_list = snr_data_list[i]      
    saved_path = f'./figs/ror_intercept{intercept_length}-relax{relax_factor}-{snr}dB{suffix}{ending}.png'
    Opt_path.drawing_plot(snr,data_counter_list,plot_low_limit,saved_path)
    rank_sms = [data_counter_list[j][1] for j in range(len(total_teps_list))]
    size_list = [data_counter_list[j][2] for j in range(len(total_teps_list))]
    avr_ranks = [round(rank_sms[j]/size_list[j],2) for j in range(len(total_teps_list))]
    print(f'SNR {snr:.2f}dB Size={size_list}:')
    print(f'Rank_sum:{rank_sms}')
    print(f'avr_sum:{avr_ranks}')
    file_name = log_dir + f'TEPs_summary{suffix}.txt'
    with open(file_name,'a+') as f:
        f.write(f'SNR {snr:.2f}dB{ending}:\n')
        f.write(f'avr_sum:{avr_ranks}\n')

T2 =time.time()
print('Running time:%s seconds!'%(T2 - T1))