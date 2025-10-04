import itertools,math,pickle
import numpy as np
import globalmap as GL
import ordered_statistics_decoding as Osd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
from scipy.special import erf
from scipy.optimize import fsolve
from scipy.special import comb
import os,sys
def append_unique_rows_vectorized(a, b):
    a = tf.stack(a)
    b = tf.stack(b)
    # Compare all rows of b with all rows of a
    # Reshape for broadcasting
    a_expanded = tf.expand_dims(a, 1)  # shape [a_rows, 1, cols]
    b_expanded = tf.expand_dims(b, 0)  # shape [1, b_rows, cols]
    
    # Compare all pairs
    matches = tf.reduce_all(tf.equal(a_expanded, b_expanded), axis=2)
    
    # Find rows in b that don't match any row in a
    is_unique = tf.logical_not(tf.reduce_any(matches, axis=0))
    unique_rows = tf.boolean_mask(b, is_unique)
    
    # Concatenate if there are unique rows
    return tf.cond(tf.shape(unique_rows)[0] > 0,
                  lambda: tf.concat([a, unique_rows], axis=0),
                  lambda: a)
def F_lambda_i(x, sigma):
    if x < 0:
        return 0
    else:
        return 0.5 * (erf((x - 1) / (np.sqrt(2) * sigma)) + erf((x + 1) / (np.sqrt(2) * sigma)))

def inverse_F_lambda_i(y, sigma):
    # Define a function to find the root of
    def equation(x):
        return F_lambda_i(x, sigma) - y
    
    # Initial guess for the root
    x0 = 1.0  # This is a starting point; may need adjustment based on the specific case
    
    # Use fsolve to find the root
    x = fsolve(equation, x0)
    
    return x[0]

def theoretical_ordered_statistics_mean(snr):
    code = GL.get_map('code_parameters')
    sigma =  np.sqrt(1. / (2 * (float(code.k)/float(code.n)) * 10**(snr/10)))
    inverse_list = []
    for i in range(1,code.n+1):
        y_value = i/(code.n+1)
        inverse_value = inverse_F_lambda_i(y_value, sigma)
        #print(f"Inverse F_lambda_i({y_value}) = {inverse_value}")
        inverse_list.append(inverse_value)
    return inverse_list

def sum_of_combinations(n, p):
    """
    Compute the sum of combinations C(n, 0) + C(n, 1) + ... + C(n, p).
    
    Args:
        n (int): Total number of distinct balls.
        p (int): Maximum number of balls to draw.
    
    Returns:
        int: Sum of combinations.
    """
    return sum(int(comb(n, k)) for k in range(p + 1))

def ALMLT_ranking(snr,required_length):
    code = GL.get_map('code_parameters')
    theory_ordered_mean = theoretical_ordered_statistics_mean(snr)
    top_sequences = get_top_sequences_dp(theory_ordered_mean[-code.k:])
    return top_sequences[:required_length]
    
def get_top_sequences_dp(sorted_reliability):
    code = GL.get_map('code_parameters')
    order_p = GL.get_map('max_order_p')
    candidate_length = sum_of_combinations(code.k,order_p)
    # Sort reliability measurements in ascending order
    #sorted_reliability = sorted(reliability)

    # DP table: dp[k] will store the top M sequences for the first k bits
    dp = [[] for _ in range(code.k + 1)]
    dp[0] = [('', 0)]  # Base case: empty sequence with measurement 0

    for index_k in range(1, code.k + 1):
        r = sorted_reliability[index_k - 1]
        new_sequences = []
        for seq, meas in dp[index_k - 1]:
            # Add 0 to the sequence (measurement remains the same)
            new_sequences.append((seq + '0', meas))
            # Add 1 to the sequence (measurement increases by r)
            new_sequences.append((seq + '1', meas + r))

        # Sort new sequences by measurement and keep the top M
        new_sequences.sort(key=lambda x: x[1])
        dp[index_k] = new_sequences[:candidate_length]

    # Extract the top M sequences from the final DP table
    # Transform the string into a list of binary digits
    top_sequences = [tuple(list(map(int, seq))) for seq, meas in dp[code.k]]
    top_sequences
    return top_sequences     


def retrieve_saved_model(restore_info):
    restore_ckpts_dir,ckpt_nm,restore_step = restore_info
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

def sort_rows_by_occurrences(error_pattern_batch_list):
    # Step 1: Concatenate all batches into a single matrix
    error_pattern_batch_matrix = np.concatenate(error_pattern_batch_list, axis=0) 
    # Step 2: Convert each row to an integer and store it with the original row
    row_with_integers = [(row, int(''.join(map(str, row)))) for row in error_pattern_batch_matrix]
    # Step 3: Count the occurrences of each integer
    counts = Counter(int_val for _, int_val in row_with_integers) 
    # Step 4: Sort the rows based on the counts in descending order
    # Use the counts of the integer representation as the key for sorting
    sorted_rows = [row for row, int_val in sorted(row_with_integers, key=lambda x: counts[x[1]], reverse=True)]
    # Step 5: Remove duplicate rows while preserving order
    unique_sorted_rows = []
    seen = set()
    for row in sorted_rows:
        row_tuple = tuple(row)  # Convert row to a tuple (hashable) for use in a set
        if row_tuple not in seen:
            unique_sorted_rows.append(row)
            seen.add(row_tuple)
    return unique_sorted_rows

def compelement_patterns(error_pattern_batch_list,convention_teps_ordering):
    intercept_length = GL.get_map('intercept_length')
    sorted_train_teps= sort_rows_by_occurrences(error_pattern_batch_list)
    print(f'original shape:({len(sorted_train_teps)},{len(sorted_train_teps[0])})')
    relaxed_sorted_train_teps_matrix = append_unique_rows_vectorized(sorted_train_teps,convention_teps_ordering)   
    # Convert to list of row lists
    relaxed_sorted_train_teps = relaxed_sorted_train_teps_matrix.numpy().tolist()
    print(f'relaxed_ shape:{len(relaxed_sorted_train_teps)}')    
    return relaxed_sorted_train_teps[:intercept_length]

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
    
def NN_gen(model,restore_info,observe_length=100,input_dim=3): 
    # Explicitly build the model with dummy input
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
        ckpt_f = retrieve_saved_model(restore_info)
        status = checkpoint.restore(ckpt_f)
        weights = model.get_weights()
        rounded_weights = format_weights(weights, decimals=2)
        print("\nPost-restoration weights:\n",rounded_weights)
        #status.assert_existing_objects_matched()
        status.expect_partial()
    return model 
     
def find_error_pattern(DIA_model,input_iterator):
    code = GL.get_map('code_parameters')
    #query authentic error patterns
    osd_model = Osd.osd_light(code)
    data_list =list(input_iterator) 
    if GL.get_map('DIA_deployment'):
        error_pattern_batch_list = query_error_patterns_dia(osd_model,DIA_model,data_list)
    else:
        error_pattern_batch_list = query_error_patterns(osd_model,data_list)

    return error_pattern_batch_list

#query authentic error patterns
def query_error_patterns_dia(osd_model,DIA_model,data_list):
    code = GL.get_map('code_parameters')
    error_pattern_batch_list = []
    for batch in data_list:
        print('. ',end='')
        squashed_inputs,labels,super_input_matrix = DIA_model.preprocessing_inputs(batch)
        dia_outputs = tf.reshape(DIA_model(squashed_inputs),[-1,code.n])  
        dia_outputs += super_input_matrix[0]
        element = osd_model.convention_osd_preprocess(dia_outputs,labels)
        error_pattern_batch_list.append(element)
    return error_pattern_batch_list
    
#query authentic error patterns
def query_error_patterns(osd_model,data_list):
    list_length = GL.get_map('num_iterations')+1
    error_pattern_batch_list = []
    for batch in data_list:
        print('. ',end='')
        inputs,labels = batch[0][::list_length],batch[1][::list_length]
        element = osd_model.convention_osd_preprocess(inputs,labels)
        error_pattern_batch_list.append(element)
    return error_pattern_batch_list
# Function to create batches from the data
def create_batches(error_pattern_batch_list, batch_size,shuffle_indicator=True):
    error_pattern_batch_matrix = np.concatenate(error_pattern_batch_list,axis=0)
    if shuffle_indicator:
        np.random.shuffle(error_pattern_batch_matrix)  # Shuffles in-place
    batch_length = math.ceil(error_pattern_batch_matrix.shape[0]/batch_size)
    batches = []
    for i in range(batch_length):
        element = error_pattern_batch_matrix[i*batch_size:(i+1)*batch_size]
        batches.append(element)
    return batches

def generate_error_patterns(n, p):
    """
    Generate all binary sequences of length n with Hamming weight at most p.
    Each sequence is represented as a list of integers (0 or 1).

    Args:
        n (int): Length of the binary sequence.
        p (int): Maximum Hamming weight (number of 1s).

    Returns:
        list: List of error patterns, where each pattern is a list of integers.
    """
    error_patterns = []
    # Generate all combinations of positions for 1s
    for k in range(p + 1):  # k is the number of 1s (Hamming weight)
        for positions in itertools.combinations(range(n), k):
            # Create a list of integers representing the binary sequence
            sequence = [0] * n
            for pos in positions:
                sequence[pos] = 1
            error_patterns.append(sequence)
    return error_patterns

      
class ErrorPatternManager:
    def __init__(self, initial_patterns):
        # Initialize the sorted list of error patterns
        # Convert lists to tuples if necessary
        initial_patterns = [tuple(pattern) if isinstance(pattern, list) else pattern for pattern in initial_patterns]
        #self.sorted_patterns = sorted(initial_patterns)
        self.sorted_patterns = initial_patterns
        # Create a dictionary to map patterns to their indices
        self.pattern_to_index = {tuple(pattern): [idx,0] for idx, pattern in enumerate(self.sorted_patterns)} # second component of the keyed tuple is height of barrier bar
        #print('here')
    def one_step_optimize(self,error_pattern_batch_list):
        self.barrier_adding(error_pattern_batch_list)
        self.update_teps_ordering()
                  
    def barrier_adding(self,error_pattern_batch_list):
        unit_batch_size = GL.get_map('unit_batch_size')
        intercept_length = GL.get_map('intercept_length')
        relax_factor = GL.get_map('relax_factor')
        outlier_count = 0
        batches = create_batches(error_pattern_batch_list, unit_batch_size,shuffle_indicator=False)
        for error_pattern_batch in batches:
            for key in error_pattern_batch:
                if tuple(key) in self.pattern_to_index:
                    self.pattern_to_index[tuple(key)][1] += 1
                else:
                    outlier_count += 1
        min_threshold_capacity = min(len(self.sorted_patterns),(relax_factor-1)*intercept_length)
        if outlier_count >= min_threshold_capacity:
            print(min_threshold_capacity)
            print('Too many outliers exist, enlarge relax factor or required basic length!')
            sys.exit(-1)

    def update_teps_ordering(self):
        tuple_list = []
        for key in self.pattern_to_index.keys():
            #print(self.pattern_to_index[key][1],end=' ')
            tuple_element = (key, self.pattern_to_index[key][0]-self.pattern_to_index[key][1],self.pattern_to_index[key][0])
            tuple_list.append(tuple_element)
        sorted_tuple_list = sorted(tuple_list, key=lambda x: (x[1],x[2]))
        self.sorted_patterns = [key for key,_,_ in sorted_tuple_list]        
        self.pattern_to_index = {tuple(element[0]): [idx,element[1]] for idx, element in enumerate(sorted_tuple_list)} 

    def count_position(self, key):
        out_bound_indicator = 0
        if key not in self.pattern_to_index:
            position = len(self.sorted_patterns)+1
            out_bound_indicator = 1
        else:
            position = self.pattern_to_index[key][0]+1
        return position,out_bound_indicator

    # def find_key(self, dictionary, target_value):
    #     series = pd.Series(dictionary)
    #     return series[series[0] == target_value].index[0]  # Returns the first matching key   

    def update_and_add(self, key):
        #retrieve the last element
        last_item_key = self.sorted_patterns[-1]
        if key not in self.pattern_to_index:
            if self.pattern_to_index[last_item_key][1] == 0:
                self.sorted_patterns[-1] = key
                self.pattern_to_index[key] = [len(self.sorted_patterns)-1,0]
                del self.pattern_to_index[last_item_key]
            else:
                self.pattern_to_index[last_item_key][1] -= 1
        else:
            self.forward_position(key)
        
    def forward_position(self, pattern):
        # Get the current index of the pattern
        idx = self.pattern_to_index[pattern][0]
        # If not the first element, swap with the previous element if its captital is zero, otherwise weaken its captital by one but keeping the order intact
        #print(idx)
        self.pattern_to_index[self.sorted_patterns[idx]][1] += 1
        if idx > 0:           
            if self.pattern_to_index[self.sorted_patterns[idx - 1]][1] > 0:                
                self.pattern_to_index[self.sorted_patterns[idx - 1]][1] -= 1
            else:            
                self.sorted_patterns[idx], self.sorted_patterns[idx - 1] = self.sorted_patterns[idx - 1], self.sorted_patterns[idx]
                # Update the indices in the dictionary
                self.pattern_to_index[self.sorted_patterns[idx]] = [idx,0]
                self.pattern_to_index[self.sorted_patterns[idx - 1]][1] -= 1
                self.pattern_to_index[self.sorted_patterns[idx - 1]][0] = idx - 1

    def calculate_rank_sum(self, batch):
        # Calculate the sum of ranks for the batch
        batch_rank_sum = 0
        batch_out_bound = 0
        for element in batch:           
            pattern = tuple(element)
            #print(self.count_position(pattern),end=' ')
            rank,out_bound_indicator = self.count_position(pattern)
            batch_rank_sum += rank
            batch_out_bound += out_bound_indicator
        return batch_rank_sum,batch_out_bound
    def statistics_batches(self,error_pattern_batch_list):
        batch_matrix = np.concatenate(error_pattern_batch_list,axis=0)
        # Convert each row to a tuple (to make it hashable)
        rows_as_tuples = [tuple(row) for row in batch_matrix]
        frequency_counter = Counter(rows_as_tuples)
        #print(frequency_counter)
        # Create a new Counter with updated keys
        my_dict = {i: 0 for i in range(len(self.sorted_patterns))} #last key is specialized for all out-of-list TEPs
        for key, value in frequency_counter.items():
            if key in self.pattern_to_index:
                my_dict[self.pattern_to_index[key][0]] = value              
        new_counter = Counter(my_dict)
        return new_counter
        
    def process_batch(self, error_pattern_batch):
        # Store the original state before making any changes
        original_sorted_patterns = self.sorted_patterns.copy()
        original_pattern_to_index = self.pattern_to_index.copy()
        original_rank_sum,_ = self.calculate_rank_sum(error_pattern_batch)
        
        # Forward the position of each pattern in the batch
        for idx,element in enumerate(error_pattern_batch):
            pattern = tuple(element) 
            self.update_and_add(pattern)
        
        # Calculate the rank sum after forwarding
        new_rank_sum,_ = self.calculate_rank_sum(error_pattern_batch)
        
        # If the new rank sum is not less than the original, revert to the original state
        if new_rank_sum >= original_rank_sum:
            self.sorted_patterns = original_sorted_patterns
            self.pattern_to_index = original_pattern_to_index
        return original_rank_sum,new_rank_sum
    
    # def iterate_optimize(self,error_pattern_batch_list):
    #     num_steps = GL.get_map('termination_step')
    #     print_interval = GL.get_map('print_interval')
    #     record_interval = GL.get_map('record_interval')
    #     unit_batch_size = GL.get_map('unit_batch_size')
    #     batches = create_batches(error_pattern_batch_list, unit_batch_size,shuffle_indicator=True)
    #     infinite_batches = itertools.cycle(batches)
    #     for i in range(num_steps):           
    #             # Get the next batch from the iterator
    #             error_pattern_batch = next(infinite_batches)
    #             original_rank_sum, new_rank_sum = self.process_batch(error_pattern_batch)
    #             #print(original_rank_sum, new_rank_sum)
    #             # Print progress every `print_interval` batches
    #             if (i+1) % print_interval == 0:
    #                 print(f'{i+1}th rank_sum:({original_rank_sum},{new_rank_sum})')
    #             # Save sorted_patterns and double the batch size every `record_interval` batches
    #             if (i+1) % record_interval == 0:
    #                 unit_batch_size *= 2  # Double the batch size
    #                 # Reorganize the data into larger batches
    #                 batches = create_batches(error_pattern_batch_list, unit_batch_size,shuffle_indicator=True)
    #                 infinite_batches = itertools.cycle(batches)  # Update the infinite_batches iterator
    #             # Terminate after reaching the termination step
    #             if (i+1) >= GL.get_map('termination_step'):
    #                 break

# Define a function to compute cumulative proportions for a counter
def compute_cumulative_proportions(counter,size_sum,max_key):
    # Initialize proportions, defaulting to 0 for missing keys
    counters = [counter.get(key, 0) for key in range(max_key)]      
    # Compute cumulative proportions
    cumulative_values = []
    cumulative_sum = 0
    for value in counters:
        cumulative_sum += value
        cumulative_values.append(size_sum-cumulative_sum)  
    proportion_list = [cumulative_values[i]/size_sum for i in range(len(cumulative_values))]
    return proportion_list
    
def summary_ranking(ranked_list,error_pattern_batch_list):
    unit_batch_size = GL.get_map('unit_batch_size')
    print_interval = GL.get_map('print_interval')
    manager = ErrorPatternManager(ranked_list)
    data_counter = manager.statistics_batches(error_pattern_batch_list)
    rank_sum = 0
    size_sum = 0
    out_bound_sum = 0
    batches = create_batches(error_pattern_batch_list, unit_batch_size,shuffle_indicator=False)
    for i in range(len(batches)):
        error_pattern_batch = batches[i]
        actual_size = error_pattern_batch.shape[0]
        size_sum += actual_size
        batch_rank,batch_out_bound = manager.calculate_rank_sum(error_pattern_batch)
        rank_sum += batch_rank
        out_bound_sum += batch_out_bound
        if (i+1) % print_interval == 0:
            print(f'{i+1}th rank_sum:{rank_sum} avr_sum:{rank_sum/size_sum:.4f}')
    return data_counter,rank_sum,size_sum,out_bound_sum


def drawing_plot1(snr,data_counter_list,low_limit,save_path=None):
    max_key = GL.get_map('intercept_length')
    all_keys = range(max_key)
    # Define styles for each counter (colors, markers, linestyles)
    styles = [
        {'color': 'teal', 'linestyle': '-', 'label': 'CTV/CTV_MS','linewidth': 1.5},
        {'color': 'b',  'linestyle': '-', 'label': 'ALMLT/ALMLT_MS','linewidth': 1.5},
        {'color': 'r',  'linestyle': '-', 'label': 'CTV/CTV_MS +DIA','linewidth': 1.5},
        {'color': 'm',  'linestyle': '-', 'label': 'ALMLT/ALMLT_MS +DIA','linewidth': 1.5},
        {'color': 'orange', 'linestyle': '-', 'label': 'ALMLT_MS','linewidth': 1},
        {'color': 'black',  'linestyle': '-', 'label': 'CVT_S','linewidth': 1},
        {'color': 'navy','linestyle': '--', 'label': 'ALMLT_S','linewidth': 1},
        {'color': 'y','linestyle': '-', 'label': 'ALMLT_S','linewidth': 1}
    ]    

    # Create a cyclic iterator for styles
    style_cycle = itertools.cycle(styles)

    # Plot the cumulative distributions for all counters
    for i, tuple_node in enumerate(data_counter_list):
        if i in [0,1,2,5,6,7]:
            continue
        node_counter,rank_sum,size_sum,out_bound_sum = tuple_node
        print(f'Counter{i+1}: rank_sum:{rank_sum} Size_sum:{size_sum} avr_sum:{rank_sum/size_sum:.4f} obr:{out_bound_sum/size_sum:.4f}')          
        cumulative = compute_cumulative_proportions(node_counter,size_sum,max_key)
        #plt.plot(all_keys, cumulative, **styles[i])   
        plt.semilogy(all_keys, cumulative, **next(style_cycle))  # Use plt.semilogy() for semi-log y-axis

    # Add labels and title
    plt.xlabel('Indexed TEPs')
    plt.ylabel('Residual Omission Rate')
    #plt.title(f'ROR at SNR={snr}dB') #Complementary Cumulative Distribution Curves
    # Use automatic tick labels with a clean format
    # Set y-axis limits to ensure a wider range
    plt.ylim(bottom=low_limit, top=0.4)  # Adjust these values based on your data range
    # Use a better tick formatter for logarithmic scale
    plt.gca().yaxis.set_major_formatter(ticker.LogFormatterSciNotation())  # Scientific notation
    plt.gca().yaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor tick labels
    # Add grid lines for both major and minor ticks
    plt.grid(True, which="both", ls="--")
    plt.legend(loc='upper right', ncol=2)    # Show the plot
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')  # High quality save
        print(f'Figure saved to: {save_path}')
    # Show the plot
    plt.show()
    plt.close()  # Important: Close the figure to free memory
    
def drawing_plot(snr,data_counter_list,low_limit,save_path=None):
    max_key = GL.get_map('intercept_length')
    all_keys = range(max_key)
    # Define styles for each counter (colors, markers, linestyles)
    styles = [
        {'color': 'teal', 'linestyle': '-', 'label': 'Train_bound','linewidth': 1},
        {'color': 'b',  'linestyle': '-', 'label': 'CVT','linewidth': 1},
        {'color': 'r',  'linestyle': 'dotted', 'label': 'ALMLT','linewidth': 2},
        {'color': 'm',  'linestyle': '-', 'label': 'CVT_MS','linewidth': 1},
        {'color': 'orange', 'linestyle': '-', 'label': 'ALMLT_MS','linewidth': 1},
        {'color': 'black',  'linestyle': '-', 'label': 'CVT_S','linewidth': 1},
        {'color': 'navy','linestyle': '--', 'label': 'ALMLT_S','linewidth': 2}
    ]    

    # Create a cyclic iterator for styles
    style_cycle = itertools.cycle(styles)

    # Plot the cumulative distributions for all counters
    for i, tuple_node in enumerate(data_counter_list):
        node_counter,rank_sum,size_sum,out_bound_sum = tuple_node
        print(f'Counter{i+1}: rank_sum:{rank_sum} Size_sum:{size_sum} avr_sum:{rank_sum/size_sum:.4f} obr:{out_bound_sum/size_sum:.4f}')          
        cumulative = compute_cumulative_proportions(node_counter,size_sum,max_key)
        #plt.plot(all_keys, cumulative, **styles[i])   
        plt.semilogy(all_keys, cumulative, **next(style_cycle))  # Use plt.semilogy() for semi-log y-axis

    # Add labels and title
    plt.xlabel('Indexed TEPs')
    plt.ylabel('Residual Omission Rate')
    plt.title(f'ROR at SNR={snr}dB') #Complementary Cumulative Distribution Curves
    # Use automatic tick labels with a clean format
    # Set y-axis limits to ensure a wider range
    plt.ylim(bottom=low_limit, top=0.4)  # Adjust these values based on your data range
    # Use a better tick formatter for logarithmic scale
    plt.gca().yaxis.set_major_formatter(ticker.LogFormatterSciNotation())  # Scientific notation
    plt.gca().yaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor tick labels
    # Add grid lines for both major and minor ticks
    plt.grid(True, which="both", ls="--")
    plt.legend(loc='upper right', ncol=2)    # Show the plot
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')  # High quality save
        print(f'Figure saved to: {save_path}')
    # Show the plot
    plt.show()
    plt.close()  # Important: Close the figure to free memory
