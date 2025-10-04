import itertools,math,pickle
import numpy as np
import globalmap as GL
import ordered_statistics_decoding as Osd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
import nn_net as CRNN_DEF
from scipy.special import erf
from scipy.optimize import fsolve
from scipy.special import comb
import itertools

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


def retore_saved_model(restore_model_info):
    restore_ckpts_dir,restore_step,ckpt_nm = restore_model_info
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

def iterate_optimize(manager,error_pattern_batch_list,fetch_count,output_ranking_dir,suffix):
    num_steps = GL.get_map('termination_step')
    print_interval = GL.get_map('print_interval')
    record_interval = GL.get_map('record_interval')
    unit_batch_size = GL.get_map('unit_batch_size')
    order_p = GL.get_map('max_order_p')
    batches = create_batches(error_pattern_batch_list, unit_batch_size,shuffle_indicator=True)
    infinite_batches = itertools.cycle(batches)
    for i in range(num_steps):           
            # Get the next batch from the iterator
            error_pattern_batch = next(infinite_batches)
            original_rank_sum, new_rank_sum = manager.process_batch(error_pattern_batch)
            #print(original_rank_sum, new_rank_sum)
            fetch_count += 1
            # Print progress every `print_interval` batches
            if fetch_count % print_interval == 0:
                print(f'{fetch_count}th rank_sum:({original_rank_sum},{new_rank_sum})')
            # Save sorted_patterns and double the batch size every `record_interval` batches
            if fetch_count % record_interval == 0:
                unit_batch_size *= 2  # Double the batch size
                # Reorganize the data into larger batches
                batches = create_batches(error_pattern_batch_list, unit_batch_size,shuffle_indicator=True)
                infinite_batches = itertools.cycle(batches)  # Update the infinite_batches iterator
                # Save the sorted_patterns to a file
                output_ranking_file = f'{output_ranking_dir}order-{order_p}-ranking-{fetch_count}{suffix}.pkl'
                with open(output_ranking_file, 'wb') as f:
                    pickle.dump(manager.sorted_patterns, f)
                    print(f"Saved sorted_patterns at fetch {fetch_count}")
            # Terminate after reaching the termination step
            if fetch_count >= GL.get_map('termination_step'):
                break
    output_ranking_file = f'{output_ranking_dir}order-{order_p}-ranking-{fetch_count}{suffix}.pkl'
    with open(output_ranking_file, 'wb') as f:
            pickle.dump(manager.sorted_patterns, f)
            print(f"Saved sorted_patterns at fetch {fetch_count}")   
        
def find_error_pattern(input_iterator):
    code = GL.get_map('code_parameters')
    #query authentic error patterns
    osd_model = Osd.osd_light(code)
    data_list =list(input_iterator) 
    if GL.get_map('DIA_deployment'):
        #load DIA model
        cnn = CRNN_DEF.conv_bitwise()
        checkpoint = tf.train.Checkpoint(myAwesomeModel=cnn)
        restore_model_info = GL.logistic_setting_model()
        ckpt_f = retore_saved_model(restore_model_info)
        status = checkpoint.restore(ckpt_f)
        status.expect_partial()      
        error_pattern_batch_list = query_error_patterns_dia(osd_model,cnn,data_list)
    else:
        error_pattern_batch_list = query_error_patterns(osd_model,data_list)

    return error_pattern_batch_list

#query authentic error patterns
def query_error_patterns_dia(osd_model,cnn,data_list):
    error_pattern_batch_list = []
    for batch in data_list:
        print('. ',end='')
        squashed_inputs,super_input_matrix,labels = cnn.preprocessing_inputs(batch)
        inputs = cnn(squashed_inputs)
        element = osd_model.convention_osd_preprocess(inputs,labels)
        error_pattern_batch_list.append(element)
    return error_pattern_batch_list
    
#query authentic error patterns
def query_error_patterns(osd_model,data_list):
    decoding_length = GL.get_map('num_iterations')+1
    error_pattern_batch_list = []
    for batch in data_list:
        print('. ',end='')
        inputs,labels = batch[0][::decoding_length],batch[1][::decoding_length]
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
        self.pattern_to_index = {pattern: [idx,0] for idx, pattern in enumerate(self.sorted_patterns)} # second component of the keyed tuple is height of barrier bar
        #print('here')
    def count_position(self, key):
        if key not in self.pattern_to_index:
            position = len(self.sorted_patterns)
        else:
            position = self.pattern_to_index[key][0]
        return position

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
                self.pattern_to_index[last_item_key][1] -= 0
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
        rank_sum = 0
        for element in batch:           
            pattern = tuple(element)
            #print(pattern,self.pattern_to_index[pattern])
            rank_sum += self.count_position(pattern)
        return rank_sum
    def statistics_batches(self,error_pattern_batch_list):
        batch_matrix = np.concatenate(error_pattern_batch_list,axis=0)
        # Convert each row to a tuple (to make it hashable)
        rows_as_tuples = [tuple(row) for row in batch_matrix]
        frequency_counter = Counter(rows_as_tuples)
        #print(frequency_counter)
        # Create a new Counter with updated keys
        my_dict = {}
        for key, value in frequency_counter.items():
            if key in self.pattern_to_index:
                my_dict[self.pattern_to_index[key][0]] = value
            else:
                my_dict[len(self.pattern_to_index)] = my_dict.get(len(self.pattern_to_index), 0) + value               
        new_counter = Counter(my_dict)
        # Calculate the total number of elements
        total_counts = sum(new_counter.values())   
        # Update the original Counter with proportions
        for key in new_counter:
            new_counter[key] = round(new_counter[key] / total_counts,6)
        # Print the updated Counter
        #print("\nUpdated Original Counter with Proportions:")
        #print(new_counter)
        # Define the merge threshold (m = 85)
        m = GL.get_map('intercep_length')
        # Initialize a new counter to store the updated data
        updated_counter = Counter()      
        # Sum proportions for keys >= m
        sum_merged = 0
        partial_sum = 0.
        for key, value in new_counter.items():
            if key < m:
                updated_counter[key] = value
                partial_sum += value
            else:
                sum_merged = 1-partial_sum    
        # Add the merged key (m = 85) with the summed proportion
        updated_counter[m] = sum_merged
        return updated_counter

        
    def process_batch(self, error_pattern_batch):
        # Store the original state before making any changes
        original_sorted_patterns = self.sorted_patterns.copy()
        original_pattern_to_index = self.pattern_to_index.copy()
        original_rank_sum = self.calculate_rank_sum(error_pattern_batch)
        
        # Forward the position of each pattern in the batch
        for idx,element in enumerate(error_pattern_batch):
            pattern = tuple(element) 
            self.update_and_add(pattern)
        
        # Calculate the rank sum after forwarding
        new_rank_sum = self.calculate_rank_sum(error_pattern_batch)
        
        # If the new rank sum is not less than the original, revert to the original state
        if new_rank_sum >= original_rank_sum:
            self.sorted_patterns = original_sorted_patterns
            self.pattern_to_index = original_pattern_to_index
        return original_rank_sum,new_rank_sum

# Define a function to compute cumulative proportions for a counter
def compute_cumulative_proportions(counter,max_key):
    # Initialize proportions, defaulting to 0 for missing keys
    proportions = [counter.get(key, 0) for key in range(max_key)]      
    # Compute cumulative proportions
    cumulative_values = []
    cumulative_sum = 0
    for value in proportions:
        cumulative_sum += value
        cumulative_values.append(1-cumulative_sum)    
    return cumulative_values
    
def summary_ranking(ranked_list,error_pattern_batch_list):
    unit_batch_size = GL.get_map('unit_batch_size')
    print_interval = GL.get_map('print_interval')
    manager1 = ErrorPatternManager(ranked_list)
    data_counter = manager1.statistics_batches(error_pattern_batch_list)
    rank_sum = 0
    fetch_count = 0
    batches = create_batches(error_pattern_batch_list, unit_batch_size,shuffle_indicator=False)
    for error_pattern_batch in batches:
        fetch_count += 1
        rank = manager1.calculate_rank_sum(error_pattern_batch)
        rank_sum += rank
        if fetch_count % print_interval == 0:
            print(f'{fetch_count}th rank_sum:{rank_sum} avr_sum:{rank_sum/(fetch_count*unit_batch_size):.4f}')
    return data_counter,rank_sum,fetch_count

def drawing_plot(data_counter_list):
    unit_batch_size = GL.get_map('unit_batch_size')
    max_key = GL.get_map('intercep_length')+1
    all_keys = range(max_key)
    # Define styles for each counter (colors, markers, linestyles)
    # styles = [
    #     #{'color': 'b', 'marker': '*', 'linestyle': '-', 'label': 'Counter 1','linewidth': 2},
    #     {'color': 'b',  'linestyle': '-', 'label': 'convention_teps','linewidth': 1},
    #     #{'color': 'r', 'marker': 's', 'linestyle': '--', 'label': 'Counter 2','linewidth': 2},
    #     {'color': 'r',  'linestyle': '-', 'label': 'ALMLT_teps','linewidth': 1},
    #     {'color': 'g',  'linestyle': '-', 'label': 'convention_sorted','linewidth': 1},
    #     #{'color': 'g', 'marker': '^', 'linestyle': '-.', 'label': 'Counter 3','linewidth': 2},
    #     {'color': 'm', 'marker': 'D', 'linestyle': ':', 'label': 'ALMLT_sorted','linewidth': 1},
    #     #{'color': 'c', 'marker': 'x', 'linestyle': '-', 'label': 'Counter 5','linewidth': 2},
    #     {'color': 'y', 'marker': 'o', 'linestyle': '--', 'label': 'Counter 6','linewidth': 2}
    # ]    

    # Define styles
    styles = [
        {'color': 'b', 'linestyle': '-', 'label': 'convention_teps', 'linewidth': 2},
        {'color': 'r', 'linestyle': '-', 'label': 'ALMLT_teps', 'linewidth': 2},
        {'color': 'g', 'linestyle': '-', 'label': 'convention_sorted', 'linewidth': 2},
        {'color': 'm', 'linestyle': '-', 'label': 'ALMLT_sorted','linewidth': 2 },
        {'color': 'y', 'linestyle': '--', 'label': 'Counter 6', 'linewidth': 1}
    ]
    # Create a cyclic iterator for styles
    style_cycle = itertools.cycle(styles)

    # Plot the cumulative distributions for all counters
    for i, tuple_node in enumerate(data_counter_list):
        node_counter,rank_sum,fetch_count = tuple_node
        print(f'Counter{i+1}: rank_sum:{rank_sum} fetch_count:{fetch_count} avr_sum:{rank_sum/(fetch_count*unit_batch_size):.4f}')          
        cumulative = compute_cumulative_proportions(node_counter,max_key)
        #plt.plot(all_keys, cumulative, **styles[i])   
        plt.semilogy(all_keys, cumulative, **next(style_cycle))  # Use plt.semilogy() for semi-log y-axis

    # Add labels and title
    plt.xlabel('Indexed positions (x-axis)')
    plt.ylabel('Complementary  Cumulative Proportions (y-axis, log scale)')
    plt.title('CCDC (Semi-Log)  Y-Axis)') #Complementary Cumulative Distribution Curves
    # Use automatic tick labels with a clean format
    # Set y-axis limits to ensure a wider range
    plt.ylim(bottom=1e-3, top=1)  # Adjust these values based on your data range
    # Use a better tick formatter for logarithmic scale
    plt.gca().yaxis.set_major_formatter(ticker.LogFormatterSciNotation())  # Scientific notation
    plt.gca().yaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor tick labels
    # Add grid lines for both major and minor ticks
    plt.grid(True, which="both", ls="--")
    plt.legend()
    # Show the plot
    plt.show()

