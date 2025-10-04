import itertools,math,pickle
import numpy as np
import globalmap as GL
import ordered_statistics_decoding as Osd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nn_net as CRNN_DEF
import tensorflow as tf

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

def load_error_pattern(dir_file):
    with open(dir_file, 'rb') as f:
        initial_patterns = pickle.load(f)
        authentic_error_pattern_batch_list = pickle.load(f)
        outbound_batch_list = pickle.load(f)
    return initial_patterns, authentic_error_pattern_batch_list,outbound_batch_list
    
def find_error_pattern(input_iterator,outptu_dir_file):
    code = GL.get_map('code_parameters')
    # Example usage
    order_p = GL.get_map('max_order_p')
    initial_patterns = generate_error_patterns(code.k, order_p)
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
        authentic_error_pattern_batch_list,outbound_batch_list = query_error_patterns_dia(osd_model,cnn,data_list)
    else:
        authentic_error_pattern_batch_list,outbound_batch_list = query_error_patterns(osd_model,data_list)
    with open(outptu_dir_file, 'wb') as f:
        pickle.dump(initial_patterns, f)
        pickle.dump(authentic_error_pattern_batch_list, f)
        pickle.dump(outbound_batch_list, f)
    return initial_patterns, authentic_error_pattern_batch_list,outbound_batch_list

#query authentic error patterns
def query_error_patterns_dia(osd_model,cnn,data_list):
    authentic_error_pattern_batch_list = []
    outbound_batch_list = []
    for batch in data_list:
        print('. ',end='')
        squashed_inputs,super_input_matrix,labels = cnn.preprocessing_inputs(batch)
        inputs = cnn(squashed_inputs)
        original_error_pattern_batch = osd_model.convention_osd_preprocess(inputs,labels)
        # Threshold for row weight
        threshold = GL.get_map('max_order_p')
        # Calculate row weights (sum of each row)
        row_weights = np.sum(original_error_pattern_batch, axis=1)
        # Filter rows where row weight is less than or equal to the threshold
        element1 = original_error_pattern_batch[row_weights <= threshold]
        element2 = original_error_pattern_batch[row_weights > threshold]
        authentic_error_pattern_batch_list.append(element1)
        outbound_batch_list.append(element2)
    return authentic_error_pattern_batch_list,outbound_batch_list
    
#query authentic error patterns
def query_error_patterns(osd_model,data_list):
    decoding_length = GL.get_map('num_iterations')+1
    authentic_error_pattern_batch_list = []
    outbound_batch_list = []
    for batch in data_list:
        print('. ',end='')
        inputs,labels = batch[0][::decoding_length],batch[1][::decoding_length]
        original_error_pattern_batch = osd_model.convention_osd_preprocess(inputs,labels)
        # Threshold for row weight
        threshold = GL.get_map('max_order_p')
        # Calculate row weights (sum of each row)
        row_weights = np.sum(original_error_pattern_batch, axis=1)
        # Filter rows where row weight is less than or equal to the threshold
        element1 = original_error_pattern_batch[row_weights <= threshold]
        element2 = original_error_pattern_batch[row_weights > threshold]
        authentic_error_pattern_batch_list.append(element1)
        outbound_batch_list.append(element2)
    return authentic_error_pattern_batch_list,outbound_batch_list

# Function to create batches from the data
def create_batches(authentic_error_pattern_batch_list, batch_size):
    authentic_error_pattern_batch_matrix = np.concatenate(authentic_error_pattern_batch_list,axis=0)
    np.random.shuffle(authentic_error_pattern_batch_matrix)  # Shuffles in-place
    batch_length = math.ceil(authentic_error_pattern_batch_matrix.shape[0]/batch_size)
    batches = []
    for i in range(batch_length):
        element = authentic_error_pattern_batch_matrix[i*batch_size:(i+1)*batch_size]
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
    
    def statistics_batches(self,authentic_error_pattern_batch_list,outbound_batch_list):
        authentic_error_pattern_batch_matrix = np.concatenate(authentic_error_pattern_batch_list,axis=0)
        outbound_batch_matrix = np.concatenate(outbound_batch_list,axis=0)
        # Convert each row to a tuple (to make it hashable)
        rows_as_tuples = [tuple(row) for row in authentic_error_pattern_batch_matrix]
        frequency_counter = Counter(rows_as_tuples)
        #print(frequency_counter)

        # Initialize an empty dictionary to store the new key-value pairs
        new_counter_dict = {}       
        # Iterate over the items in the frequency_counter dictionary
        
        for old_key, value in frequency_counter.items():
            # Get the new key from self.pattern_to_index[old_key][0]
            try:
                new_key = self.pattern_to_index[old_key][0]
                # Add the new key-value pair to the new_counter_dict
                new_counter_dict[new_key] = value
            except KeyError:
                print(f"Key {old_key} not found.")

        # Create a Counter object from the new_counter_dict
        new_counter = Counter(new_counter_dict)
        #new_counter = Counter({self.pattern_to_index[old_key][0]: value for old_key, value in frequency_counter.items()})
        # Find the maximum key in the dictionary
        #max_key = max(new_counter.keys(), default=0)  # Use default=0 to handle empty Counter
        max_key = len(self.pattern_to_index)
        # Insert a new key-value pair, to take into account the impact of occurrences of out of order patterns
        new_counter[max_key] = outbound_batch_matrix.shape[0]
        #print(new_counter)
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
    # Define a function to compute cumulative proportions for a counter
    def compute_cumulative_proportions(self,counter,max_key):
        # Initialize proportions, defaulting to 0 for missing keys
        proportions = [counter.get(key, 0) for key in range(max_key)]      
        # Compute cumulative proportions
        cumulative_values = []
        cumulative_sum = 0
        for value in proportions:
            cumulative_sum += value
            cumulative_values.append(1-cumulative_sum)    
        return cumulative_values
    def drawing_plot(self, data_counter_list):
        #code = GL.get_map('code_parameters')
        #order_p = GL.get_map('max_order_p')
        # Sum of binomial coefficients from k=0 to k=p
        #max_key = sum(math.comb(code.n, k) for k in range(order_p + 1)) 
        max_key = GL.get_map('intercep_length')+1
        all_keys = range(max_key)
        # Define styles for each counter (colors, markers, linestyles)
        styles = [
            #{'color': 'b', 'marker': '*', 'linestyle': '-', 'label': 'Counter 1'},
            {'color': 'b',  'linestyle': '-', 'label': 'Counter 1'},
            #{'color': 'r', 'marker': 's', 'linestyle': '--', 'label': 'Counter 2'},
            {'color': 'r',  'linestyle': '--', 'label': 'Counter 2'},
            {'color': 'g', 'marker': '^', 'linestyle': '-.', 'label': 'Counter 3'},
            {'color': 'm', 'marker': 'D', 'linestyle': ':', 'label': 'Counter 4'},
            {'color': 'c', 'marker': 'x', 'linestyle': '-', 'label': 'Counter 5'},
            {'color': 'y', 'marker': 'o', 'linestyle': '--', 'label': 'Counter 6'}
        ]       
        # Plot the cumulative distributions for all counters
        for i, counter in enumerate(data_counter_list):
            cumulative = self.compute_cumulative_proportions(counter,max_key)
            #plt.plot(all_keys, cumulative, **styles[i])   
            plt.semilogy(all_keys, cumulative, **styles[i])  # Use plt.semilogy() for semi-log y-axis

        # Add labels and title
        plt.xlabel('Key Values (x-axis)')
        plt.ylabel('Complementary  Cumulative Proportions (y-axis, log scale)')
        plt.title('Cumulative Distribution Curves (Semi-Logarithmic Y-Axis)')
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
        
    def process_batch(self, authentic_error_pattern_batch):
        # Store the original state before making any changes
        original_sorted_patterns = self.sorted_patterns.copy()
        original_pattern_to_index = self.pattern_to_index.copy()
        original_rank_sum = self.calculate_rank_sum(authentic_error_pattern_batch)
        
        # Forward the position of each pattern in the batch
        for element in authentic_error_pattern_batch:
            pattern = tuple(element) 
            self.forward_position(pattern)
        
        # Calculate the rank sum after forwarding
        new_rank_sum = self.calculate_rank_sum(authentic_error_pattern_batch)
        
        # If the new rank sum is not less than the original, revert to the original state
        if new_rank_sum >= original_rank_sum:
            self.sorted_patterns = original_sorted_patterns
            self.pattern_to_index = original_pattern_to_index
        return original_rank_sum,new_rank_sum
    def get_sorted_patterns(self,first_n):
        return self.sorted_patterns[:first_n]


