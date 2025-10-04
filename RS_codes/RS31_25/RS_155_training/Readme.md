# Training Guide for RS (31,625)/Binary image of (155,125)

## Development Environment

* **IDE**: Spyder (Anaconda Distribution)
* **OS**: Windows 10
* **Entrance File**: `RS_155_training.py`

---

## Quick Start

### Configuration Setup

In the entrance file `RS_155_training.py`, configure the parameters as follows:

```
sys.argv = "python 3.5 3.5 100 1000 8 Hopt_RS_31_25_3_1440ones.alist NMS-1".split()
```
This corresponds to:

```
sys.argv = "python <min_snr> <max_snr> <batch_size> <num_batches> <max_iterations> <parity_check_matrix_file> <decoder_type>".split()
```
#### Parameter Description

* **min_snr**: Minimum SNR value in dB (e.g., `3.5`)
* **max_snr**: Maximum SNR value in dB (e.g., `3.5`)
* **batch_size**: Number of samples per batch (e.g., `100`)
* **num_batches**: Total number of batches to process (e.g., `100`)
* **max_iterations**: Number of iterations for NMS (e.g., `8`; higher values provide only marginal FER improvement; `4` is recommended for code lengths below `100`, and `8` for lengths above `100`.)
* **parity_check_matrix_file**: RS parity-check matrix file (e.g., `Hopt_RS_31_25_3_1440ones.alist`)
* **decoder_type**: One of the BP variants (e.g., `NMS-1` refers to the conventional NMS with a single evaluation parameter.)

---

### Execution

1. Open `RS_155_training.py` in Spyder.
2. Ensure the configuration line matches your desired parameters.
3. Click the **Run File** icon in Spyder’s toolbar.

Training begins, with intermediate results periodically saved in the output directory. 
After completion, the model is evaluated using the validation dataset. 
NMS decoding failure trajectories are also stored as training samples for the DIA model.

---

## Settings Overview (in `globalmap.py`)

```python
def global_setting(argv):

    set_map('initial_learning_rate', 0.01)
    set_map('decay_rate', 0.95)
    set_map('decay_step', 500)
    set_map('nms_termination_step', 300) # Adam optimizer stops after the specified number of steps.     
    
    set_map('reduction_iteration',4)     #Iterations for acquiring parity-check matrix rows with minimal weights.   
    set_map('redundancy_factor',2)       # Regulates the number of rows in the parity-check matrix
    set_map('num_shifts',5)              # Allowed shifts per received sequence
    
    set_map('print_interval',20)
    set_map('record_interval',20)       # Print results and save model at this interval
    
    set_map('regular_matrix',False)     #Disable conventional parity-check matrix
    set_map('generate_extended_parity_check_matrix',True)  #eEnable optimized parity-check matrix with redundant rows for enhanced NMS decoding
    
    set_map('enhanced_NMS_indicator',True)  #Enable enhanced NMS decoder
    set_map('original_NMS_indicator',False) #Disable conventional NMS decoder (must align with chosen parity-check matrix) 
    set_map('extend_F',5)                   #Extension field to binary image (depends on chosen RS code)
    
def logistic_setting():
    restore_step = 'latest' # restore_step = 'latest' # '' → start fresh; 'latest' → load the most recent model
```

---

## Notes

* Adjust parameters according to your computational resources and training requirements.
* The SNR range `[min_snr, max_snr]` must exactly match the settings in the `Training_data_gen_155` package
* Ensure the parity-check matrix file is accessible within the project directory.

---

## Project Structure

```
├── RS_155_training.py              # 🎯 Main training script (Entrance file)
├── Hopt_RS_31_25_3_1440ones.alist  # 📊 RS parity-check matrix
├── output                          # 📂 Well-trained NMS decoder
└── failures                        # 📂 NMS decoding failure trajectories used as DIA training samples
```
