# Guide to RS (31,25)/Binary Image (155,125) OSD Testing with DIA and SWA Model Support

## Development Environment

* **IDE**: Spyder (Anaconda Distribution)
* **OS**: Windows 10
* **Entrance File**: `Main_DL_OSD.py`

---

## Quick Start

### Configuration Setup

In the entrance file `Main_DL_OSD.py`, configure the parameters as follows:

```python
sys.argv = "python 2.0 4.5 6 100  8 Hopt_RS_31_25_3_1440ones.alist NMS-1".split()
```
for 

```
sys.argv = "python <min_snr> <max_snr> <num_points> <batch_size> <max_iterations> <parity_check_matrix_file> <decoder_type>".split()
```
#### Parameter Description

* **min_snr**: Minimum SNR value in dB (e.g., `2.0`)
* **max_snr**: Maximum SNR value in dB (e.g., `4.5`)
* **num_points**: Number of SNR points evenly spaced between `min_snr` and `max_snr` (e.g., `6`)
* **batch_size**: Number of samples per batch (e.g., `100`)
* **max_iterations**: Number of iterations for NMS (e.g., `8`)
* **parity_check_matrix_file**: RS parity-check matrix file (e.g., `Hopt_RS_31_25_3_1440ones.alist`)
* **decoder_type**: One of the BP variants (e.g., `NMS-1` refers to the conventional NMS with a single evaluation parameter)

---

### Execution

1. Open `Main_DL_OSD.py` in Spyder.
2. Ensure the configuration line matches your desired parameters in `RS_155_testing` package.
3. Click the **Run File** icon in Spyder's toolbar.

Testing will begin by calling OSD supported by DIA and SWA models to decode saved NMS decoding failures in the designated files for each SNR point, and record various performance metrics in the log file.

---

## Settings Overview (in `globalmap.py`)

```python
def global_setting(argv):

    set_map('train_snr',3.5)  #ensure consistency with setting in `DL_Training` package
    set_map('termination_threshold',10) # minimum number of OSD decoding errors collected per SNR point; at least 100 is recommended for statistical reliability.
    
    set_map('intercept_length',1000)
    set_map('relax_factor',10)
    set_map('block_size',100)           #the above three options must remain consistent with available decoding paths in `Optimizing_decoding_path` package
    set_map('DIA_deployment',True)      # toggle usage of the DIA model; this also affects decoding path selection

    #store it onto global space
    set_map('print_interval',100)
    set_map('record_interval',100)    # Print and save results every interval
    set_map('ordering_option',4)     # choose one option among the options of training, convention, ALMT, macro_conv, macro_ALMT, optimized_conv, optimized_ALMT.
              
    set_map('soft_margin',0.5)       # early termination threshold for SWA model
    set_map('win_width',5)           #ensure consistency with `DL_Training` package
    set_map('sliding_strategy',True) #toggle usage of the SWA model

    
def logistic_dia_model():
    restore_model_step = 'latest' # 'latest' ensures the most recent DIA model is loaded
def logistic_swa_model(): 
    restore_model_step = 'latest' # 'latest' ensures the most recent SWA model is loaded 
```

---

## Notes

* Adjust parameters according to your computational resources and testing requirements.
* The SNR range `[min_snr, max_snr]` must strictly match the corresponding settings in `Testing_data_gen_155` package
* Ensure the parity-check matrix file is accessible within the project directory.

---

## Project Structure

```
├── Main_DL_OSD.py       # 🎯 Main testing script (Entrance file)
├── Hopt_RS_31_25_3_1440ones.alist  # 📊 RS parity-check matrix
├── [Performance metrics of OSD aided by DIA and SWA models, saved per SNR point → output directory]
```
