# LDPC (128,64) Testing Optimized Decoding Path Guide

## Development Environment

* **IDE**: Spyder (Anaconda Distribution)
* **OS**: Windows 10
* **Entrance File**: `Main_testing.py`

---

## Quick Start

### Configuration Setup

In the entrance file `Main_testing.py`, configure the parameters as follows:

```python
sys.argv = "python 1.0 3.5 6 100 100 10 CCSDS_ldpc_n128_k64.alist NMS-1".split()
```

This corresponds to:

```python
sys.argv = "python <min_snr> <max_snr> <num_points> <batch_size> <num_batches> <max_iterations> <parity_check_matrix_file> <decoder_type>".split()
```

#### Parameter Description

* **min_snr**: Minimum SNR value in dB (e.g., `1.0`)
* **max_snr**: Maximum SNR value in dB (e.g., `3.5`)
* **num_points**: number of SNR points evenly across **min_snr** and **max_snr** (e.g., `6`)
* **batch_size**: Number of samples per batch (e.g., `100`)
* **num_batches**: Total number of batches to process (e.g., `100`)
* **max_iterations**: Number of iterations for NMS (e.g., `10`; larger values yield limited FER gain)
* **parity_check_matrix_file**: LDPC parity-check matrix file (e.g., `CCSDS_ldpc_n128_k64.alist`)
* **decoder_type**: One of the BP variants (e.g., `NMS-1` refers to the conventional NMS with a single evaluation parameter)

---

### Execution

1. Open `Main_testing.py` in Spyder.
2. Ensure the configuration line matches your desired parameters in `LDPC_128_testing` package.
3. Click the **Run File** icon in Spyder's toolbar.

The Testing begins by calling secured decoding paths in file to evaluate the designated SNR points, records various related metrics in 
log file and plots performance curves for various decoding paths.

---

## Settings Overview (in `globalmap.py`)

```python
def global_setting(argv):
    set_map('train_snr',3.0)        # it has to conform to the actual setting for Optimizing_decoding_path package
    set_map('DIA_deployment', True)  # If True, restore_step must be set to 'latest' in logistic_setting_model()     
    set_map('intercept_length', 1000)
    set_map('relax_factor', 10)      # Parameters for enhanced ALMLT, it has to conform to the actual setting for Optimizing_decoding_path package
    set_map('regenerate_data_points',False)  # Whether to regenerate data or load an existing saved data file per SNR point 

def logistic_setting_model():
    restore_step = 'latest'  # '' for fresh start; 'latest' to load the most recent DIA model
```

---

## Notes

* Adjust parameters based on computational resources and testing needs.
* The SNR range `[min_snr, max_snr]` must match the settings in the `Testing_data_gen_128` package.
* Ensure the parity-check matrix file is accessible within the project directory.

---

## Project Structure

```
├── Main_testing.py       # 🎯 Main testing decoding path script (entry point)
├── CCSDS_ldpc_n128_k64.alist  # 📊 CCSDS LDPC parity-check matrix
├── [Generate data using enhanced ALMLT decoding and record various metrics → output directory]
└── [Plots showing the effect of enhanced ALMLT decoding paths per SNR point→ output directory]
```
