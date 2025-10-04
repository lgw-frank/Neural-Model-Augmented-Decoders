# BCH (127,64) Testing Optimized Decoding Path Guide

## Development Environment

* **IDE**: Spyder (Anaconda Distribution)
* **OS**: Windows 10
* **Entrance File**: `Main_testing.py`

---

## Quick Start

### Configuration Setup

In the entrance file `Main_testing.py`, configure the parameters as follows:

```python
sys.argv = "python 2.0 4.5 6 100 100 8 BCH_127_64_10_strip.alist NMS-1".split()
```

This corresponds to:

```python
sys.argv = "python <min_snr> <max_snr> <num_points> <batch_size> <num_batches> <max_iterations> <parity_check_matrix_file> <decoder_type>".split()
```

#### Parameter Description

* **min_snr**: Minimum SNR value in dB (e.g., `2.0`)
* **max_snr**: Maximum SNR value in dB (e.g., `4.5`)
* **num_points**: number of SNR points evenly distributed between **min_snr** and **max_snr** (e.g., `6`)
* **batch_size**: Number of samples per batch (e.g., `100`)
* **num_batches**: Total number of batches to process (e.g., `100`)
* **max_iterations**: Number of iterations for NMS (e.g., `8`; larger values yield limited FER gain)
* **parity_check_matrix_file**: BCH parity-check matrix file (e.g., `BCH_127_64_10_strip.alist`)
* **decoder_type**: One of the BP variants (e.g., `NMS-1` refers to the conventional NMS with a single evaluation parameter)

---

### Execution

1. Open `Main_testing.py` in Spyder.
2. Ensure the configuration line matches your desired parameters in `BCH_127_testing` package.
3. Click the **Run File** icon in Spyder's toolbar.

Testing begins by invoking the secured decoding paths to evaluate the specified SNR points. 
It records relevant metrics in log files and generates performance curves for different decoding paths.

---

## Settings Overview (in `globalmap.py`)

```python
def global_setting(argv):
    set_map('train_snr',3.0)        # Must align with the actual setting in the `Optimizing_decoding_path` package
    set_map('DIA_deployment', True)  # If True, ensure `restore_step` is set to 'latest' in logistic_setting_model()     
    set_map('intercept_length', 1000)
    set_map('relax_factor', 10)      # Enhanced ALMLT parameter. Must match the setting in the `Optimizing_decoding_path` package
    set_map('regenerate_data_points',False)  # True → regenerate data per SNR point; False → use existing saved data 

def logistic_setting_model():
    restore_step = 'latest'  # '' → start fresh; 'latest' → load the most recent DIA model
```

---

## Notes

* Adjust parameters based on computational resources and testing needs.
* The SNR range `[min_snr, max_snr]` must match the settings in the `Testing_data_gen_127` package.
* Ensure the specified parity-check matrix file is available in the project directory..

---

## Project Structure

```
├── Main_testing.py            # 🎯 Main testing decoding path script (entry point)
├── BCH_127_64_10_strip.alist  # 📊 BCH parity-check matrix
├── output                     # 📂 Generated data and logged metrics from enhanced ALMLT decoding
└── plots                      # 📂 Performance curves showing enhanced ALMLT decoding effects across SNR points
```
