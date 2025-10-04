# BCH (127,64) Decoding Path Optimization Guide

## Development Environment

* **IDE**: Spyder (Anaconda Distribution)
* **OS**: Windows 10
* **Entrance File**: `Main_optimization.py`

---

## Quick Start

### Configuration Setup

In the entrance file `Main_optimization.py`, configure the parameters as follows:

```python
sys.argv = "python 3.0 3.0 100 1000 8 BCH_127_64_10_strip.alist NMS-1".split()
```

This corresponds to:

```python
sys.argv = "python <min_snr> <max_snr> <batch_size> <num_batches> <max_iterations> <parity_check_matrix_file> <decoder_type>".split()
```

#### Parameter Description

* **min_snr**: Minimum SNR value in dB (e.g., `3.0`)
* **max_snr**: Maximum SNR value in dB (e.g., `3.0`)
* **batch_size**: Number of samples per batch (e.g., `100`)
* **num_batches**: Total number of batches to generate (e.g., `100`)
* **max_iterations**: Number of iterations for NMS (e.g., `8`)
* **parity_check_matrix_file**: BCH parity-check matrix file (e.g., `BCH_127_64_10_strip.alist`)
* **decoder_type**: One of the BP variants (e.g., `NMS-1` refers to the conventional NMS with a single evaluation parameter)

---

### Execution

1. Open `Main_optimization.py` in Spyder.
2. Ensure the configuration line matches your desired parameters in `BCH_127_training` package.
3. Click the **Run File** icon in Spyder's toolbar.

The optimization begins by initializing the order of TEPs with the ALMLT algorithm, assigning index of each TEP to its counter. The decoding path is then updated by subtracting the occurrences of true error patterns from the respective counters and reordering TEPs in ascending order of evaluation. The updated list forms the enhanced ALMLT. More training samples improve long-term optimization, known as **on-line optimization of enhanced ALMLT**.

---

## Settings Overview (in `globalmap.py`)

```python
def global_setting(argv):
    set_map('DIA_deployment', True)  # If True, restore_step must be set to 'latest' in logistic_setting_model()
    set_map('intercept_length', 1000)
    set_map('relax_factor', 10)      # Parameters for enhanced ALMLT; decoding path file will be used by DL_Training and DL_Testing packages
    set_map('create_update_teps_ranking', True)  # Whether to update ALMLT decoding path using samples in an existing saved data file or on-the-fly samples.

def logistic_setting_model():
    restore_step = 'latest'  # '' for fresh start; 'latest' to load the most recent model
```

---

## Notes

* Adjust parameters based on computational resources and testing needs.
* The SNR range `[min_snr, max_snr]` must match the settings in the `Training_data_gen_127` package.
* Ensure the parity-check matrix file is accessible within the project directory.

---

## Project Structure

```
├── Main_optimization.py       # 🎯 Main optimization script (entry point)
├── BCH_127_64_10_strip.alist  # 📊 BCH parity-check matrix
├── [Generated data file for enhanced ALMLT decoding path files → output directory]
├── [Generated enhanced ALMLT decoding path files → output directory]
└── [Plots showing the effect of enhanced ALMLT decoding paths → output directory]
```
