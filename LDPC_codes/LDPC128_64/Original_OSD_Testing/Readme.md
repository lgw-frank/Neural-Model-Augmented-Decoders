# LDPC (128,64) Mean Delta Validation  Guide

## Development Environment

* **IDE**: Spyder (Anaconda Distribution)
* **OS**: Windows 10
* **Entrance File**: `Main_mean_delta.py`

---

## Quick Start

### Configuration Setup

In the entrance file `Main_mean_delta.py`, configure the parameters as follows:

```python
sys.argv = "python 1.0 3.5 6 10 100 CCSDS_ldpc_n128_k64.alist NMS-1".split()
```

This corresponds to:

```python
sys.argv = "python <min_snr> <max_snr> <num_points> <max_iterations>  <batch_size>  <parity_check_matrix_file> <decoder_type>".split()
```

#### Parameter Description

* **min_snr**: Minimum SNR value in dB (e.g., `1.0`)
* **max_snr**: Maximum SNR value in dB (e.g., `3.5`)
* **num_points**: Number of SNR points evenly distributed **min_snr** and **max_snr** (e.g., `6`)
* **max_iterations**: Maximum number of decoding iterations (e.g., `10`)
* **batch_size**: Number of samples per batch (e.g., `100`)
* **parity_check_matrix_file**: LDPC parity-check matrix file (e.g., `CCSDS_ldpc_n128_k64.alist`)
* **decoder_type**: One of the BP variants (e.g., `NMS-1` refers to the conventional NMS with a single evaluation parameter)

---

### Execution

1. Open `Main_mean_delta.py` in Spyder.
2. Ensure the configuration line matches your desired parameters.
3. Click the **Run File** icon in Spyder's toolbar.
---
The validation task begins by loading NMS decoding data files and NMS decoding failure data files for each SNR point. 
It then plots their mean delta distribution, benchmarked against the ALMLT theoretical calculation. Additionally, an order-p
conventional OSD (where $p=1$ in our script) is applied to demonstrate its decoding strength in terms of frame error rate. 
Various related metrics are logged to files, and mean delta curves are plotted for different data distributions.


## Settings Overview (in `globalmap.py`)

```python
def global_setting(argv):
    set_map('termination_threshold',10)  # Maximum number of errors to collect per SNR point for conventional OSD decoding, '$10^2$' or more is favored for statistical reliability.
    set_map('maximum_order',1)           # Order p for conventional OSD

```

---

## Notes

* Adjust parameters based on your computational resources and testing requirements.
* All arguments in `sys.argv` must match the settings in packages such as `LDPC_testing_128` etc.
* Ensure the parity-check matrix file is accessible within the project directory.

---

## Project Structure

```
├── Main_mean_delta.py       # 🎯 Main validation script (entry point)
├── CCSDS_ldpc_n128_k64.alist  # 📊 CCSDS LDPC parity-check matrix
├── [Various metrics are recorded in log files → output directory]
└── [Mean delta curves for different data distributions are saved as figures→ output directory]
```
