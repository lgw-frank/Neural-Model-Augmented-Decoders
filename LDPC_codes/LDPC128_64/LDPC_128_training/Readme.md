# LDPC (128,64) Training Guide

## Development Environment

* **IDE**: Spyder (Anaconda Distribution)
* **OS**: Windows 10
* **Entrance File**: `LDPC_128_training.py`

---

## Quick Start

### Configuration Setup

In the entrance file `LDPC_128_training.py`, configure the parameters as follows:

```
sys.argv = "python 3.0 3.0 100 100 10 CCSDS_ldpc_n128_k64.alist NMS-1".split()
```
for 

```
sys.argv = "python <min_snr> <max_snr> <batch_size> <num_batches> <max_iterations> <parity_check_matrix_file> <decoder_type>".split()
```
#### Parameter Description

* **min_snr**: Minimum SNR value in dB (e.g., `3.0`)
* **max_snr**: Maximum SNR value in dB (e.g., `3.0`)
* **batch_size**: Number of samples per batch (e.g., `100`)
* **num_batches**: Total number of batches to generate (e.g., `100`)
* **max_iterations**: Number of iterations for NMS (e.g., `10`; larger values yield limited FER gain)
* **parity_check_matrix_file**: LDPC parity-check matrix file (e.g., `CCSDS_ldpc_n128_k64.alist`)
* **decoder_type**: One of the BP variants (e.g., `NMS-1` refers to the conventional NMS with a single parameter to evaluate)

---

### Execution

1. Open `LDPC_128_training.py` in Spyder.
2. Ensure the configuration line matches your desired parameters.
3. Click the **Run File** icon in Spyder's toolbar.

Training will begin, with results saved periodically in the designated output directory. After training concludes, a final parameter evaluation is applied to validate the dataset. The trajectories of NMS decoding failures are stored in a designated directory to be later used as training data for the DIA model.

---

## Settings Overview (in `globalmap.py`)

```python
def global_setting(argv):
    set_map('initial_learning_rate', 0.01)
    set_map('decay_rate', 0.95)
    set_map('decay_step', 500)
    set_map('nms_termination_step', 1000) # Adam optimizer stops after 'nms_termination_step' steps.

    set_map('print_interval', 50)
    set_map('record_interval', 50) # Print results and save model every interval

def logistic_setting():
    restore_step = 'latest' # '' for fresh start; 'latest' to load most recent model
```

---

## Notes

* Adjust parameters according to your computational resources and training needs.
* The SNR range `[min_snr, max_snr]` should strictly conform to the corresponding settings in `Training_data_gen_128` package
* Ensure the parity-check matrix file is accessible within the project directory.

---

## Project Structure

```
├── LDPC_128_training.py       # 🎯 Main training script (Entrance file)
├── CCSDS_ldpc_n128_k64.alist  # 📊 CCSDS LDPC parity-check matrix
├── [Generated trained NMS decoder → output directory]
└── [Generated DIA training data → output directory]
```
