# BCH (127,64) Training Data Generation Guide

## Development Environment

* **IDE**: Spyder (Anaconda Distribution)
* **OS**: Windows 10
* **Entrance File**: `Main_train_gen.py`

---

## Quick Start

### Configuration Setup

In the entrance file `Main_train_gen.py`, configure the parameters as follows:

```
sys.argv = "python 3.0 3.0 100 100  BCH_127_64_10_strip.alist ".split()
```
for 

```
sys.argv = "python <min_snr> <max_snr> <batch_size> <num_batches> <parity_check_matrix_file>".split()
```

#### Parameter Description

* **min_snr**: Minimum SNR value in dB (e.g., `3.0`)
* **max_snr**: Maximum SNR value in dB (e.g., `3.0`)
* **batch_size**: Number of samples per batch (e.g., `100`)
* **num_batches**: Total number of batches to generate (e.g., `100`)
* **parity_check_matrix_file**: BCH parity-check matrix file (e.g., `BCH_127_64_10_strip.alist`)

---

### Execution

1. Open `Main_train_gen.py` in Spyder.
2. Ensure the configuration line matches your desired parameters.
3. Click the **Run File** icon in Spyder's toolbar.

Training data will be generated and stored in the designated output directory.

---

## Notes

* Adjust parameters according to your computational resources and training needs.
* The SNR range `[min_snr, max_snr]` creates evenly distributed batches across the specified interval.
* Ensure the parity-check matrix file is accessible within the project directory.

---

## Project Structure

```
├── Main_train_gen.py          # 🎯 Main training data script (Entrance file)
├── BCH_127_64_10_strip.alist  # 📊 BCH parity-check matrix
└── [Generated training data → output directory]
```
