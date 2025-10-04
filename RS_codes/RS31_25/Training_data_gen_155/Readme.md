# RS (31,25)[binary image (155,125)] Training Data Generation Guide

## Development Environment

* **IDE**: Spyder (Anaconda Distribution)
* **OS**: Windows 10
* **Entrance File**: `Main_train_gen.py`

---

## Quick Start

### Configuration Setup

In the entrance file `Main_train_gen.py`, configure the parameters as follows:

```
sys.argv = "python 3.5 3.5 100 100 Hopt_RS_31_25_3_1440ones.alist ".split()
```
for 

```
sys.argv = "python <min_snr> <max_snr> <batch_size> <num_batches> <parity_check_matrix_file>".split()
```

#### Parameter Description

* **min_snr**: Minimum SNR value in dB (e.g., `3.5`)
* **max_snr**: Maximum SNR value in dB (e.g., `3.5`)
* **batch_size**: Number of samples per batch (e.g., `100`)
* **num_batches**: Total number of batches to generate (e.g., `100`)
* **parity_check_matrix_file**: RS parity-check matrix file (e.g., `Hopt_RS_31_25_3_1440ones.alist`)

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
├── Hopt_RS_31_25_3_1440ones.alist  # 📊 RS parity-check matrix
└── [Generated training data → output directory]
```
