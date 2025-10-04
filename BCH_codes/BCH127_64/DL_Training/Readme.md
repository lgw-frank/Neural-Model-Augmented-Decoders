# Training Guide for BCH (127,64) with DIA and SWA Models

## Development Environment

- **IDE**: Spyder (Anaconda Distribution)  
- **OS**: Windows 10  
- **Entrance File**: `Main_DL.py`  

---

## Quick Start

### Configuration Setup

In the entrance file `Main_DL.py`, configure the parameters as follows:

```python
sys.argv = "python 3.0 3.0 100 8 BCH_127_64_10_strip.alist NMS-1".split()

#General form:
sys.argv = "python <min_snr> <max_snr> <batch_size> <max_iterations> <parity_check_matrix_file> <decoder_type>".split()
```
#### Parameter Description
* **min_snr**: Minimum SNR value in dB (e.g., `3.0`)
* **max_snr**: Maximum SNR value in dB (e.g., `3.0`)
* **batch_size**: Number of samples per batch (e.g., `100`)
* **max_iterations**: Number of iterations for NMS (e.g., `8`; higher values provide only marginal FER improvement)
* **parity_check_matrix_file**: BCH parity-check matrix file (e.g., `BCH_127_64_10_strip.alist`)
* **decoder_type**: One of the BP variants (e.g., `NMS-1` refers to the conventional NMS with a single evaluation parameter)

---

### Execution

1. Open `Main_DL.py` in Spyder.
2. Ensure the configuration line matches your desired parameters in `BCH_127_training` package.
3. Click the **Run File** icon in Spyder’s toolbar..

Training first executes the DIA model, followed by the SWA model, with results saved periodically in their respective output directories.
---

## Settings Overview (in `globalmap.py`)
```python
def global_setting(argv):
   
    set_map('initial_learning_rate', 0.01)
    set_map('decay_rate', 0.95)
    set_map('decay_step', 500)

    # Optimizer termination steps
    set_map('dia_termination_step', 2000)  # Termination steps for Adam optimizer of DIA model
    set_map('swa_termination_step', 3000)  # Termination steps for Adam optimizer of SWA model

    set_map('regular_matrix',True)        # Conventional parity-check matrix is requied
    
    # Logging intervals
    set_map('print_interval', 100)
    set_map('record_interval', 100)

    # Training switches
    set_map('dia_model_train', True)
    set_map('swa_model_train', False)  # Enable training for DIA first, then SWA

    # SWA sample handling
    set_map('regenerate_swa_samples', True)  #Enable sample collection if not already available
    set_map('win_width', 5)           
    set_map('intercept_length', 1000)
    set_map('relax_factor', 10)
    set_map('block_size', 100)

    # Deployment settings
    set_map('DIA_deployment', True)
    set_map('ALMLT_available', True)   # SWA relies on a pre-secured decoding path
                                       # Ensure the DIA model is well-trained when DIA_deployment is enabled

def logistic_dia_model():   # For DIA model
    restore_step = 'latest'  # '' for fresh start; 'latest' to load the most recent model
    
def logistic_swa_model():   # For SWA model
    restore_step = 'latest'  # '' for fresh start; 'latest' to load the most recent model. 
                             # DIA model must be trained before SWA training starts
```

---

## Notes

Adjust parameters according to your computational resources and training requirements.

The SNR range [min_snr, max_snr] must strictly match the settings in the `Training_data_gen_127` package.

Ensure the parity-check matrix file is available in the project directory.

---

## Project Structure

```
├── Main_DL.py                  # 🎯 Main training script (Entrance file)
├── BCH_127_64_10_strip.alist   # 📊 BCH parity-check matrix
├── output                      # 📂 Generated trained DIA and SWA models
└── README.md                   # 📄 Project documentation
```