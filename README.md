# HDC-ICD

Code for an anonymous submission on hyperdimensional computing for ventricular arrhythmia detection.

The repository contains the training and evaluation code, the fault-injection experiment, the C
detector for the STM32F303K8, and the scripts that draw the result figures.

## Setup

    pip install -r requirements.txt

The dataset is not included. Download the public single-lead IEGM dataset that the paper uses and
its train and test index files, then build the cache once (about a minute):

    python data.py --path_data <folder with the .txt windows> --path_indices <folder with the index files>

This writes `cache_iegm.npz`. All scripts read it, hold out the same ten training patients for
validation, and use the test patients only for the final numbers.

## Reproducing the results

| Result in the paper | Command | Output |
|---|---|---|
| Configuration selection, dimension and representation (Table II) | `python select_config.py` | `results_select/` |
| Dimensions 8192 and 16384 | `python select_config.py --sets no_fft --epochs 20 --dims 8192 16384 --budget_kib 1000 --outdir results_select_big` | `results_select_big/` |
| Baselines on the same features (Table I) | `python baselines.py` | `results_baselines/` |
| Bit-flip robustness (Fig. 2) | `python robustness.py --trials 100` | `results_robustness/` |
| Encoder comparison | `python run_encoders_d4096.py` | `results_enc_d4096/` |
| Training, threshold and feature-group ablations | `python sweep_hdc.py` | `results_hdc/` |
| Model export and on-device run | see `README_DEVICE.md` | `device_export/` |
| Figures | `python make_robustness.py results_robustness.json`, `python make_configs.py`, `python make_energy.py` | PDF files |

The files `results_*.md` and `results_robustness.json` hold the summaries behind the numbers in the paper.

## Files

- `data.py`, `features.py`: data loading, patient-wise split, the 30 candidate features.
- `encoders.py`: key-level encoder and the three encoders it is compared with.
- `train_hdc.py`: prototype training, retraining and threshold tuning.
- `select_config.py`, `dim_sweep.py`, `binary_eval.py`: configuration grid and binary models.
- `baselines.py`: logistic regression, decision tree, SVM and neural network.
- `robustness.py`: random bit flips in the stored model of every classifier.
- `export_model.py`, `hdc_infer.c`, `hdc_infer.h`, `hdc_main_snippet.c`, `host_eval.py`, `c_check.py`:
  bit-exact C detector with level regeneration, and the host script that sends the test windows
  to the board and collects predictions and cycle counts.
