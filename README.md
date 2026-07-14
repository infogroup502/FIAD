# FIAD
This repository contains the official implementation for the paper [Fourier-Informed Spatiotemporal Anomaly Detection for Industrial Multisensor Signals]().

## Requirements
The recommended requirements for FIAD are specified as follows:
- arch==7.0.0
- einops==0.8.0
- hurst==0.0.5
- matplotlib==3.9.2
- numpy==1.26.4
- pandas==1.5.3
- scikit-learn==1.3.2
- scipy==1.13.1
- statsmodels==0.14.1
- torch==1.13.1
- tqdm==4.66.2
- tsfresh==0.20.3

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

## Data
The datasets can be obtained and put into the `dataset/` folder in the following way:
- FIAD supports anomaly detection for multivariate industrial and multisensor time series datasets.
- If you want to use your own dataset, please place your dataset files in the `/dataset/<dataset>/` folder, following the format `<dataset>_train.npy`, `<dataset>_test.npy`, `<dataset>_test_label.npy`.
- For our datasets:
  - [SMD](https://drive.google.com/drive/folders/1k89esMkTtRhgl4X-ipU8eHx7J0dXuc6k) should be placed at `dataset/SMD/`.
  - [SMAP](https://drive.google.com/drive/folders/1wUOMwWTOAwkhzedL5tBJwqN1rrl_HHTa) should be placed at `dataset/SMAP/`.
  - [MSL](https://drive.google.com/drive/folders/1GOEQ7RdG3bjiEmxzDI_IgN3_VmiHkh5s) should be placed at `dataset/MSL/`.
  - [PSM](https://drive.google.com/drive/folders/1TdpjZpmH3CbkKgk24YX4jVZMQgcrVz0T) should be placed at `dataset/PSM/`.

## Code Description
There are six files/folders in the source:
- data_factory: The preprocessing folder. Dataset loading and preprocessing code is here.
- main.py: The training entry script. Main experiment parameters can be adjusted here.
- metrics: The evaluation metrics folder.
- model: FIAD model folder.
- solver.py: The training, validation, and testing process is implemented here.
- requirements.txt: Python packages needed to run this repository.

## Usage
1. Install Python 3.9 and PyTorch >= 1.4.0.
2. Download the datasets and place them under the `dataset/` folder.
3. To train and evaluate FIAD on a dataset, run the following command:
```bash
python main.py
```
