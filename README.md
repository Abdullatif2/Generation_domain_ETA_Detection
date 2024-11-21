# Generation Domain ETA Detection

This repository contains the implementation of the methodologies described in the paper:  
**"Fine-Tuned RNN-Based Detector for Electricity Theft Attacks in Smart Grid Generation Domain"**  
Published in the IEEE Open Journal of the Industrial Electronics Society, 2022.  
[Link to Paper](https://ieeexplore.ieee.org/document/9964082)

## Overview

The code in this repository focuses on detecting electricity theft attacks (ETA) in the generation domain of smart grids using advanced deep learning techniques. The implemented models and algorithms aim to identify anomalies and attacks, ensuring the security and reliability of smart grid operations.  

The repository includes data preprocessing scripts, neural network models, and training and evaluation pipelines designed for both baseline and enhanced detection approaches.

## Repository Contents

### 1. Data Preparation
- **`Preper_data.py`**: Script to preprocess and prepare datasets for training and testing.
- **`data_processing.py`**: Handles data cleaning and transformation.
- **`data_util.py`**: Utility functions for efficient data handling.
- **`benign_ready.csv`** and **`attacked_ready.csv`**: Preprocessed datasets for benign and attacked scenarios.

### 2. Models and Training
- **`nn_models.py`**: Defines deep learning architectures for electricity theft detection.
- **`Base_line_DA.py`** and **`Base_line_DA_RNN.py`**: Scripts for training baseline models.
- **`Main_rnn.py`** and **`Main_rnn_baseline.py`**: Training scripts for RNN-based models, with and without data augmentation.
- **`main_rnn_with_DA.py`**: Trains RNN models enhanced with data augmentation techniques.

### 3. Attack Simulation
- **`Attack_Funcs.py`**: Simulates different types of electricity theft attacks in the generation domain.

### 4. Evaluation and Visualization
- **`evaluate.py`**: Script for model evaluation and performance analysis.
- **`visual_data.py`**: Generates visualizations for datasets and model performance metrics.

## How to Use

1. **Data Preparation**:  
   - Use `Preper_data.py` to preprocess raw datasets.
   - Ensure that the prepared datasets (`benign_ready.csv` and `attacked_ready.csv`) are placed in the appropriate directory.

2. **Model Training**:  
   - Run `Main_rnn.py` to train RNN models.
   - For baseline models, execute `Main_rnn_baseline.py` or `Base_line_DA.py`.

3. **Evaluation**:  
   - Use `evaluate.py` to test model performance on unseen data.

4. **Visualization**:  
   - Generate data insights and performance metrics using `visual_data.py`.

## Dependencies

Ensure all required libraries are installed. You can refer to the `requirements.txt` file for details. Common dependencies include:
- Python 3.x
- TensorFlow or PyTorch (as applicable)
- Pandas
- NumPy
- Matplotlib

## Citation

If you find this repository or the corresponding paper useful in your research, please cite as follows:

```bibtex
@ARTICLE{9964082,
  author={Eddin, Maymouna Ez and Albaseer, Abdullatif and Abdallah, Mohamed and Bayhan, Sertac and Qaraqe, Marwa K. and Al-Kuwari, Saif and Abu-Rub, Haitham},
  journal={IEEE Open Journal of the Industrial Electronics Society}, 
  title={Fine-Tuned RNN-Based Detector for Electricity Theft Attacks in Smart Grid Generation Domain}, 
  year={2022},
  volume={3},
  number={},
  pages={733-750},
  doi={10.1109/OJIES.2022.3224784}
}
