# README

## Introduction
This repository contains code to train and evaluate a Transformer-based model for predicting interactions between antigens and T-cell receptors (TCRs) using amino acid sequences. The model leverages PyTorch and includes functions for data preprocessing, model training, evaluation, and prediction.

## Prerequisites
- Python 3.7+
- PyTorch 1.7.1+
- pandas
- scikit-learn
- numpy

Install the required packages using:
```bash
pip install torch pandas scikit-learn numpy
```

## Code Overview

### Data Encoding and Loading
- `encode_sequence(seq, max_len)`: Encodes amino acid sequences into tensor indices.
- `create_sequence_data_loader(sequences, max_len, batch_size=32)`: Creates a DataLoader for sequences.
- `load_data(filename)`: Loads antigen, TCR, and interaction data from a CSV file.
- `prepare_datasets(antigens, tcrs, interactions)`: Prepares datasets and DataLoader for training.

### Dataset Class
- `InteractionDataset(Dataset)`: Custom dataset class for handling antigen, TCR, and interaction data.

### Transformer Model
- `make_tfm_model(T)`: Creates a Transformer model for either 'tcr' or 'antigen'.
- `make_predict_model(M_antigen, M_tcr)`: Creates the full interaction prediction model using the antigen and TCR models.

### Training and Evaluation
- `pretrain_tfm_model(model, data_loader, n_epochs)`: Pretrains the Transformer models.
- `train_model(model, data_loader, n_epochs)`: Trains the full interaction prediction model.
- `evaluate_model(model, data_loader)`: Evaluates the model and prints accuracy.
- `cross_validate_model(data, n_splits=3, epochs=5)`: Performs cross-validation and returns accuracies.

### Prediction
- `predict(M, L_antigen, L_tcr)`: Predicts interactions using the trained model.

### Model Saving and Loading
- `load_trained_model(path_to_model)`: Loads a trained model from a file.

## Usage

### 1. Data Preparation
Ensure your data is in a CSV file with columns `antigen`, `TCR`, and `interaction`. Update the file path in the code as needed.
```python
antigens, tcrs, interactions = load_data('data.csv')
full_dataset, full_loader = prepare_datasets(antigens, tcrs, interactions)
```

### 2. Pretrain Models
Pretrain the Transformer models for antigens and TCRs.
```python
antigen_model = make_tfm_model('antigen')
tcr_model = make_tfm_model('tcr')
pretrain_tfm_model(antigen_model, create_sequence_data_loader(antigens, MAX_LEN_ANTIGEN), 6)
pretrain_tfm_model(tcr_model, create_sequence_data_loader(tcrs, MAX_LEN_TCR), 6)
```

### 3. Train Full Model
Train the full interaction prediction model.
```python
full_model = make_predict_model(antigen_model, tcr_model)
train_model(full_model, full_loader, 6)
```

### 4. Evaluate Model
Evaluate the trained model on the full dataset.
```python
accuracy = evaluate_model(full_model, full_loader)
print("Full model accuracy: ", accuracy)
```

### 5. Save Model
Save the trained model to a file.
```python
torch.save(full_model.state_dict(), 'full_model.pth')
```

### 6. Cross-Validate Model
Perform cross-validation and save the results.
```python
accuracies = cross_validate_model((full_dataset.tensors[0], full_dataset.tensors[1], full_dataset.tensors[2]), n_splits=3, epochs=5)
torch.save(accuracies, '3foldtrained.pth')
```

### 7. Load and Predict
Load the trained model and make predictions.
```python
full_model.load_state_dict(torch.load('full_model.pth'))
predictions = predict(full_model, full_dataset.tensors[0], full_dataset.tensors[1])
torch.save(predictions, 'predictions.pth')
```
