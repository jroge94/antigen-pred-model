import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np



MAX_LEN_ANTIGEN = 21 
MAX_LEN_TCR = 21
EMBEDDING_DIM = 256
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWYX'

char_to_index = {ch: idx for idx, ch in enumerate(AMINO_ACIDS)}

# Encodes a sequence of amino acids into a tensor of indices
def encode_sequence(seq, max_len):
    encoded = torch.zeros(max_len, dtype=torch.long)
    for i, char in enumerate(seq[:max_len]):
        encoded[i] = char_to_index.get(char, char_to_index['X'])  # 'X' for unknown or padding
    return encoded

# Necessary for DataLoader
def create_sequence_data_loader(sequences, max_len, batch_size=32):
    encoded_sequences = [encode_sequence(seq, max_len) for seq in sequences]
    sequences_tensor = torch.stack(encoded_sequences)
    dataset = TensorDataset(sequences_tensor)  # Ensure this is just one tensor per batch
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_data(filename):
    data = pd.read_csv(filename)
    data.columns = data.columns.str.strip()
    print("Columns:", data.columns)  # Print column names to verify
    return data['antigen'], data['TCR'], data['interaction']

def prepare_datasets(antigens, tcrs, interactions):
    # Encode and prepare tensors
    antigen_seqs = [encode_sequence(seq, MAX_LEN_ANTIGEN) for seq in antigens]
    tcr_seqs = [encode_sequence(seq, MAX_LEN_TCR) for seq in tcrs]
    interactions_tensor = torch.tensor(interactions, dtype=torch.float32)

    # Stack sequences into tensors
    antigen_tensor = torch.stack(antigen_seqs)
    tcr_tensor = torch.stack(tcr_seqs)

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(antigen_tensor, tcr_tensor, interactions_tensor)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataset, data_loader


# Define the dataset class
class InteractionDataset(Dataset):
    def __init__(self, antigens, tcrs, interactions):
        self.antigens = torch.stack(antigens)
        self.tcrs = torch.stack(tcrs)
        self.interactions = interactions

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        return self.antigens[idx], self.tcrs[idx], self.interactions[idx]
    
# Define the Transformer model
def pretrain_tfm_model(model, data_loader, n_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        total_loss = 0
        for batch in data_loader:
            #
            seq = batch[0].to(device) 

            optimizer.zero_grad()
            input_seq = seq[:, :-1]  
            target_seq = seq[:, 1:]  
            output = model(input_seq)
            loss = criterion(output.permute(0, 2, 1), target_seq)  
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(data_loader)}")

    
def make_predict_model(M_antigen, M_tcr):
    model = InteractionModel(M_antigen, M_tcr)
    return model

class InteractionModel(nn.Module):
    def __init__(self, M_antigen, M_tcr):
        super().__init__()
        
        self.antigen_model = M_antigen
        self.tcr_model = M_tcr
      
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, tcr_seq, antigen_seq):
        tcr_embedding = torch.mean(self.tcr_model(tcr_seq), dim=1)
        antigen_embedding = torch.mean(self.antigen_model(antigen_seq), dim=1)
        combined = torch.cat((tcr_embedding, antigen_embedding), dim=1)
        return self.classifier(combined).squeeze()

def make_tfm_model(T):
    assert T in ['tcr', 'antigen'], "Invalid type specified. Choose either 'tcr' or 'antigen'"
    # Configuration for the transformer model
    embedding_size = 256
    num_heads = 8
    num_layers = 3
    num_tokens = len(AMINO_ACIDS) + 1  # Adding 1 for padding token to account for any possible padding

    # Create an embedding layer to convert token indices into embeddings
    embedding = nn.Embedding(num_tokens, embedding_size)

    # Define a single Transformer encoder layer
    encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, batch_first=True)

    # Stack multiple encoder layers to form the transformer encoder
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    
    return nn.Sequential(embedding, transformer_encoder)



def train_model(model, data_loader, n_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        total_loss = 0
        for antigens, tcrs, interactions in data_loader:  
            antigens, tcrs, interactions = antigens.to(device), tcrs.to(device), interactions.to(device)

            optimizer.zero_grad()
            predictions = model(antigens, tcrs)
            loss = criterion(predictions.squeeze(), interactions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(data_loader)}")

def evaluate_model(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() 
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for antigens, tcrs, interactions in data_loader:
         
            antigens, tcrs, interactions = antigens.to(device), tcrs.to(device), interactions.to(device)
            predictions = model(antigens, tcrs)
            total_correct += (predictions.round() == interactions).sum().item()
            total_samples += interactions.size(0)
    accuracy = total_correct / total_samples
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

import torch

def load_trained_model(path_to_model):
    data = torch.load(path_to_model)
    model = data['full_model.pth']
    return model


def predict(M, L_antigen, L_tcr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = M.to(device) 
    L_antigen = L_antigen.to(device)  #
    L_tcr = L_tcr.to(device)  

    if L_antigen.dim() == 4:  
        L_antigen = L_antigen.squeeze(0)
    if L_tcr.dim() == 4:
        L_tcr = L_tcr.squeeze(0)

    pred = M(L_antigen, L_tcr)
    return pred


def cross_validate_model(data, n_splits=3, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

   
    antigens, tcrs, interactions = data

    for train_index, test_index in kf.split(antigens): 
     
        train_data = TensorDataset(antigens[train_index], tcrs[train_index], interactions[train_index])
        test_data = TensorDataset(antigens[test_index], tcrs[test_index], interactions[test_index])

     
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

       
        M_antigen = make_tfm_model('antigen').to(device)
        M_tcr = make_tfm_model('tcr').to(device)
        M = make_predict_model(M_antigen, M_tcr)

       
        train_model(M, train_loader, epochs)

        # Evaluate model
        M.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for antigen_seq, tcr_seq, interaction in test_loader:
                antigen_seq, tcr_seq, interaction = antigen_seq.to(device), tcr_seq.to(device), interaction.to(device)
                pred_interaction = M(antigen_seq, tcr_seq)
                predicted = (pred_interaction.squeeze() > 0.5).float()
                correct += (predicted == interaction).sum().item()
                total += interaction.size(0)
        accuracy = correct / total
        accuracies.append(accuracy)

    average_accuracy = np.mean(accuracies)
    print(f"Average accuracy over {n_splits} folds: {average_accuracy * 100:.2f}%")
    return accuracies

import torch

if __name__ == '__main__':
    
    print("Loading data...")
    # Ensure load_data and prepare_datasets are correctly defined and handle exceptions
    antigens, tcrs, interactions = load_data('data.csv')
    full_dataset, full_loader = prepare_datasets(antigens, tcrs, interactions)
    
    print("Pretraining models...")
    # Verify that make_tfm_model and pretrain_tfm_model functions are properly defined
    antigen_model = make_tfm_model('antigen')
    tcr_model = make_tfm_model('tcr')
    pretrain_tfm_model(antigen_model, create_sequence_data_loader(antigens, MAX_LEN_ANTIGEN), 6)
    pretrain_tfm_model(tcr_model, create_sequence_data_loader(tcrs, MAX_LEN_TCR), 6)
    
    print("Model pretraining complete")
    
    print("Training full model...")
    # Ensure make_predict_model and train_model are appropriately defined
    full_model = make_predict_model(antigen_model, tcr_model)
    train_model(full_model, full_loader, 6)
    
    print("Full model training complete")
    
    print("Evaluating model...")
    # Check evaluate_model function definition
    print("Full model accuracy: ", evaluate_model(full_model, full_loader))
    
    print("Model evaluation complete")
    
    print("Saving model...")
    # Check the path and permissions
    torch.save(full_model.state_dict(), 'full_model.pth')
    print("Model saved")
    
    print("Cross-validating model...")
    # Ensure cross_validate_model function is defined correctly
    accuracies = cross_validate_model((full_dataset.tensors[0], full_dataset.tensors[1], full_dataset.tensors[2]), n_splits=3, epochs=5)
    torch.save(accuracies, '3foldtrained.pth')
    
    print("Cross-validation complete")
    
    # Save cross-validation accuracies if needed
    torch.save(accuracies, 'accuracies.pth')
    
    print("Loading trained model...")
    # Loading model state into an existing model structure
    full_model.load_state_dict(torch.load('full_model.pth'))
    
    # Ensure predict function is defined with correct handling of device assignments
    predictions = predict(full_model, full_dataset.tensors[0], full_dataset.tensors[1])
    
    torch.save(predictions, 'predictions.pth')

    
    
    