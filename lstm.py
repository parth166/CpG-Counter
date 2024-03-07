import streamlit as st
import random
import torch
import numpy as np
import random
import torch.nn as nn
from typing import Sequence
from functools import partial
import re
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class VariableLengthDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = [torch.tensor(s, dtype=torch.long) for s in sequences]
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], len(self.sequences[idx])

class PadSequence:
    def __init__(self, padded_length=None, batch_first=True):
        self.padded_length = padded_length  # Maximum sequence length for padding
        self.batch_first = batch_first  # Whether the batch size is the first dimension

    def __call__(self, batch):
        # Extract sequences and labels from the batch
        sequences, labels, lengths = zip(*[(x[0], x[1], x[2]) for x in batch])
        # Pad sequences
        sequences_padded = pad_sequence(sequences, batch_first=self.batch_first, padding_value=5)
        
        # Optionally truncate sequences to a fixed length
        if self.padded_length is not None:
            sequences_padded = sequences_padded[:, :self.padded_length] if self.batch_first else sequences_padded[:self.padded_length, :]
        
        # Convert labels to a tensor
        labels = torch.tensor(labels, dtype=torch.float)
        # Convert lengths to a tensor
        lengths = torch.tensor(lengths, dtype=torch.long)

        return sequences_padded, labels, lengths

class VariableLengthLSTM(nn.Module):
    def __init__(self, hidden_dim, num_layers=2):
        super(VariableLengthLSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=6, embedding_dim=10)
        self.lstm = nn.LSTM(input_size=10, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, lengths):
        x = self.embedding(x)
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(packed_input)
        output = self.classifier(ht[-1])
        return output

class CpGPredictor(torch.nn.Module):
    ''' Simple model that uses a LSTM to count the number of CpGs in a sequence '''
    def __init__(self, hidden_size=32, num_layers=4):
        super(CpGPredictor, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=5, embedding_dim=5)
        self.lstm = nn.LSTM(input_size=5, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Forward propagate through LSTM
        x = self.embedding(x)
        output, (hn, cn) = self.lstm(x)
        last_output = output[-1, :]
        logits = self.classifier(last_output)
        return logits

def model_v1(input):
    output_text = "Processed output: "

    model = torch.load('model')
    model.eval()

    outputs = model(input)
    ans = str(outputs.item())

    return output_text + ans

def model_v2(dataloader):
    criterion = nn.MSELoss()
    model = torch.load('model_v3')
    model.eval()

    total_error = 0
    for batch in dataloader:
        sequences_padded, labels, lengths = batch
        outputs = model(sequences_padded, lengths)
        return str(outputs.item())

# Use this for getting y label
def count_cpgs(seq: str) -> int:
    cgs = 0
    for i in range(0, len(seq) - 1):
        dimer = seq[i:i+2]
        # note that seq is a string, not a list
        if dimer == "CG":
            cgs += 1
    return cgs

def is_valid_input(inp):
    cleaned_string = re.sub(r'[^a-zA-Z]', '', inp)

    return not re.search(r'[^NACGTnacgt]', cleaned_string)

def process_input(inp):
    cleaned_string = re.sub(r'[^a-zA-Z]', '', inp)
    cleaned_string = cleaned_string.upper()

    alphabet = 'NACGT'
    dna2int = { a: i for a, i in zip(alphabet, range(5))}

    dnaseq_to_intseq = partial(map, dna2int.get)
    output = [list(dnaseq_to_intseq(cleaned_string))]

    dataset = VariableLengthDataset(torch.tensor(output, dtype=torch.int32), torch.tensor([count_cpgs(cleaned_string)], dtype=torch.float))
    pad_sequence_fn = PadSequence(batch_first=True)
    train_data_loader = DataLoader(dataset, batch_size=1, collate_fn=pad_sequence_fn)
    return train_data_loader
    # output = list(dnaseq_to_intseq(cleaned_string))
    # return torch.tensor(output, dtype=torch.int32)

def main():
    # ip = helper()
    # print(ip)
    st.title('CpG counter')

    # Create a text input widget in the Streamlit frontend.
    user_input = st.text_input("Enter your input here:")

    if st.button('Process'):
        # Check if there is any input.
        if user_input:
            # Call the model function with the user input.
            if is_valid_input(user_input):
                processed_input = process_input(user_input)
                # result = model_v1(processed_input)
                result = model_v2(processed_input)
                # Display the result in the frontend.
                st.write("Model Output:", result)
            else:
                st.write("String contains characters other than NACGT! Check Input!")
        else:
            # Prompt the user to enter an input if they haven't already.
            st.write("Please enter an input.")

if __name__ == '__main__':
    main()