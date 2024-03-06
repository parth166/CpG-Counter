import streamlit as st
import random
import torch
import numpy as np
import random
import torch.nn as nn
from typing import Sequence
from functools import partial
import re

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

def my_model(input):
    output_text = "Processed output: "

    model = torch.load('model')
    model.eval()

    outputs = model(input)
    ans = str(outputs.item())

    return output_text + ans

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

    alphabet = 'NACGT'
    dna2int = { a: i for a, i in zip(alphabet, range(5))}

    dnaseq_to_intseq = partial(map, dna2int.get)

    output = list(dnaseq_to_intseq(cleaned_string))
    return torch.tensor(output, dtype=torch.int32)

def main():
    st.title('CpG counter')

    # Create a text input widget in the Streamlit frontend.
    user_input = st.text_input("Enter your input here:")

    if st.button('Process'):
        # Check if there is any input.
        if user_input:
            # Call the model function with the user input.
            if is_valid_input(user_input):
                processed_input = process_input(user_input)
                result = my_model(processed_input)
                # Display the result in the frontend.
                st.write("Model Output:", result)
            else:
                st.write("String contains characters other than NACGT! Check Input!")
        else:
            # Prompt the user to enter an input if they haven't already.
            st.write("Please enter an input.")

if __name__ == '__main__':
    main()