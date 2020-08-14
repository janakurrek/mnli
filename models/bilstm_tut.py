import math
import torch
from torch import nn
import torch.optim as O
import torch.nn.functional as F


class MultiNLIModel(nn.Module):
    def __init__(self, input_size, output_size, embed_size, device,
                 hidden_size, batch_size, dropout, n_layers, n_cells):
        
        super(MultiNLIModel, self).__init__()
        
        self.device = device
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_cells = n_cells
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers=n_layers, dropout=dropout, 
                            bidirectional=True)
        self.fc_hidden1 = nn.Linear(hidden_size * 2, batch_size, bias=False)
        self.fc_hidden2 = nn.Linear(batch_size, hidden_size * 2, bias=False)
        self.fc_hidden3 = nn.Linear(hidden_size * 2,  output_size, bias=False)
    
    def encode(self, embed):
        state_shape = self.n_cells, self.batch_size, self.hidden_size
        h0 = c0 = embed.new_zeros(state_shape)
        outputs, (ht, ct) = self.lstm(embed, (h0, c0))
        return ht[-2:].transpose(0, 1).contiguous().view(self.batch_size, -1)
    
    def forward(self, pair):
        
        # conditionally update batch size in linear layers
        if pair.batch_size != self.batch_size:
            self.batch_size = pair.batch_size
            self.fc_hidden1 = nn.Linear(self.hidden_size * 2, pair.batch_size, bias=False).to(self.device)
            self.fc_hidden2 = nn.Linear(pair.batch_size, self.hidden_size * 2, bias=False).to(self.device)

        # seq_length, batch_size, embed_size
        prem_embed = self.embed(pair.premise)
        hypo_embed = self.embed(pair.hypothesis)
        
        # fix word embeddings
        prem_embed.detach()
        hypo_embed.detach()
        
        # batch_size, hidden_size * 2
        prem_embed = self.encode(prem_embed)
        hypo_embed = self.encode(hypo_embed)
        
        # batch_size, batch_size
        prem_embed = self.relu(self.fc_hidden1(prem_embed))
        hypo_embed = self.relu(self.fc_hidden1(hypo_embed))
        
        # batch_size, hidden_size * 2
        pair_embed = prem_embed - hypo_embed
        pair_embed = self.relu(self.fc_hidden2(pair_embed))
        
        # hidden_size * 2, output_size
        pair_output = self.relu(self.fc_hidden3(pair_embed))
        
        return pair_output