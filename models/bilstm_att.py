import math
import torch
from torch import nn
import torch.optim as O
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 4, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.reshape((1, hidden.shape[1], hidden.shape[2] * 2))
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        catted = torch.cat([hidden, encoder_outputs], 2)
        energy = F.relu(self.attn(catted))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]
    
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
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers=n_layers, dropout=dropout, 
                            bidirectional=True)
        
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.fc_output = nn.Linear(hidden_size,  output_size, bias=False)
    
    def encode(self, embed):
        # pass embedding input through lstm
        state_shape = self.n_cells, self.batch_size, self.hidden_size
        h0 = c0 = embed.new_zeros(state_shape)
        outputs, (ht, ct) = self.lstm(embed, (h0, c0))

        # pass outcomes through attention layer
        weights = self.attention(ht[-2:], outputs)
        context = weights.bmm(outputs.transpose(0, 1))
        context = context.transpose(0, 1)
        context = context.squeeze(0)
        return context
        
    def forward(self, pair):
        # batch size discrepancy
        if self.batch_size != pair.batch_size:
            self.batch_size = pair.batch_size
        
        # seq_length, batch_size, embed_size
        prem_embed = self.dropout(self.embed(pair.premise))
        hypo_embed = self.dropout(self.embed(pair.hypothesis))
        
        prem_contx = self.encode(prem_embed)
        hypo_contx = self.encode(hypo_embed)
        
        # seq_len, hidden_size * 2
        pair_embed = prem_contx - hypo_contx
        pair_embed = self.relu(self.fc_hidden(pair_embed))
        
        # hidden_size * 2, output_size
        pair_output = self.relu(self.fc_output(pair_embed))
        
        return pair_output