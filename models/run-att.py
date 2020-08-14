import time
import math
import torch
import os
import matplotlib.pyplot as plt

from torch import nn
import torch.optim as O
import torch.nn.functional as F
from torchtext import data, vocab, datasets

import sys
sys.path.append('..')

from params import Parameters
    
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

if __name__ == '__main__':  
    
    params = Parameters()

    inputs = data.Field(
        lower=True,
        tokenize='spacy'
    )

    answers = data.Field(
        sequential=False
    )

    train, val, test = datasets.MultiNLI.splits(
            text_field=inputs,
        label_field=answers
        )

    inputs.build_vocab(train, val, test)

    if params.word_vectors:
        inputs.vocab.load_vectors(vocab.Vectors(params.glove_path, cache="."))

    answers.build_vocab(train)

    params.n_embed = len(inputs.vocab)
    params.d_out = len(answers.vocab)

    print(f"Unique tokens in inputs vocabulary: {params.n_embed}")
    print(f"Unique tokens in answers vocabulary: {params.d_out}")

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train, val, test), batch_size=params.batch_size, device=params.device)
    
    # load or instantiate model
    if params.load_model:
        model = torch.load(params.loadpath)
    else:
        model = MultiNLIModel(params.input_size, params.output_size, params.embed_size, params.device,
                      params.hidden_size, params.batch_size, params.dropout, params.n_layers, params.n_cells).to(params.device)
    
    # train
    criterion = nn.CrossEntropyLoss()
    opt = O.Adam(model.parameters(), lr=params.learning_rate)

    val_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
    log_template =  ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))

    iterations = 0
    start = time.time()

    acc_loss = []

    for epoch in range(params.epochs):
        train_iterator.init_epoch()
        n_correct, n_total = 0, 0
        for batch_idx, batch in enumerate(train_iterator):

            # switch model to training mode, clear gradient accumulators
            model.train();
            opt.zero_grad()

            iterations += 1

            # forward pass
            answer = model(batch)

            # calculate accuracy of predictions in the current batch
            n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
            n_total += batch.batch_size
            train_acc = 100. * n_correct/n_total

            loss = criterion(answer, batch.label)
            loss.backward()
            opt.step()

            # evaluate performance on validation set periodically
            if iterations % 1000 == 0:
                # switch model to evaluation mode
                model.eval()
                valid_iterator.init_epoch()

                # calculate accuracy on validation set
                n_val_correct, val_loss = 0, 0
                with torch.no_grad():
                    for val_batch_idx, val_batch in enumerate(valid_iterator):
                        answer = model(val_batch)
                        n_val_correct += (torch.max(answer, 1)[1].view(val_batch.label.size()) == val_batch.label).sum().item()
                        val_loss = criterion(answer, val_batch.label)
                val_acc = 100. * n_val_correct / len(val)

                print(log_template.format(time.time()-start,
                    epoch, iterations, 1+batch_idx, len(train_iterator),
                    100. * (1+batch_idx) / len(train_iterator), loss.item(), val_loss.item(), train_acc, val_acc))

            if iterations % 500 == 0:

                # print progress message
                print(val_log_template.format(time.time()-start,
                    epoch, iterations, 1+batch_idx, len(train_iterator),
                    100. * (1+batch_idx) / len(train_iterator), loss.item(), ' '*8, n_correct/n_total*100, ' '*12))

            acc_loss.append((loss.item(), n_correct/n_total*100))

    if params.save_model:
        torch.save(model, params.outpath)

        with open(params.outfile, "w") as output:
            output.write(str(acc_loss))
