import time
import math
import torch
import os
import matplotlib.pyplot as plt

from torch import nn
import torch.optim as O
import torch.nn.functional as F
from torchtext import data, vocab, datasets

class Parameters():
    def __init__(self):
        # gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # word vectors
        self.embed_size = 50
        self.word_vectors = True
        self.glove_path = '/home/ndg/users/jkurre/mnli/utils/embeddings/glove.6B.50d.txt'
        # model configs
        self.hidden_size = 1024
        self.batch_size = 32
        self.input_size = 76790
        self.output_size = 4
        self.n_layers = 2
        self.n_cells = 4
        self.dropout = 0.5
        # training
        self.epochs = 15
        self.learning_rate = 0.0001
        self.outpath = '/home/ndg/users/jkurre/mnli/models/bilstm_revised.pt' # _onehot.pt

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
    
    model = MultiNLIModel(params.input_size, params.output_size, params.embed_size, params.device,
                      params.hidden_size, params.batch_size, params.dropout, params.n_layers, params.n_cells).to(params.device)
    
    # https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

    criterion = nn.CrossEntropyLoss()
    opt = O.Adam(model.parameters(), lr=params.learning_rate)

    val_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
    log_template =  ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))

    iterations = 0
    start = time.time()

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
            if iterations % 2000 == 0:
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

    torch.save(model, params.outpath)
