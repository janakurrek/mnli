from bilstm_att import MultiNLIModel
from torchtext import data, vocab, datasets

import sys
sys.path.append('..')

from params import Parameters

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
    
    if params.load_model:
        model = torch.load(params.loadpath)
    else:
        model = MultiNLIModel(params.input_size, params.output_size, params.embed_size, params.device,
                      params.hidden_size, params.batch_size, params.dropout, params.n_layers, params.n_cells).to(params.device)
    
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

    if params.save_model:
        torch.save(model, params.outpath)

        with open(params.outfile, "w") as output:
            output.write(str(acc_loss))
