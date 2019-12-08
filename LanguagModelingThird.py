import time
import math
import os

import torch
import torch.nn as nn

from torch.autograd import Variable

from torchtext import datasets
from torchtext import data as d
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda')

def dataPre():
    # downloading the WikiText2 data and splits it into train,valid, and test datasets
    TEXT = d.Field(lower=True, batch_first=True, )
    train, valid, test = datasets.WikiText2.splits(TEXT, root='data')
    len_ = len(train[0].text)      # 2088628
    print(f'len(train[0].text\n{len(train[0].text)}')

    train_iter, valid_iter, test_iter = d.BPTTIterator.splits((train, valid, test), batch_size=batch_size,
                                                              bptt_len=bptt_len, device=0, repeat=False)

    TEXT.build_vocab(train)
    return train_iter,valid_iter,TEXT
    pass


batch_size= 50
bptt_len=50      # 反向传播通过时间,表示the sequence length the model needs toremember. The higher the number, the betterbbut the complexity of the model and the GPU memory required for the model also increase
clip = 0.25
lr = 20
log_interval = 200


class RNNModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        # ntoken represents the number of words in vocabulary.
        # ninp Embedding dimension for each word,which is the input for the LSTM
        # nlayers Number of layers required to be used in the LSTM.
        # Dropout to avoid overfitting
        # tie_weights -use the same weights for both encoder and decoder
        super().__init__()
        self.drop = nn.Dropout()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers


    #   Once we have made the weights of encoder and decoder tied，it must initialize the weights of the layer
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        # RuntimeError: Expected hidden[0] size (2, 20, 200), got (2, 30, 200)
        # 出错原因：是因为到i = 2320时，data.shape = torch.Size([30, 20]) ，但是hidden里的batch_size仍然为30
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        s = output.size()
        decoded = self.decoder(output.view(s[0] * s[1], s[2]))
        return decoded.view(s[0], s[1], decoded.size(1)), hidden

    # For an LSTM model, along with the input, we also need to pass the hidden variables
    # The init_hidden function will take the batch size as input and then return a hidden variable, which can be used along with the inputs
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))


criterion = nn.CrossEntropyLoss()



emsize = 200
nhid=200
nlayers=2
dropout = 0.2

train_iter,valid_iter,TEXT = dataPre()
ntokens = len(TEXT.vocab)           # ntokens = 28913
print(f'ntoken\n{ntokens}')
lstm = RNNModel(ntokens, emsize, nhid,nlayers, dropout, 'store_true')
lstm = lstm.cuda()


# Using the previous values of the hidden state
def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.Tensor:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)



def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    lstm.eval()
    total_loss = 0
    hidden = lstm.init_hidden(batch_size)
    for batch in data_source:
        data, targets = batch.text,batch.target.view(-1)
        output, hidden = lstm(data.cuda(), hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets.cuda()).data
        hidden = repackage_hidden(hidden)
    return total_loss[0]/(len(data_source.dataset[0].text)//batch_size)






def trainf(epoch):
    # Turn on training mode which enables dropout.
    lstm = RNNModel(ntokens, emsize, nhid, nlayers, dropout, 'store_true')
    lstm = lstm.cuda()
    lstm.train()
    total_loss = 0
    start_time = time.time()
    hidden = lstm.init_hidden(batch_size)
    for  i,batch in enumerate(train_iter):
        # print(f'i:{i}')

        # batch =
        # [torchtext.data.batch.Batch of size 30]
        # 	[.text]:[torch.LongTensor of size 30x30]
        # 	[.target]:[torch.LongTensor of size 30x30]
        m = batch
        temp = i
        # if i > 2200:
        #     break
        data, targets = batch.text,batch.target.view(-1)        # data.shape = torch.Size([30, 30]),      targets.shape = torch.Size([900])
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)    # hidden[0].shape = torch.Size([2, 30, 200])
        lstm.zero_grad()
        temp = data
        output, hidden = lstm(data.cuda(), hidden)
        loss = criterion(output.view(-1, ntokens), targets.cuda())
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(lstm.parameters(), clip)

        # Implementing an optimizer manually gives more flexibility than using a prebuilt optimizer
        #  iterating through all the parameters and adding up the value of the gradients, multiplied by the learning rate
        for p in lstm.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        # if i % log_interval == 0 and i > 0 and hidden[0].size == [2,30,200]:
        if i % log_interval == 0 and i > 0 :
        # if i > 2317:
            print(f'i:{i}')
            # if hidden[0].size != [2,30,200]
            #     break

            cur_loss = total_loss.item() / log_interval
            elapsed = time.time() - start_time
            (print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(epoch, i, len(train_iter), lr,elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss))))
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
best_val_loss = None
epochs = 40
def trainEpoch():
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        trainf(epoch)
        # train_iter, valid_iter = dataPre()
        # val_loss = evaluate(valid_iter)
        # print('-' * 89)
        # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        #       'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
        #                                  val_loss, math.exp(val_loss)))
        # print('-' * 89)
        # if not best_val_loss or val_loss < best_val_loss:
        #     best_val_loss = val_loss
        # else:
        #     # Anneal the learning rate if no improvement has been seen in the validation dataset.
        #     lr /= 4.0
    pass

def main():
    # dataPre()
    trainEpoch()
    pass

if __name__ == '__main__':
    main()




#
# # Some part of the code was referenced from below.
# # https://github.com/pytorch/examples/tree/master/word_language_model
# import torch
# import torch.nn as nn
# import numpy as np
# from torch.nn.utils import clip_grad_norm_
# from data_utils import Dictionary, Corpus
#
# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # Hyper-parameters
# embed_size = 128
# hidden_size = 1024
# num_layers = 1
# num_epochs = 5
# num_samples = 1000  # number of words to be sampled
# batch_size = 20
# seq_length = 30
# learning_rate = 0.002
#
# # Load "Penn Treebank" dataset
# corpus = Corpus()
# ids = corpus.get_data('data/train.txt', batch_size)
# vocab_size = len(corpus.dictionary)
# num_batches = ids.size(1) // seq_length
#
#
# # RNN based language model
# class RNNLM(nn.Module):
#     def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
#         super(RNNLM, self).__init__()
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
#         self.linear = nn.Linear(hidden_size, vocab_size)
#
#     def forward(self, x, h):
#         # Embed word ids to vectors
#         x = self.embed(x)
#
#         # Forward propagate LSTM
#         out, (h, c) = self.lstm(x, h)
#
#         # Reshape output to (batch_size*sequence_length, hidden_size)
#         out = out.reshape(out.size(0) * out.size(1), out.size(2))
#
#         # Decode hidden states of all time steps
#         out = self.linear(out)
#         return out, (h, c)
#
#
# model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)
#
# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#
# # Truncated backpropagation
# def detach(states):
#     return [state.detach() for state in states]
#
#
# # Train the model
# for epoch in range(num_epochs):
#     # Set initial hidden and cell states
#     states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
#               torch.zeros(num_layers, batch_size, hidden_size).to(device))
#
#     for i in range(0, ids.size(1) - seq_length, seq_length):
#         # Get mini-batch inputs and targets
#         inputs = ids[:, i:i + seq_length].to(device)
#         targets = ids[:, (i + 1):(i + 1) + seq_length].to(device)
#
#         # Forward pass
#         states = detach(states)
#         outputs, states = model(inputs, states)
#         loss = criterion(outputs, targets.reshape(-1))
#
#         # Backward and optimize
#         model.zero_grad()
#         loss.backward()
#         clip_grad_norm_(model.parameters(), 0.5)
#         optimizer.step()
#
#         step = (i + 1) // seq_length
#         if step % 100 == 0:
#             print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
#                   .format(epoch + 1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))
#
# # Test the model
# with torch.no_grad():
#     with open('sample.txt', 'w') as f:
#         # Set intial hidden ane cell states
#         state = (torch.zeros(num_layers, 1, hidden_size).to(device),
#                  torch.zeros(num_layers, 1, hidden_size).to(device))
#
#         # Select one word id randomly
#         prob = torch.ones(vocab_size)
#         input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)
#
#         for i in range(num_samples):
#             # Forward propagate RNN
#             output, state = model(input, state)
#
#             # Sample a word id
#             prob = output.exp()
#             word_id = torch.multinomial(prob, num_samples=1).item()
#
#             # Fill input with sampled word id for the next time step
#             input.fill_(word_id)
#
#             # File write
#             word = corpus.dictionary.idx2word[word_id]
#             word = '\n' if word == '<eos>' else word + ' '
#             f.write(word)
#
#             if (i + 1) % 100 == 0:
#                 print('Sampled [{}/{}] words and save to {}'.format(i + 1, num_samples, 'sample.txt'))
#
# # Save the model checkpoints
# torch.save(model.state_dict(), 'model.ckpt')