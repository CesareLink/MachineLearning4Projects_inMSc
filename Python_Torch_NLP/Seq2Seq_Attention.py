# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:01:19 2020

@author: Cesare
"""
#______________________________________________________________________________
#Module input
import os
import sys
import math
from collections import Counter
import numpy as np
import random 

import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk  #tokenize for English sentence

#______________________________________________________________________________
#DataLoader
def load_data(in_file):
    '''
    Parameters
    ----------
    in_file : name or road for dataset, usually txt
        The dataset of the target.

    Returns
    -------
    en : list
        en data in form of [BOS,A,B,C,D,EOS,BOS,D,E,F,G,EOS,...]
    cn : list
        cn data.

    '''
    cn = [] #the list of cn data
    en = [] #the list of en data
    num_examples = 0 #count the number of examples
    with open(in_file, 'r') as f: #load the data in in_file, read only
        for line in f: #load every line in f
            line = line.strip().split('\t') #tonkenize with \t(space)
            #lower the characters and tokenize with nltk, then load into en[]
            en.append(['BOS'] + nltk.work_tokenize(line[0].lower()) + ['EOS'])
            #split chinese sentence into characters, and load into cn[]
            cn.append(['BOS'] + [c for c in line[1]] + ['EOS'])
    return en, cn

#______________________________#
#Main#-prepare the data in txt
train_file = 'nmt/en-cn/train.txt'
dev_file = 'nmt/en-cn/dev.txt'
train_en, train_cn = load_data(train_file)  
#list[a,b,c,d,...] every character means a sentence
dev_en, dev_cn =load_data(dev_file)
#HERE：we got a dataset with [BOS, words, words,EOS]
#______________________________________________________________________________
#Vocabulary List Setting
UNK_IDX = 0
PAD_IDX = 1
def build_dict(sentences, max_words=50000):
    '''
    Parameters
    ----------
    sentences : list
        The dataset from dataloader.
    max_words : int, optional
        The max word number of the dictionary. The default is 50000.

    Returns
    -------
    word_dict : dictionary [word:number]
        The word dictionary.
    total_words : int
        The number of the total words used.

    '''
    word_count = Counter() #words as key, count as value
    #count the frequency of the mostly-used words
    for sentence in sentences: #loop for sentence load
        for s in sentence: #loop for words read
            word_count[s] += 1 #count the frequency of 's'
    ls = word_count.most_common(max_words) 
    #counter.most_common(n)-the most used n words
    #ls:[word:number-frequency] order in common frequency
    total_words = len(ls) + 2 #add unk and pad, then get the total words number
    word_dict = {w[0]: index+2 for index, w in enumerate(ls)} #+2 for unk and pad
    #enumerate can return a list in form of [(0,'a'),(1,'b')]
    #word_dict[word:order]
    #set unk and pad, the word_dict setted from the list 
    word_dict['UNK'] = UNK_IDX #word as key
    word_dict['PAD'] = PAD_IDX
    return word_dict, total_words #[word:order],int

#______________________________#
#Main#-set dictionary from dataloader
#set dictionary and total words number for en and cn
#en_dict:[word:order],en_total_words:int_number
en_dict, en_total_words = build_dict(train_en)
cn_dict, cn_total_words = build_dict(train_cn)
#reshape the dictionary with [order:word]
inv_en_dict = {v: k for k, v in en_dict.items()}
inv_cn_dict = {v: k for k, v in cn_dict.items()}
print(en_dict[6])
#______________________________________________________________________________
#Encode
#-not encoder, just turn these sentences into orders for data acquistion
def encode(en_sentences, cn_sentences, en_dict, cn_dict, sort_by_len=True):
    '''
    Parameters
    ----------
    en_sentences : list
        the dataset of en sentences with [a,b,...] every character is a sentence
    cn_sentences : list
        the dataset of cn sentences with [a,b,...] every character is a sentence.
    en_dict : dictionary
        the dictionary corresponding to the dataset en.
    cn_dict : dictionary
        the dictionary corresponding to the dataset cn.
    sort_by_len : True or False, optional
        whether sort by length. The default is True.

    Returns
    -------
    out_en_sentences : list in list
        with list there are lists of the words of the sentences.[[abcd],[dege]]
    out_cn_sentences : list in list
        with list there are lists of the words of the sentences.[[abcd],[dege]].
        
    Description
    -------
    Set a list of dictionary for dataset, this is the encode

    '''
    #load sentences from en_dict and cn_dict
    length = len(en_sentences)
    #.get(w, 0)-return the value of the key
    #here the en/cn_sentences are lists of sentences [BOS,A,B,C,EOS,BOS,S,D,E,EOS]
    #these two sentence mean that {for every word in data, we got a corresponding
    # dictionary value(order) of key(w)}
    out_en_sentences = [[en_dict.get(w, 0) for w in sent] for sent in en_sentences]
    out_cn_sentences = [[cn_dict.get(w, 0) for w in sent] for sent in cn_sentences]
    
    #sort sentences by english lengths
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x:len(seq[x]))
    
    #sort cn and en in same order
    if sort_by_len:
        #set the sentences with order in english
        sorted_index = len_argsort(out_en_sentences)
        out_en_sentences = [out_en_sentences[i] for i in sorted_index]
        out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]
    #out_en/cn)sentences in form of [[order,order],[order,order],...]   
    #[number_sentences*sentence_content_order]
    return out_en_sentences, out_cn_sentences

#______________________________#
#Main#-Set up the train set and dev set in code(order)
train_en, train_cn = encode(train_en, train_cn, en_dict, cn_dict)
dev_en, dev_cn = encode(dev_en, dev_cn, en_dict, cn_dict)

#______________________________________________________________________________
#Batch Preparation
#Get minibatches for dataset:devide all the sentences into batches for training
def get_minibatches(n, minibatch_size, shuffle=True):
    '''
    Parameters
    ----------
    n : int
        the lengths of dataset.
    minibatch_size : int
        the size of minibatches.
    shuffle : Boolean, optional
        Whether shuffle or not. The default is True.

    Returns
    -------
    minibatches : Tensor[[],[],...] 
        [minibatch_number * minibatch_size].

    '''
    #n is the length of en_sentences
    idx_list = np.arange(0, n, minibatch_size) #[0, minibatch_size, 2*minibatch_size,... n]
    #0 as the beginning point, n as the final point, minibatch_size is the step length
    if shuffle:
        np.random.shuffle(idx_list) #shuffle the idx_list
    minibatches = [] #set the minibatches list
    for idx in idx_list: #idx_list has splited the whole list into n(length of data)/minibatch_size
        #then for every batch, set up a list in list with items of [idx, ..., idx+1)
        #start at idx, final at min of XXX, step = 1
        minibatches.append(np.arange(idx, min(idx+minibatch_size, n)))
    return minibatches #[minibatch_number*minibatch_size] with 0 in each list

#
def prepare_data(seqs):
    '''    
    Parameters
    ----------
    seqs : list in list
        [minibatch_size*[sentence_content_order]].

    Returns
    -------
    x : matrix in 2D
        [size of minibatch*data_order in each sentence].
    x_lengths : array in 1D
        [each length of each sentence].

    '''
    #get the length of each sequence in the dataset
    lengths = [len(seq) for seq in seqs]
    #lengths = [batch_size*[int(length of sequence)]]
    n_samples = len(seqs) #n_samples = the number of sentences in minibatch
    max_len = np.max(lengths) #the max length of one sentence   
    
    #set a matrix in form of [size of minibatch * max length of sentences]
    x = np.zeros((n_samples, max_len)).astype('int32')
    #one-dimension array for sequence, turn the list into array for improving speed
    x_lengths = np.array(lengths).astype('int32') #[size of minibatch]
    #fill x with the order_idx*lengths[idx]
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq  #[idx,startpoint:endpoint(:step)]
    return x, x_lengths #return x:matrix, x_lengths

def gen_examples(en_sentences, cn_sentences, batch_size):
    '''
    Parameters
    ----------
    en_sentences : list in list
        [number_sentences*[sentence_content_order]].
    cn_sentences : list in list
        [number_sentences*[sentence_content_order]].
    batch_size : int
        the minibatch size.

    Returns
    -------
    all_ex : [list in list and then fill the tuple]:[([],[])]
        with four list in eacn tuple. The en-encode of sentences in minibatch,
        its length of each sentences, and these of cn words.

    '''
    #prepare to devide the dataset into batch*samples
    minibatches = get_minibatches(len(en_sentences), batch_size)
    #minibatches : [[],[],] with 0 in list
    all_ex = []
    for minibatch in minibatches:
        #for each batch of sentence in order form, load the orders
        #en/cn_sentences: [number_sentences*[sentence_content_order]]
        mb_en_sentences = [en_sentences[t] for t in minibatch] 
        mb_cn_sentences = [cn_sentences[t] for t in minibatch]
        #mb_en/cn_sentences: [minibatch_size*[sentence_content_order]]
        mb_x, mb_x_len = prepare_data(mb_en_sentences) 
        mb_y, mb_y_len = prepare_data(mb_cn_sentences)
        #mb_x/y:matrix[size of minibatch*data_order in each sentence]
        #mb_x/y_len:array[each length of each sentence]
        all_ex.append((mb_x, mb_x_len, mb_y, mb_y_len))
        #all_ex:[([minibatch_size*order],[minibatch_size-lengths-int],
        #[minibatch_size*order],[minibatch_size-lengths-int])]
    return all_ex

#______________________________#
#Main#-set up teh train_data and dev_data
batch_size = 64
train_data = gen_examples(train_en, train_cn, batch_size) #input [[order,order],...]
random.shuffle(train_data) #shuffle
dev_data = gen_examples(dev_en, dev_cn, batch_size)




#______________________________________________________________________________
#Encoder and decoder with attention
class Encoder(nn.Module):
    '''
    the encoder with attention output
    '''
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        #1 layer of embedding,1 layer of GRU, 1 dropout, 1 linear
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, x, lengths):
        #forward pass-load idx and length
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx.long()]
        embedded = self.dropout(self.embed(x_sorted))
        #
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(), batch_first=True)
        #get output and hidden state
        packed_out, hid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()
        #splice the hidden state
        hid = torch.cat([hid[-2], hid[-1]], dim=1)
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)

        return out, hid

#Attention module
class Attention(nn.Module):
    '''
    A two layer linear network as the attention module
    return: output-the output of attention layer, attn-the matrix of attention transformer
    '''
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()
        #set the encoder and decoder hidden size for attention tranform 
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        #2 layer of linear as the attention model
        self.linear_in = nn.Linear(enc_hidden_size*2, dec_hidden_size, bias=False)
        self.linear_out = nn.Linear(enc_hidden_size*2 + dec_hidden_size, dec_hidden_size)
        
    def forward(self, output, context, mask):
        # output: batch_size, output_len, dec_hidden_size
        # context: batch_size, context_len, 2*enc_hidden_size
    
        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1)
        
        context_in = self.linear_in(context.view(batch_size*input_len, -1)).view(                
            batch_size, input_len, -1) # batch_size, context_len, dec_hidden_size
        
        # context_in.transpose(1,2): batch_size, dec_hidden_size, context_len 
        # output: batch_size, output_len, dec_hidden_size
        attn = torch.bmm(output, context_in.transpose(1,2)) 
        # batch_size, output_len, context_len
        attn.data.masked_fill(mask, -1e6)
        attn = F.softmax(attn, dim=2) 
        # batch_size, output_len, context_len
        context = torch.bmm(attn, context) 
        # batch_size, output_len, enc_hidden_size
        output = torch.cat((context, output), dim=2) # batch_size, output_len, hidden_size*2
        output = output.view(batch_size*output_len, -1)
        output = torch.tanh(self.linear_out(output))
        output = output.view(batch_size, output_len, -1)
        return output, attn

#Decoder module
class Decoder(nn.Module):
    '''
    decoder with embedding, attention module, GRU, a layer of linear, dropout
    '''
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_mask(self, x_len, y_len):
        # a mask of shape x_len * y_len
        # mask the future information to avoid information leakage
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        x_mask = torch.arange(max_x_len, device=x_len.device)[None, :] < x_len[:, None]
        y_mask = torch.arange(max_y_len, device=x_len.device)[None, :] < y_len[:, None]
        mask = (1 - x_mask[:, :, None] * y_mask[:, None, :]).byte()
        return mask
    
    def forward(self, ctx, ctx_lengths, y, y_lengths, hid):
        #forward pass
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]
        y_sorted = self.dropout(self.embed(y_sorted)) # batch_size, output_length, embed_size
        #get output and hidden state of rnn
        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        out, hid = self.rnn(packed_seq, hid)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()
        #use mask function to avoid information leakage
        mask = self.create_mask(y_lengths, ctx_lengths)
        #get output and atten
        output, attn = self.attention(output_seq, ctx, mask)        
        output = F.log_softmax(self.out(output), -1)
        #output is the final order of data, hid-hidden state of decoder, attn-attention matrix
        return output, hid, attn

#Seq2Seq Module    
class Seq2Seq(nn.Module):
    '''
    sequence to sequence main funtion, adduce the class of encoder, attention, decoder
    '''
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)
        output, hid, attn = self.decoder(ctx=encoder_out, 
                    ctx_lengths=x_lengths,
                    y=y,
                    y_lengths=y_lengths,
                    hid=hid)
        return output, attn
    
    def translate(self, x, x_lengths, y, max_length=100):
        encoder_out, hid = self.encoder(x, x_lengths)
        preds = []
        batch_size = x.shape[0]
        attns = []
        for i in range(max_length):
            output, hid, attn = self.decoder(ctx=encoder_out, 
                    ctx_lengths=x_lengths,
                    y=y,
                    y_lengths=torch.ones(batch_size).long().to(y.device),
                    hid=hid)
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)
            attns.append(attn)
        return torch.cat(preds, 1), torch.cat(attns, 1)

    
# masked cross entropy loss
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # input: (batch_size * seq_len) * vocab_size
        input = input.contiguous().view(-1, input.size(2))
        # target: batch_size * 1
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        output = -input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output
    
#______________________________#    
#Main#-the hyper parameter setting
dropout = 0.2
embed_size = hidden_size = 100
encoder = Encoder(vocab_size=en_total_words,
                       embed_size=embed_size,
                      enc_hidden_size=hidden_size,
                       dec_hidden_size=hidden_size,
                      dropout=dropout)
decoder = Decoder(vocab_size=cn_total_words,
                      embed_size=embed_size,
                      enc_hidden_size=hidden_size,
                       dec_hidden_size=hidden_size,
                      dropout=dropout)
model = Seq2Seq(encoder, decoder)
model = model
loss_fn = LanguageModelCriterion()
optimizer = torch.optim.Adam(model.parameters())

#______________________________________________________________________________               
#Train and Evaluate
#Train
def train(model, data, num_epochs=20):
    '''
    Parameters
    ----------
    model : model name
        The model used in seq2seq.
    data : list in tuple
        the data used to evaluate the model.
    num_epochs: int
        the number of training. Default is 20.

    Returns
    -------
    print out the training loss.
    And for every 5 epoch, print the test loss.
    '''
    for epoch in range(num_epochs):
        model.train() # the flag of training process
        total_num_words = total_loss = 0. #the total loss of batch
        #load the data and length of data
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            #tranform into tensor from numpy array
            mb_x = torch.from_numpy(mb_x).long()
            mb_x_len = torch.from_numpy(mb_x_len).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).long()
            mb_y_len = torch.from_numpy(mb_y_len-1).long()
            mb_y_len[mb_y_len<=0] = 1
            #set up the model with these data input
            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)
            #set the output
            mb_out_mask = torch.arange(mb_y_len.max().item())[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()
            #calculate the loss
            loss = loss_fn(mb_pred, mb_output, mb_out_mask)
            #the total loss and number of words in sentences
            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
            
            #update the model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            
            if it % 100 == 0:
                print("Epoch", epoch, "iteration", it, "loss", loss.item())

                
        print("Epoch", epoch, "Training loss", total_loss/total_num_words)
        if epoch % 5 == 0:
            evaluate(model, dev_data)

#______________________________#   
#Main#-Train set
train(model, train_data, num_epochs=20)

#Evaluate model
def evaluate(model, data):
    '''
    Parameters
    ----------
    model : model name
        The model used in seq2seq.
    data : list in tuple
        the data used to evaluate the model.

    Returns
    -------
    print out the evaluation loss.

    '''
    model.eval()
    total_num_words = total_loss = 0.
    with torch.no_grad():
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).long()
            mb_x_len = torch.from_numpy(mb_x_len).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).long()
            mb_y_len = torch.from_numpy(mb_y_len-1).long()
            mb_y_len[mb_y_len<=0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item())[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
    print("Evaluation loss", total_loss/total_num_words)

    
#______________________________________________________________________________               
#Translate for dev data    
def translate_dev(i):
    '''
    Parameters
    ----------
    i : int
        the line of the translation target.

    Returns
    -------
    print the result without return.

    '''
    en_sent = " ".join([inv_en_dict[w] for w in dev_en[i]])
    print(en_sent)
    cn_sent = " ".join([inv_cn_dict[w] for w in dev_cn[i]])
    print("".join(cn_sent))
    #tranform the data from array into tensor
    mb_x = torch.from_numpy(np.array(dev_en[i]).reshape(1, -1)).long()
    mb_x_len = torch.from_numpy(np.array([len(dev_en[i])])).long()
    bos = torch.Tensor([[cn_dict["BOS"]]]).long()
    #translate
    translation, attn = model.translate(mb_x, mb_x_len, bos)
    #set up the translation list
    translation = [inv_cn_dict[i] for i in translation.data.cpu().numpy().reshape(-1)]
    #collect the words in cn
    trans = []
    for word in translation:
        if word != "EOS":
            trans.append(word)
        else:
            break
    print("".join(trans))

#Main#- translate
for i in range(100,120):
    translate_dev(i)
    print()
    
    
    
    
    
    
    
    
    
    
    
    
    