# %% [markdown]
# Import Libraries

# %%
"""
This script applies to IOB2 or IOBES tagging scheme.
If you are using a different scheme, please convert to IOB2 or IOBES.
IOB2:
- B = begin, 
- I = inside but not the first, 
- O = outside
e.g. 
John   lives in New   York  City  .
B-PER  O     O  B-LOC I-LOC I-LOC O
IOBES:
- B = begin, 
- E = end, 
- S = singleton, 
- I = inside but not the first or the last, 
- O = outside
e.g.
John   lives in New   York  City  .
S-PER  O     O  B-LOC I-LOC E-LOC O
prefix: IOBES
chunk_type: PER, LOC, etc.
"""
from __future__ import division, print_function, unicode_literals

import sys
from collections import defaultdict

def split_tag(chunk_tag):
    """
    split chunk tag into IOBES prefix and chunk_type
    e.g. 
    B-PER -> (B, PER)
    O -> (O, None)
    """
    if chunk_tag == 'O':
        return ('O', None)
    return chunk_tag.split('-', maxsplit=1)

def is_chunk_end(prev_tag, tag):
    """
    check if the previous chunk ended between the previous and current word
    e.g. 
    (B-PER, I-PER) -> False
    (B-LOC, O)  -> True
    Note: in case of contradicting tags, e.g. (B-PER, I-LOC)
    this is considered as (B-PER, B-LOC)
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == 'O':
        return False
    if prefix2 == 'O':
        return prefix1 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']

def is_chunk_start(prev_tag, tag):
    """
    check if a new chunk started between the previous and current word
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == 'O':
        return False
    if prefix1 == 'O':
        return prefix2 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']


def calc_metrics(tp, p, t, percent=True):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = tp / p if p else 0
    recall = tp / t if t else 0
    fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * fb1
    else:
        return precision, recall, fb1


def count_chunks(true_seqs, pred_seqs):
    """
    true_seqs: a list of true tags
    pred_seqs: a list of predicted tags
    return: 
    correct_chunks: a dict (counter), 
                    key = chunk types, 
                    value = number of correctly identified chunks per type
    true_chunks:    a dict, number of true chunks per type
    pred_chunks:    a dict, number of identified chunks per type
    correct_counts, true_counts, pred_counts: similar to above, but for tags
    """
    correct_chunks = defaultdict(int)
    true_chunks = defaultdict(int)
    pred_chunks = defaultdict(int)

    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    prev_true_tag, prev_pred_tag = 'O', 'O'
    correct_chunk = None

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        if true_tag == pred_tag:
            correct_counts[true_tag] += 1
        true_counts[true_tag] += 1
        pred_counts[pred_tag] += 1

        _, true_type = split_tag(true_tag)
        _, pred_type = split_tag(pred_tag)

        if correct_chunk is not None:
            true_end = is_chunk_end(prev_true_tag, true_tag)
            pred_end = is_chunk_end(prev_pred_tag, pred_tag)

            if pred_end and true_end:
                correct_chunks[correct_chunk] += 1
                correct_chunk = None
            elif pred_end != true_end or true_type != pred_type:
                correct_chunk = None

        true_start = is_chunk_start(prev_true_tag, true_tag)
        pred_start = is_chunk_start(prev_pred_tag, pred_tag)

        if true_start and pred_start and true_type == pred_type:
            correct_chunk = true_type
        if true_start:
            true_chunks[true_type] += 1
        if pred_start:
            pred_chunks[pred_type] += 1

        prev_true_tag, prev_pred_tag = true_tag, pred_tag
    if correct_chunk is not None:
        correct_chunks[correct_chunk] += 1

    return (correct_chunks, true_chunks, pred_chunks, 
        correct_counts, true_counts, pred_counts)

def get_result(correct_chunks, true_chunks, pred_chunks,
    correct_counts, true_counts, pred_counts, verbose=True):
    """
    if verbose, print overall performance, as well as preformance per chunk type;
    otherwise, simply return overall prec, rec, f1 scores
    """
    # sum counts
    sum_correct_chunks = sum(correct_chunks.values())
    sum_true_chunks = sum(true_chunks.values())
    sum_pred_chunks = sum(pred_chunks.values())

    sum_correct_counts = sum(correct_counts.values())
    sum_true_counts = sum(true_counts.values())

    nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != 'O')
    nonO_true_counts = sum(v for k, v in true_counts.items() if k != 'O')

    chunk_types = sorted(list(set(list(true_chunks) + list(pred_chunks))))

    # compute overall precision, recall and FB1 (default values are 0.0)
    prec, rec, f1 = calc_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks)
    res = (prec, rec, f1)
    if not verbose:
        return res

    # print overall performance, and performance per chunk type
    
    print("processed %i tokens with %i phrases; " % (sum_true_counts, sum_true_chunks), end='')
    print("found: %i phrases; correct: %i.\n" % (sum_pred_chunks, sum_correct_chunks), end='')
        
    print("accuracy: %6.2f%%; (non-O)" % (100*nonO_correct_counts/nonO_true_counts))
    print("accuracy: %6.2f%%; " % (100*sum_correct_counts/sum_true_counts), end='')
    print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" % (prec, rec, f1))

    # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
    for t in chunk_types:
        prec, rec, f1 = calc_metrics(correct_chunks[t], pred_chunks[t], true_chunks[t])
        print("%17s: " %t , end='')
        print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
                    (prec, rec, f1), end='')
        print("  %d" % pred_chunks[t])

    return res
    # you can generate LaTeX output for tables like in
    # http://cnts.uia.ac.be/conll2003/ner/example.tex
    # but I'm not implementing this

def evaluate(true_seqs, pred_seqs, verbose=True):
    (correct_chunks, true_chunks, pred_chunks,
        correct_counts, true_counts, pred_counts) = count_chunks(true_seqs, pred_seqs)
    result = get_result(correct_chunks, true_chunks, pred_chunks,
        correct_counts, true_counts, pred_counts, verbose=verbose)
    return result

def evaluate_conll_file(fileIterator):
    true_seqs, pred_seqs = [], []
    
    for line in fileIterator:
        cols = line.strip().split()
        # each non-empty line must contain >= 3 columns
        if not cols:
            true_seqs.append('O')
            pred_seqs.append('O')
        elif len(cols) < 3:
            raise IOError("conlleval: too few columns in line %s\n" % line)
        else:
            # extract tags from last 2 columns
            true_seqs.append(cols[-2])
            pred_seqs.append(cols[-1])
    return evaluate(true_seqs, pred_seqs)

# %%
!pip install matplotlib
!pip freeze > requirements.txt
!pip install torchsummary
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
import matplotlib.pyplot as plt
from torchsummary import summary

# %% [markdown]
# Download Dataset

# %%
with open('//Users/mindmumi/Desktop/train.txt', 'r') as f: 
    data_train = f.read().rstrip()

# %%
with open('/Users/mindmumi/Desktop/dev.txt', 'r') as f:
    data_val = f.read().rstrip()

# %%
with open('/Users/mindmumi/Desktop/test-submit.txt', 'r') as f:
    data_test = f.read().rstrip()

# %% [markdown]
# Data Preprocressing

# %%
# to create dataset
def list_data(data):
    sentences = data.split("\n\n")

    words = []
    for i in range(len(sentences)):
        w = sentences[i].split("\n")
        words.append(w)

    # list all words
    ww = []    
    tt = []
    for i in range(len(words)):
        for j in range(len(words[i])):
            w11, t11 = words[i][j].split("\t")
            ww.append(w11)
            tt.append(t11)

    # list all word in sentence
    w = []    
    t = []
    for i in range(len(words)):
        w_ = []
        t_ = []
        for j in range(len(words[i])):
            w1, t1 = words[i][j].split("\t")
            w_.append(w1)
            t_.append(t1)
        w.append(w_)
        t.append(t_)
       
    return ww, tt, w, t

    # ww [0] = list all words
    # tt [1] = list all tags
    # w [2] = list words in sentences
    # t [3] = list tags in sentences

# %%
# to make the letter lowercase
def lowercase(sen):
    sen_lower = []
    for i in range(len(sen)):
        s_ = []
        for j in range(len(sen[i])):
            s1 = sen[i][j].lower()
            s_.append(s1)
        sen_lower.append(s_)
    return sen_lower

# %% [markdown]
# --- Train Dataset

# %%
# create train dataset
ww_train = list_data(data_train)[0] 
tt_train = list_data(data_train)[1]
words_train = list_data(data_train)[2]
tags_train = list_data(data_train)[3]
train_sentences = lowercase(words_train)

# %%
# check the word sentence numbers are equal to tag sentence numbers
len(train_sentences) == len(tags_train)

# %%
# check if there's any NaN value
len(ww_train) == len(tt_train)

# %%
# set the unique words
vocab_uni = set(w for s in train_sentences for w in s)

# %%
# add the unknown <unk> and <pad> token to our vocabulary
vocab_uni.add("<unk>")
vocab_uni.add("<pad>")

# %%
# map each word into its id representation and vice versa
ix2word = sorted(list(vocab_uni))
word2ix = {word: ix for ix, word in enumerate(ix2word)}

# %%
print("<pad> ID : ", (word2ix["<pad>"]))

# %%
# set the unique tags
tag_uni = set(tt_train)
len(tag_uni)

# %%
# add the <pad> token to tag 
tag_uni.add("<pad>")

# %%
# map each tag into its id representation and vice versa
tag2ix = {k: v for v, k in enumerate(sorted(tag_uni))}
ix2tag = {v: k for v, k in enumerate(sorted(tag_uni))}
tag2ix

# %%
print("tag of index 0 : ", (ix2tag.get(0)))

# %% [markdown]
# --- Validation Dataset

# %%
# create validation dataset
ww_val = list_data(data_val)[0] 
tt_val = list_data(data_val)[1]
words_val = list_data(data_val)[2]
tags_val = list_data(data_val)[3]
val_sentences = lowercase(words_val)

# %%
# check the word sentence numbers are equal to tag sentence numbers
len(val_sentences) == len(tags_val)

# %%
# check if there's any NaN value
len(ww_val) == len(tt_val)

# %% [markdown]
# --- Test Dataset

# %%
def list_data_(data):

    sentences = data.split("\n\n")

    words = []
    for i in range(len(sentences)):
        w = sentences[i].split("\n")
        words.append(w)

    return words

# %%
# create test dataset
words_test = list_data_(data_test)
len(words_test)

# %%
test_sentences = lowercase(words_test)
len(test_sentences)

# %% [markdown]
# - Batching

# %% [markdown]
#   -- Batch Training Dataset

# %%
# find out how many words the longest sentence contain 
batch_max_len = max([len(s) for s in train_sentences])
batch_max_len

# %%
pad_token = "<pad>"

# add <pad> to words
def batch_word(sen, max_len):
  batch_word = []
  for i in range(len(sen)):
    b = sen[i] + ([pad_token]*(max_len- len(sen[i])))
    batch_word.append(b)
  return batch_word

# add <pad> to tags
def batch_tag(tag, max_len):
  batch_tag = []
  for i in range(len(tag)):
    b = tag[i] + ([pad_token]*(max_len - len(tag[i])))
    batch_tag.append(b)
  return batch_tag

# %%
# batch for words/tags train dataset
b_w_train = batch_word(train_sentences, batch_max_len)
b_t_train = batch_tag(tags_train, batch_max_len)

# %%
# covert word to ix
def convert_word2ix(sentence, word2ix):
  return [word2ix.get(word, word2ix["<unk>"]) for word in sentence]

# covert tag to ix
def convert_tag2ix(tags, tag2ix):
  return [tag2ix.get(tag) for tag in tags]

# %%
# batch ix for words/tags train dataset
b_word2ix_train = [convert_word2ix(s, word2ix) for s in b_w_train]
b_tag2ix_train = [convert_tag2ix(s, tag2ix) for s in b_t_train]

# %%
# change to tensor
b_word2ix_train = torch.tensor(b_word2ix_train)
b_tag2ix_train = torch.tensor(b_tag2ix_train)

# %%
# put into dataLoader generator
train_data = list(zip(b_word2ix_train, b_tag2ix_train))
batch_size = 6
shuffle = True

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)

# %% [markdown]
# -- Batch Validation Dataset

# %%
# find the longest sentence contain how many words
batch_max_len_ = max([len(s) for s in val_sentences])
batch_max_len_

# %%
# batch for words/tags train dataset
b_w_val = batch_word(val_sentences, batch_max_len_)
b_t_val = batch_tag(tags_val, batch_max_len_)

# %%
# batch ix for words/tags train dataset
b_word2ix_val = [convert_word2ix(s, word2ix) for s in b_w_val]
b_tag2ix_val = [convert_tag2ix(s, tag2ix) for s in b_t_val]

# %%
# change to tensor
b_word2ix_val = torch.tensor(b_word2ix_val)
b_tag2ix_val = torch.tensor(b_tag2ix_val)

# %%
# put into dataLoader generator
val_data = list(zip(b_word2ix_val, b_tag2ix_val))
batch_size = len(val_data)
shuffle = False

val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle)

# %% [markdown]
# -- Batching Test Dataset

# %%
# find the longest sentence contain how many words
batch_max_len_t = max([len(s) for s in test_sentences])
batch_max_len_t

# %%
# batch for words/tags train dataset
b_w_test = batch_word(test_sentences, batch_max_len_t)

# %%
# batch ix for words/tags train dataset
b_word2ix_test = [convert_word2ix(s, word2ix) for s in b_w_test]

# %%
# change to tensor
b_word2ix_test = torch.tensor(b_word2ix_test)

# %%
# put into dataLoader generator
batch_size = len(b_word2ix_test)
shuffle = False

test_dataloader = DataLoader(b_word2ix_test, batch_size=batch_size, shuffle=shuffle)

# %% [markdown]
# Pre - Trained Embeddings

# %%
# load the whole embedding into memory
embeddings_index = dict()

with open('/Users/mindmumi/Desktop/glove.840B.300d.txt', encoding='utf-8') as f:
  for line in f:
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

print('Loaded %s word vectors.' % len(embeddings_index))

# %%
# add <pad> and <unk> into the embedding
embeddings_index['<pad>'] = np.random.rand(300).tolist()
embeddings_index['<umk>'] = np.random.rand(300).tolist()

# %%
# create a weight matrix for words in glove
token = {word: ix for ix in enumerate(embeddings_index.keys())}

embeddings = torch.tensor(list(embeddings_index.values()))

# %%
embeddings.size()

# %% [markdown]
# Model

# %%
class BiLSTMTagger(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_layers, bidirectional, num_classes):
        super(BiLSTMTagger, self).__init__()        
        self.embedding = nn.Embedding.from_pretrained(embeddings)  
        self.embedding.requires_grad=False
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers      
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, bidirectional=True, batch_first=True)       
        self.fc = nn.Linear(hidden_size*2 if bidirectional else hidden_size, num_classes)

    def forward(self, word):

        # S = Sentence Length
        # B = Batch Size
        # E = Embedding Dimension 300
        # C = Classes Number
        # H = Hidden Size
        
        # word (B,S) (8,39)
        embedded = self.embedding(word) 
        # embedd (B,S) (8,39,300)
        out, (h, c) = self.lstm(embedded) 
        # out (B,S,H) (8,39,256)
        out = self.fc(out)
        out[:,:,0] = float("-inf")
        # out (B,S,C) (8,39,22)

        return out
        

# %% [markdown]
# Training

# %%
EMBED_DIM = 300
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BIDIRECTIONAL = True
NUM_CLASSES = len(tag2ix) # 22


model = BiLSTMTagger(EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, BIDIRECTIONAL, NUM_CLASSES)

# %%
model

# %%
# loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# train the model
def train(model, num_layers, hidden_size, num_epochs, optimizer, dataloader):

    epoch_loss = 0
    num = 0
    model.train()

    for batch_ix, (word, act_tag) in enumerate(dataloader):
      #clear the gradients
      optimizer.zero_grad()
      
      # forward
      pred_tag = model(word)
      pred_tag = torch.transpose(pred_tag, 1, 2)

      # compute the batch loss
      loss = criterion(pred_tag, act_tag)

      # choose the maximum index
      pred_tag_ = pred_tag.argmax(dim=1)

      # backward (calculate the gradients)  
      loss.backward()

      # gradient descent or Adam step (update the parameters)
      optimizer.step()

      epoch_loss += loss.item()

      num+=1
      epoch_loss_ = epoch_loss/num
      
    return epoch_loss, epoch_loss_, pred_tag_, act_tag

# %%
num_epochs = 24

epoch_ = []
loss_ = []

for epoch in range(num_epochs):
    
    train_loss, epoch_loss_, _, _ = train(model, NUM_LAYERS, HIDDEN_SIZE, num_epochs, optimizer, train_dataloader)

    print(f'Epoch:', (epoch+1),'Train_loss_iter:', (epoch_loss_), 'Train Loss:', (train_loss))

    epoch_.append(epoch)
    loss_.append(epoch_loss_)

# %%
plt.plot(epoch_, loss_)
plt.title('Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# %% [markdown]
# Validation

# %%
# evaluation function
def eval(model, num_layers, hidden_size, num_epochs, dataloader):

    for batch_ix, (word, act_tag) in enumerate(dataloader):
      # forward
      pred_tag = model(word)
      pred_tag = torch.transpose(pred_tag, 1, 2)

      # choose the maximum index
      pred_tag_ = pred_tag.argmax(dim=1)

    return pred_tag_, act_tag

# %%
# remove all <pad> function
def unpad(tag, sen_original):
  result_nopad = []
  for i in range(len(sen_original)):
    result = tag[i][:len(sen_original[i])]
    if 0 in result.tolist():
      print(result)
      print(tag[i])
      exit()
    result_nopad.append(result)
  return result_nopad

# %%
# convert index to tag function
def convert_ix2tag(ixs, ix2tag):
  return [ix2tag.get(i) for ix in ixs for i in ix.tolist()]

# %%
# evaluation
pred_tag_, act_tag = eval(model, NUM_LAYERS, HIDDEN_SIZE, num_epochs, val_dataloader)

# remove all <pad>
pred_tag_nopad = unpad(pred_tag_, tags_val)
act_tag_nopad = unpad(act_tag, tags_val)

# covert index to tag
pred_tag_nopad_label = convert_ix2tag(pred_tag_nopad, ix2tag)
act_tag_nopad_label = convert_ix2tag(act_tag_nopad, ix2tag)

# calculate score metrics
prec = evaluate(act_tag_nopad_label, pred_tag_nopad_label, verbose=True)

# %% [markdown]
# Testing

# %%
# convert index to tag function
def convert_ix2tag_(ixs, ix2tag):
  return [list(map(lambda i:ix2tag[i] ,ix.tolist())) for ix in ixs]

# %%
def test_predict(model, num_layers, hidden_size, dataloader):

    for batch_ix, (word) in enumerate(dataloader):
      # forward
      pred_tag = model(word)
      pred_tag = torch.transpose(pred_tag, 1, 2)

      # choose the maximum index
      pred_tag_ = pred_tag.argmax(dim=1)

    return word, pred_tag_

# %%
word_test, pred_tag_test = test_predict(model, NUM_LAYERS, HIDDEN_SIZE, test_dataloader)

# %%
# remove all <pad>
pred_tag_nopad_test = unpad(pred_tag_test, words_test)

# covert index to tag
pred_tag_nopad_label_test = convert_ix2tag_(pred_tag_nopad_test, ix2tag)

# %%
# to check the word namber in one sentence
len(pred_tag_nopad_test[0]) == len(pred_tag_nopad_label_test[0])

# %%
# to check the number of the sentences
len(test_sentences) == len(pred_tag_nopad_label_test)

# %%
# to check the number of the words 
for i in range(len(test_sentences)):
    for j in range(len(test_sentences[i])):
        if len(test_sentences[i]) == len(pred_tag_nopad_label_test[i]):
            pass
        else:
            print('not ok')

# %%
with open("./test-submit.txt", "w", encoding="utf-8") as f:
    for xi,yi in zip(test_sentences,pred_tag_nopad_label_test):
        for word,tag in zip(xi,yi):
            f.write(f"{word}\t{tag}\n")
        f.write("\n")

# %%



