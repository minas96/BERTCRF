# -*- coding: utf-8 -*-
"""Greek BERT NER System EKPA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Xg_d3Snvm9NTCph0x__OjG41s2eIH_fG

# Imports, Seeds, Initialization of Modules

Setup required python modules
"""

# !pip install transformers
# !pip install sentence_splitter

"""Import everything"""

import os
import json
import torch
import random
import numpy as np
import torch.nn as nn
import tqdm as tq
import torch.nn.functional as F
from torch.backends import cudnn
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import classification_report

from typing import List, Optional

from termcolor import colored, cprint

from nltk import ngrams
from sentence_splitter import split_text_into_sentences
from BERT_CRF import *
"""Seeds for reproducibility and Check for GPU"""

my_seed = 1
random.seed(my_seed)
np.random.seed(my_seed)
torch.manual_seed(my_seed)
cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

gpu_device = 0
use_cuda = torch.cuda.is_available()
if (use_cuda):
    torch.cuda.manual_seed(my_seed)
    print("Using GPU")
else:
    print("NOT Using GPU")

"""# Main code

## Setup

Initialize global parameters
"""

# Batch size
batch_size = 2
# Hidden size of MLP
hidden_size = 200
# Learning rate
lr = 1e-3
# Max epochs of system
epochs = 100
# Patience of system
max_patience = 10
# Window of context
window = 2

# The classes that we can predict
# NOTE: The 'X' class is used as a class label of all the subwords of a word entity (other than 'O') proceeding the first word
class_dict = {
    'O': 0,
    'B-Cue': 1,
    'I-Cue': 2,
    'B-Content': 3,
    'I-Content': 4,
    'B-Source': 5,
    'I-Source': 6,
    'X': 11
}

# Get the inverted class dictionary
inv_class_dict = {class_dict[k]: k for k in class_dict}

"""Load data and Split to train, dev and test sets"""

with open('train_data.json', encoding='utf-8') as fin:
    train_instances = json.load(fin)

with open('dev_data.json', encoding='utf-8') as fin:
    dev_instances = json.load(fin)

with open('test_data.json', encoding='utf-8') as fin:
    test_instances = json.load(fin)

# # Find the sizes of the train, dev and test splits using 70% - 20% - 10%
# train_len = int(len(instances) * 0.7)
# dev_len = int(len(instances) * 0.2)
# test_len = int(len(instances) * 0.1)
#
# # Add any trailing instances to the train (due to the integer casting)
# if len(instances) > train_len + dev_len + test_len:
#     train_len += len(instances) - (train_len + dev_len + test_len)
#
# # Randomly sample the train, dev and test splits from the data
# train_instances = random.sample(instances, train_len)
# rest_instances = [e for e in instances if e not in train_instances]
# dev_instances = random.sample(rest_instances, dev_len)
# test_instances = [e for e in rest_instances if e not in dev_instances]

"""Initialize the System and Optimizer"""

model = BERT_CRF(b_size=batch_size,
                 n_classes_output=len(class_dict),
                 hidden_size=hidden_size,
                 window=window)

# If GPU available use cuda
if use_cuda:
    model.to('cuda')

# Initialize an Adam optimizer with the learning rate from the global parameters
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Initialize the CrossEntropyLoss
cross_entropy = nn.CrossEntropyLoss()

"""Preprocess all batches"""

# Preprocess all train batches
print('Preprocessing train batches')
train_batches = [train_instances[i * batch_size: (i + 1) * batch_size] for i in
                 tq.trange((len(train_instances) // batch_size) + 1)]
if not train_batches[-1]:
    train_batches = train_batches[:-1]

train_batches = [model.preprocess_batch(batch) for batch in tq.tqdm(train_batches)]

# Preprocess all dev batches
print('Preprocessing dev batches')
dev_batches = [dev_instances[i * batch_size: (i + 1) * batch_size] for i in
               tq.trange((len(dev_instances) // batch_size) + 1)]
if not dev_batches[-1]:
    dev_batches = dev_batches[:-1]

dev_batches = [model.preprocess_batch(batch) for batch in tq.tqdm(dev_batches)]

# Preprocess all test batches
print('Preprocessing test batches')
test_batches = [test_instances[i * batch_size: (i + 1) * batch_size] for i in
                tq.trange((len(test_instances) // batch_size) + 1)]
if not test_batches[-1]:
    test_batches = test_batches[:-1]

test_batches = [model.preprocess_batch(batch) for batch in tq.tqdm(test_batches)]

"""## Training

Global variables for training
"""

patience = max_patience
best_f1 = 0.0
best_f1_output = 0.0
best_epoch = 0
best_cr = None

"""Train the model"""

# Open a file for logging
log_f = open('system_log.txt', 'w', encoding='utf-8')
log_f.flush()

# Train for the max epochs
for epoch in range(epochs):
    print(f'\n{"=" * 30} EPOCH {epoch} {"=" * 30}\n')
    log_f.write(f'\n{"=" * 30} EPOCH {epoch} {"=" * 30}\n\n')
    log_f.flush()

    # Set model to train mode for the train set
    model.train()

    # Shuffle the training batches
    random.shuffle(train_batches)

    losses = list()
    true_labels_output = list()
    pred_labels_output = list()
    for p_batch, p_batch_token_lens, label_tags in tq.tqdm(train_batches):
        if len(p_batch) == 0:
            continue

        # Find the batch sentence lengths
        p_batch_sents = [[sum(s) for s in b] for b in p_batch_token_lens]

        # Reset the gradients
        optimizer.zero_grad()

        # Pass the batch though the system
        model_out = model(p_batch, p_batch_sents)

        # Calculate the predictions of the model using argmax and add them to a list
        pred_labels_output.extend([torch.max(e, dim=2)[1].cpu().numpy().tolist()[0] for e in model_out])

        # Get only the label_tags of the middle sentence
        # label_tags_middle = [l[s[0]:s[0]+s[1]] for l, s in zip(label_tags, p_batch_sents)]
        label_tags_middle = []# = label_tags # maybe they should be list of 6 items (now it is 2 of 3)
        for item in label_tags:
            for label_tags_item in item:
                label_tags_middle.append(label_tags_item)

        # Add the label tags of the middle sentence to a list
        true_labels_output.extend([[class_dict[e2] for e2 in e] for e in label_tags_middle])
        #true_labels_output.extend(label_tags_middle)

        # Add the label tags of the middle sentence to a Tensor object (use GPU if available)
        labels_output = [torch.LongTensor([class_dict[e2] for e2 in e]) for e in label_tags_middle]
        if use_cuda:
            labels_output = [e.cuda(gpu_device) for e in labels_output]

        # Calculate the loss of each instance of the batch using CrossEntropyLoss
        loss_output = [cross_entropy(e[0][0][e[1] != 11], e[1][e[1] != 11]) for e in zip(model_out, labels_output)]

        # Calculate the loss of the batch (mean of the losses of each instance)
        loss = torch.mean(torch.stack(loss_output))

        # Add the loss value to a list
        losses.append(loss.item())

        # Perform back propagation to calculate the gradients
        loss.backward()

        # Update the parameters of the model
        optimizer.step()
    # Get the true and pred labels of all batches into a flat list
    true_labels_output_flat = [item for sublist in true_labels_output for item in sublist]
    pred_labels_output_flat = [item for sublist in pred_labels_output for item in sublist]

    print(f'Epoch {epoch} Train Loss: {np.mean(losses)}')
    log_f.write(f'Epoch {epoch} Train Loss: {np.mean(losses)}\n')
    log_f.flush()

    # Calculate the PRF scores using the true and pred labels for the epoch
    # NOTE: We don't take into account the 'X' labels for the scores
    cr_output = classification_report(y_true=true_labels_output_flat,
                                      y_pred=pred_labels_output_flat,
                                      labels=[e for e in inv_class_dict.keys() if e != 11],
                                      target_names=[e for e in class_dict.keys() if e != 'X'],
                                      output_dict=True)

    print(classification_report(y_true=true_labels_output_flat,
                                y_pred=pred_labels_output_flat,
                                labels=[e for e in inv_class_dict.keys() if e != 11],
                                target_names=[e for e in class_dict.keys() if e != 'X'],
                                output_dict=False))

    log_f.write(classification_report(y_true=true_labels_output_flat,
                                      y_pred=pred_labels_output_flat,
                                      labels=[e for e in inv_class_dict.keys() if e != 11],
                                      target_names=[e for e in class_dict.keys() if e != 'X'],
                                      output_dict=False) + '\n')
    log_f.flush()
    print('Train Output F1-Score: {}'.format(cr_output['macro avg']['f1-score']))
    log_f.write('Train Output F1-Score: {}\n'.format(cr_output['macro avg']['f1-score']))
    log_f.flush()

    # Set model to evaluation mode for the dev set
    model.eval()

    losses = list()
    true_labels_output = list()
    pred_labels_output = list()

    for p_batch, p_batch_token_lens, label_tags in tq.tqdm(dev_batches):

        # TODO
        if len(p_batch) == 0:
            continue

        # Find the batch sentence lengths
        p_batch_sents = [[sum(s) for s in b] for b in p_batch_token_lens]

        # Pass the batch though the system
        model_out = model(p_batch, p_batch_sents)

        # Calculate the predictions of the model using argmax and add them to a list
        pred_labels_output.extend([torch.max(e, dim=2)[1].cpu().numpy().tolist()[0] for e in model_out])

        # Get only the label_tags of the middle sentence
        #label_tags_middle = [l[s[0]:s[0] + s[1]] for l, s in zip(label_tags, p_batch_sents)]
        label_tags_middle = []  # = label_tags # maybe they should be list of 6 items (now it is 2 of 3)
        for item in label_tags:
            for label_tags_item in item:
                label_tags_middle.append(label_tags_item)

        # Add the label tags of the middle sentence to a list
        true_labels_output.extend([[class_dict[e2] for e2 in e] for e in label_tags_middle])

        # Add the label tags of the middle sentence to a Tensor object (use GPU if available)
        labels_output = [torch.LongTensor([class_dict[e2] for e2 in e]) for e in label_tags_middle]
        if use_cuda:
            labels_output = [e.cuda(gpu_device) for e in labels_output]

        # Calculate the loss of each instance of the batch using CrossEntropyLoss
        loss_output = [cross_entropy(e[0][0][e[1] != 11], e[1][e[1] != 11]) for e in zip(model_out, labels_output)]

        # Calculate the loss of the batch (mean of the losses of each instance)
        loss = torch.mean(torch.stack(loss_output))

        # Add the loss value to a list
        # NOTE: We don't perform back propagation because we are in the development set
        losses.append(loss.item())

    # Get the true and pred labels of all batches into a flat list
    true_labels_output_flat = [item for sublist in true_labels_output for item in sublist]
    pred_labels_output_flat = [item for sublist in pred_labels_output for item in sublist]

    print(f'Epoch {epoch} Dev Loss: {np.mean(losses)}')
    log_f.write(f'Epoch {epoch} Dev Loss: {np.mean(losses)}\n')
    log_f.flush()

    # Calculate the PRF scores using the true and pred labels for the epoch
    # NOTE: We don't take into account the 'X' labels for the scores
    cr_output = classification_report(y_true=true_labels_output_flat,
                                      y_pred=pred_labels_output_flat,
                                      labels=[e for e in inv_class_dict.keys() if e != 11],
                                      target_names=[e for e in class_dict.keys() if e != 'X'],
                                      output_dict=True)

    cr_output_text = classification_report(y_true=true_labels_output_flat,
                                           y_pred=pred_labels_output_flat,
                                           labels=[e for e in inv_class_dict.keys() if e != 11],
                                           target_names=[e for e in class_dict.keys() if e != 'X'],
                                           output_dict=False)

    print(cr_output_text)
    log_f.write(cr_output_text + '\n')
    print('Dev Output F1-Score: {}'.format(cr_output['macro avg']['f1-score']))
    log_f.write('Dev Output F1-Score: {}\n\n'.format(cr_output['macro avg']['f1-score']))
    log_f.flush()

    print()

    # Check if the macro avg f1-score has improved
    if cr_output['macro avg']['f1-score'] > best_f1:
        # Assign the new best macro avg f1-score, classification reports, epoch and reset patience
        best_f1 = cr_output['macro avg']['f1-score']
        best_f1_output = cr_output['macro avg']['f1-score']
        best_cr = [cr_output, cr_output_text]
        best_epoch = epoch
        patience = max_patience

        # Save the parameters of the best state of the system
        state = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            best_f1=best_f1,
            best_epoch=best_epoch,
            best_cr=best_cr
        )
        torch.save(state, 'system_best_epoch.pth.tar')
        print('Model saved')
        print()
        log_f.write('Model saved\n\n')
        log_f.flush()

    else:
        # Decrease the patience
        patience -= 1
        # If the patience variable goes to 0, then we stop the training of the system
        if patience == 0:
            break
    print(f'Best epoch: {best_epoch}')
    print(f'Patience: {patience}')
    print()
    log_f.write(f'Best epoch: {best_epoch}\n')
    log_f.write(f'Patience: {patience}\n\n')
    log_f.flush()

print(f'\n{"=" * 30} FINAL RESULTS {"=" * 30}\n')
log_f.write(f'\n{"=" * 30} FINAL RESULTS {"=" * 30}\n\n')

print('Best Epoch: {}\nOutput F1 Score: {}'.format(best_epoch, best_f1_output))
print(best_cr[1])

log_f.write('Best Epoch: {}\nOutput F1 Score: {}\n'.format(best_epoch, best_f1_output))
log_f.write(best_cr[1] + '\n')
log_f.flush()
# Close log file
log_f.close()

"""## Testing

Load the best state of the system
"""

# Load the best state from the file that it is stored
state = torch.load('system_best_epoch.pth.tar')

# Load the parameters into the model
model.load_state_dict(state['model'])

# Load the best epoch, macro avg f1-score and classification report
best_epoch = state['best_epoch']
best_f1 = state['best_f1']
best_cr = state['best_cr']
cr_output = best_cr[0]

"""Test the model"""
from evaluation import *
evaluate(model, class_dict, inv_class_dict, test_batches, use_cuda, gpu_device)