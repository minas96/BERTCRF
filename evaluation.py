# -*- coding: utf-8 -*-
"""Import everything"""

import numpy as np
import torch
import torch.nn as nn
import tqdm as tq
from sklearn.metrics import classification_report

def evaluate(model, class_dict, inv_class_dict, test_batches, use_cuda, gpu_device):

# Uncomment this for running only evaluation
# if __name__ == '__main__':
#     from initialize import *
#     model, class_dict, inv_class_dict, test_batches, use_cuda, gpu_device = my_initialize()

    # If GPU available use cuda
    if use_cuda:
        model.to('cuda')

    # Initialize the CrossEntropyLoss
    cross_entropy = nn.CrossEntropyLoss()

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

    # Set model to evaluation mode for the dev set
    model.eval()

    losses = list()
    true_labels_output = list()
    pred_labels_output = list()

    for p_batch, p_batch_token_lens, label_tags in tq.tqdm(test_batches):

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
        # NOTE: We don't perform back propagation because we are in the test set
        losses.append(loss.item())

    # Get the true and pred labels of all batches into a flat list
    true_labels_output_flat = [item for sublist in true_labels_output for item in sublist]
    pred_labels_output_flat = [item for sublist in pred_labels_output for item in sublist]

    log_f = open('system_log.txt', 'a', encoding='utf-8')
    log_f.flush()
    print(f'Best Epoch: {best_epoch} Test Loss: {np.mean(losses)}')
    log_f.write(f'Best Epoch: {best_epoch} Test Loss: {np.mean(losses)}')
    log_f.flush()

    # Calculate the PRF scores using the true and pred labels for the test set
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

    print('Test Output F1-Score: {}'.format(cr_output['macro avg']['f1-score']))

    log_f.write(cr_output_text)

    log_f.write('Test Output F1-Score: {}'.format(cr_output['macro avg']['f1-score']))


    log_f.flush()


    print("Evaluation Exact \n")
    log_f.write("Evaluation Exact \n")
    counter = 0
    correct_pred_sentences = []
    not_correct_pred_sentences = []
    for i in range(len(true_labels_output)):
        not_matched = False
        for j in range(len(true_labels_output[i])):
            if(true_labels_output[i][j] != pred_labels_output[i][j]):
                not_matched = True
                break
        if not_matched:
            counter = counter + 1
            not_correct_pred_sentences.append(model.not_skipped_sentences_examples[i])
        else:
            correct_pred_sentences.append(model.not_skipped_sentences_examples[i])

    print("\t",(len(true_labels_output)-counter),"/",len(true_labels_output))
    log_f.write("\t"+str(len(true_labels_output)-counter)+"/"+str(len(true_labels_output)))
    log_f.flush()

    import random

    print("Correct insatnces: ", len(correct_pred_sentences), " / not correct instance: ", len(not_correct_pred_sentences))
    print("Some examples of correct instances")

    log_f.write("Correct insatnces: "+ str(len(correct_pred_sentences))+ " / not correct instance: "+ str(len(not_correct_pred_sentences)))
    # log_f.write("Some examples of correct instances")
    # list_of_indexes = random.sample(range(len(correct_pred_sentences)), 5)
    # for i in list_of_indexes:
    #     print(correct_pred_sentences[i])
    #     log_f.write(correct_pred_sentences[i])
    #     log_f.flush()
    #
    # print("Some examples of not correct instances")
    # log_f.write("Some examples of not correct instances")
    # list_of_indexes = random.sample(range(len(not_correct_pred_sentences)), 5)
    # for i in list_of_indexes:
    #     print(not_correct_pred_sentences[i])
    #     log_f.write(not_correct_pred_sentences[i])
    #     log_f.flush()

    for key in class_dict.keys():
        print("\nEvaluation Exact "+ key)
        log_f.write("\nEvaluation Exact "+ key)
        log_f.flush()
        total = 0
        counter = 0
        for i in range(len(true_labels_output)):
            for j in range(len(true_labels_output[i])):
                if true_labels_output[i][j] == class_dict[key]:
                    total = total + 1
                    if(true_labels_output[i][j] != pred_labels_output[i][j]):
                        counter = counter + 1
                        break
        print("\t",(total-counter),"/",total)
        log_f.write("\t"+ str(total - counter)+ "/"+ str(total))
        log_f.flush()


    print("\nEvaluation Jaccard\n")
    log_f.write("\nEvaluation Jaccard\n")
    total = []
    correct_pred_sentences = []
    not_correct_pred_sentences = []
    percentage = 0.8
    for i in range(len(true_labels_output)):
        counter = 0
        for j in range(len(true_labels_output[i])):
            if(true_labels_output[i][j] == pred_labels_output[i][j]):
                counter = counter + 1
        if counter/len(true_labels_output[i]) >= percentage:
            correct_pred_sentences.append(model.not_skipped_sentences_examples[i])
        else:
            not_correct_pred_sentences.append(model.not_skipped_sentences_examples[i])
        total.append(counter/len(true_labels_output[i]))
    print("\t",(sum(total) / len(total)))
    log_f.write(("\t"+str(sum(total) / len(total))))
    log_f.flush()

    print("(Percentage:",percentage,") Correct insatnces: ", len(correct_pred_sentences), " / not correct instance: ", len(not_correct_pred_sentences))
    # print("Some examples of correct instances")
    # log_f.write("(Percentage:"+str(percentage)+") Correct insatnces: "+ str(len(correct_pred_sentences))+ " / not correct instance: "+ str(len(not_correct_pred_sentences)))
    # log_f.write("Some examples of correct instances")
    # log_f.flush()
    # list_of_indexes = random.sample(range(len(correct_pred_sentences)), 5)
    # for i in list_of_indexes:
    #     print(correct_pred_sentences[i])
    #     log_f.write(correct_pred_sentences[i])


    # print("Some examples of not correct instances")
    # log_f.write("Some examples of not correct instances")
    # list_of_indexes = random.sample(range(len(not_correct_pred_sentences)), 5)
    # for i in list_of_indexes:
    #     print(not_correct_pred_sentences[i])
    #     log_f.write(not_correct_pred_sentences[i])
    #     log_f.flush()


    for key in class_dict.keys():
        print("\nEvaluation Jaccard ", key)
        log_f.write(("\nEvaluation Jaccard "+ key))
        total = []
        key_found = 0
        for i in range(len(true_labels_output)):
            counter = 0
            for j in range(len(true_labels_output[i])):
                if true_labels_output[i][j] == class_dict[key]:
                    key_found = key_found + 1
                    if(true_labels_output[i][j] == pred_labels_output[i][j]):
                        counter = counter + 1
            if key_found != 0:
                total.append(counter/key_found)
        print("\t",(sum(total) / len(total)))
        log_f.write("\t"+ str(sum(total) / len(total)))
        log_f.flush()


    print("sentences_to_be_preprocessed ", model.sentences_to_be_preprocessed)
    print("before_skip ", model.before_skip)
    print("total_combined_sentences ", model.total_combined_sentences)
    print("skipped_sentences ", model.skipped_sentences)
    print("Examples fo skipped sentences")
    # log_f.write("sentences_to_be_preprocessed ", model.sentences_to_be_preprocessed)
    # log_f.write("before_skip ", model.before_skip)
    # log_f.write("total_combined_sentences ", model.total_combined_sentences)
    # log_f.write("skipped_sentences ", model.skipped_sentences)
    # log_f.write("Examples fo skipped sentences")
    # log_f.flush()
    # for example in  model.skipped_sentences_examples[]:
    #     # print(example.shape[1], " ", example)
    #     print(sum(len(ex[0]) for ex in example), " ", example)
    #     log_f.write(sum(len(ex[0]) for ex in example), " ", example)
    #     log_f.flush()
    #

    log_f.close()