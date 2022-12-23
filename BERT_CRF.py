import torch
import random
from torch.backends import cudnn
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

def initialize():
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
        print("BERT_CRF - Using GPU")
    else:
        print("BERT_CRF - NOT Using GPU")
    return use_cuda

class BERT_CRF(nn.Module):#TODO BERT_CRF model να το ονομασω + να βαλω window property
    def __init__(self, b_size, n_classes_output, hidden_size, window):
        super(BERT_CRF, self).__init__()
        self.use_cuda = initialize()

        self.b_size = b_size
        self.n_classes_output = n_classes_output
        self.hidden_size = hidden_size
        self.window = window

        self.sentences_to_be_preprocessed = 0
        self.before_skip = 0
        self.total_combined_sentences = 0
        self.skipped_sentences = 0

        self.skipped_sentences_examples = []
        self.not_skipped_sentences_examples = []

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = AutoModel.from_pretrained("bert-base-uncased")

        # Freeze the parameters
        # Comment this for finetuning
        for param in self.bert.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(768, self.hidden_size, bias=True)
        self.linear_out_output = nn.Linear(self.hidden_size, self.n_classes_output, bias=True)

    def preprocess_batch(self, batch):
        # Preprocess the instances for a batch
        # NOTE: id 101 is [CLS], 102 is [SEP], [103] is MASK
        preprocessed_batch = list()
        for inst in batch:
            self.sentences_to_be_preprocessed = self.sentences_to_be_preprocessed + len(inst)
            #inst is the article that has sentences
            # Tokenize each word inside of every sentence
            inst_tokens = [[self.tokenizer(w, add_special_tokens=False, return_tensors='pt')['input_ids'] for w in s[0]]
                           for s in inst]
            # Calculate the lengths of subtokens for each token (we will use it later to connect the subtokens)
            inst_token_lens = [[w.shape[1] for w in s] for s in inst_tokens]
            # Calculate the length of each sentence in subtokens
            inst_sent_lens = [sum(s) for s in inst_token_lens]
            # If the length of the combined sentence is > 510 then discard the instance
            # (510 instead of 512 because we will add the [CLS] and [SEP] tokens)
            # if sum(inst_sent_lens) > 510:
            #     print('Instance subtokens > 512 , skipping...')
            #     continue
            # Get the tags of the instance
            inst_tags = list()
            for s_i, s in enumerate(inst_token_lens):
                inst_tags.append(list())
                for tok_i, tok_len in enumerate(s):
                    # Get the tag of the token and create a list of subtoken tags
                    # The first tag gets the tag of the token and the rest get the 'X' token except if the tag if
                    # 'O', which don't get a 'X' tag at all
                    tok_tag = inst[s_i][1][tok_i]
                    # tags = []
                    if tok_tag == 'O':
                        tags = ['O'] * tok_len
                    # else:
                    #     tag = tok_tag.replace("B", "I")
                    #     tags.extend([tag] * tok_len)
                    #     # tags = tag * tok_len
                    #     try:
                    #         if len(tags)>0:
                    #             tags[0] = tok_tag
                    #
                    #     except:
                    #         print("as")
                    else:
                        tags = ['X'] * tok_len
                        try:
                            if len(tags)>0:
                                tags[0] = tok_tag
                        except:
                            print("as")
                    inst_tags[-1].extend(tags)

            new_inst_tokens = []
            for i in range(len(inst_tokens)):
                if len(inst_tokens[i]) == 0:
                    continue
                temp_inst_senteces = []
                temp_inst_tokens = torch.cat([torch.LongTensor([[101]])], dim=1)
                for j in range(i - self.window, i + self.window + 1):
                    if j < 0:
                        continue
                    elif j > len(inst_tokens) - 1:
                        break
                    if len(inst_tokens[j]) != 0:
                        temp_inst_tokens = torch.cat([temp_inst_tokens] + [torch.cat(inst_tokens[j], dim=1)], dim=1)
                        temp_inst_senteces.append(inst[j])
                temp_inst_tokens = torch.cat([temp_inst_tokens] + [torch.LongTensor([[102]])], dim=1)
                # TODO
                self.before_skip = self.before_skip + 1
                if temp_inst_tokens.shape[1] > 512:
                    print('Instance subtokens > 512 , skipping...new_inst_tokens')
                    self.skipped_sentences = self.skipped_sentences + 1
                    # self.skipped_sentences_examples.append(temp_inst_tokens)
                    self.skipped_sentences_examples.append(temp_inst_senteces)
                    continue
                self.total_combined_sentences = self.total_combined_sentences + 1
                self.not_skipped_sentences_examples.append(temp_inst_senteces)
                new_inst_tokens.append(temp_inst_tokens)

            new_inst_tags = []
            for i in range(len(inst_tags)):
                if len(inst_tags[i]) == 0:
                    continue
                temp_inst_tags = ['O']
                for j in range(i - self.window, i + self.window + 1):
                    if j < 0:
                        continue
                    elif j > len(inst_tags)-1:
                        break

                    if len(inst_tags[j]) != 0:
                        temp_inst_tags.extend(inst_tags[j])
                temp_inst_tags.extend(['O'])
                # TODO
                if len(temp_inst_tags) > 512:
                    print('Instance subtokens > 512 , skipping...new_inst_tags')
                    continue
                new_inst_tags.append(temp_inst_tags)

            new_inst_token_lens = []
            for i in range(len(inst_token_lens)):
                if len(inst_token_lens[i]) == 0:
                    continue
                temp_inst_token_lens = [1]
                for j in range(i - self.window, i + self.window + 1):
                    if j < 0:
                        continue
                    elif j > len(inst_token_lens) - 1:
                        break

                    if inst_token_lens[j] != [0]:
                        temp_inst_token_lens = temp_inst_token_lens + inst_token_lens[j]
                temp_inst_token_lens = temp_inst_token_lens + [1]
                # TODO
                if len(temp_inst_token_lens) > 512:
                    print('Instance subtokens > 512 , skipping...new_inst_token_lens')
                    continue
                new_inst_token_lens.append(temp_inst_token_lens)

            preprocessed_batch.append([
                new_inst_tokens,
                new_inst_token_lens,
                new_inst_tags
            ])

        return [e[0] for e in preprocessed_batch], [e[1] for e in preprocessed_batch], [e[2] for e in preprocessed_batch]

    def forward(self, instances, instances_sents):
        instances_logits_output = list()
        for inst, sents in zip(instances, instances_sents):
            # If GPU is available use cuda
            for s in inst:
                if self.use_cuda:
                    s = s.type(torch.LongTensor)
                    s = s.to('cuda')
                    inst_token_type_ids = torch.zeros(s.shape).type(torch.LongTensor).to('cuda')
                    inst_attention_mask = torch.ones(s.shape).type(torch.LongTensor).to('cuda')
                else:
                    inst_token_type_ids = torch.zeros(s.shape).type(torch.LongTensor)
                    inst_attention_mask = torch.ones(s.shape).type(torch.LongTensor)

                # Pass the input tokens through the BERT model and get the contextual representation for the tokens
                bert_out = self.bert(s,
                                     token_type_ids = inst_token_type_ids,
                                     attention_mask = inst_attention_mask)[0]

                linear_out = self.linear(bert_out)
                logits_out = self.linear_out_output(F.relu(linear_out))

                instances_logits_output.append(logits_out)

        return instances_logits_output