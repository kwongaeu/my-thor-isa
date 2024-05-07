import os
import math
import torch
import numpy as np
import pickle as pkl
from src.utils import prompt_direct_inferring, prompt_direct_inferring_twice, prompt_direct_inferring_few_shot, prompt_direct_inferring_masked, prompt_for_aspect_inferring
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import random


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.data_length = 0

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# +
class MyDataLoader:
    def __init__(self, config):
        self.config = config
        config.preprocessor = Preprocessor(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

        self.add_phrase = self.config.add_phrase
        self.few_shot_example_indices = self.config.few_shot_example_indices

    def worker_init(self, worked_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_data(self):
        cfg = self.config
        path = os.path.join(self.config.preprocessed_dir,
                            '{}_{}_{}.pkl'.format(cfg.data_name, cfg.model_size, cfg.model_path).replace('/', '-'))
        if os.path.exists(path):
            self.data = pkl.load(open(path, 'rb'))
        else:
            self.data = self.config.preprocessor.forward()
            pkl.dump(self.data, open(path, 'wb'))

        train_data, valid_data, test_data = self.data[:3]
        self.config.word_dict = self.data[-1]
        
        print('len(test_data): ', len(test_data))
        print('test_data[0][0]: ', test_data[0][0])
        print('-'*77)
        print('test_data[0][1]: ', test_data[0][1])
        print('-'*77)
        print('test_data[0][2]: ', test_data[0][2])
        print('-'*77)
        print('test_data[0][3]: ', test_data[0][3])
        print('-'*77)
        
        
        load_data = lambda dataset: DataLoader(MyDataset(dataset), num_workers=0, worker_init_fn=self.worker_init, \
                                               shuffle=self.config.shuffle, batch_size=self.config.batch_size,
                                               collate_fn=self.collate_fn)
        train_loader, valid_loader, test_loader = map(load_data, [train_data, valid_data, test_data])
        train_loader.data_length, valid_loader.data_length, test_loader.data_length = math.ceil(
            len(train_data) / self.config.batch_size), \
            math.ceil(len(valid_data) / self.config.batch_size), \
            math.ceil(len(test_data) / self.config.batch_size)

        res = [train_loader, valid_loader, test_loader]

        return res, self.config

    def get_data_twice(self):
        cfg = self.config
        path = os.path.join(self.config.preprocessed_dir,
                            '{}_{}_{}.pkl'.format(cfg.data_name, cfg.model_size, cfg.model_path).replace('/', '-'))
        if os.path.exists(path):
            self.data = pkl.load(open(path, 'rb'))
        else:
            self.data = self.config.preprocessor.forward()
            pkl.dump(self.data, open(path, 'wb'))

        train_data, valid_data, test_data = self.data[:3]
        self.config.word_dict = self.data[-1]
        
        print('len(test_data): ', len(test_data))
        print('test_data[0][0]: ', test_data[0][0])
        print('-'*77)
        print('test_data[0][1]: ', test_data[0][1])
        print('-'*77)
        print('test_data[0][2]: ', test_data[0][2])
        print('-'*77)
        print('test_data[0][3]: ', test_data[0][3])
        print('-'*77)
        
        load_data = lambda dataset: DataLoader(MyDataset(dataset), num_workers=0, worker_init_fn=self.worker_init, \
                                               shuffle=self.config.shuffle, batch_size=self.config.batch_size,
                                               collate_fn=self.collate_fn_twice)
        train_loader, valid_loader, test_loader = map(load_data, [train_data, valid_data, test_data])
        train_loader.data_length, valid_loader.data_length, test_loader.data_length = math.ceil(
            len(train_data) / self.config.batch_size), \
            math.ceil(len(valid_data) / self.config.batch_size), \
            math.ceil(len(test_data) / self.config.batch_size)

        res = [train_loader, valid_loader, test_loader]

        return res, self.config

    def get_data_few_shot(self):
        cfg = self.config
        path = os.path.join(self.config.preprocessed_dir,
                            '{}_{}_{}.pkl'.format(cfg.data_name, cfg.model_size, cfg.model_path).replace('/', '-'))
        if os.path.exists(path):
            self.data = pkl.load(open(path, 'rb'))
        else:
            self.data = self.config.preprocessor.forward()
            pkl.dump(self.data, open(path, 'wb'))

        train_data, valid_data, test_data = self.data[:3]
        self.config.word_dict = self.data[-1]
        
        self.few_shot_examples = {'sent': [], 'target': [], 'answer': []}

        random_indices = list(range(len(train_data)))
        random.shuffle(random_indices)
        self.few_shot_example_indices = []

        answer_l = {0: 'positive', 1: 'negative', 2: 'neutral'}

        n_pos = 0
        n_neut = 0 
        n_neg = 0
        
        n_imp = 0
        n_exp = 0
        
#         # Consider implicit + polarity
#         for ind in random_indices:
#             print("ind: " + str(ind))
#             if train_data[ind][2] == 0 and train_data[ind][3] == 0: # explicit pos
#                 self.few_shot_example_indices.append(ind)
#                 n_exp += 1
#                 n_pos += 1
#             elif train_data[ind][2] == 1 and train_data[ind][3] == 0: # explicit neg
#                 self.few_shot_example_indices.append(ind)
#                 n_exp += 1
#                 n_neg += 1
#             elif train_data[ind][2] == 2 and train_data[ind][3] == 0: # explicit neut
#                 self.few_shot_example_indices.append(ind)
#                 n_exp += 1
#                 n_neut += 1
#             if n_exp == 3:
#                 print("filled all explicit samples")
#                 break
                
#         for ind in random_indices:
#             print("ind: " + str(ind))
#             if train_data[ind][2] == 0 and train_data[ind][3] == 1: # implicit pos
#                 self.few_shot_example_indices.append(ind)
#                 n_imp += 1
#                 n_pos += 1
#             elif train_data[ind][2] == 1 and train_data[ind][3] == 1: # implicit neg
#                 self.few_shot_example_indices.append(ind)
#                 n_imp += 1
#                 n_neg += 1
#             elif train_data[ind][2] == 2 and train_data[ind][3] == 1: # implicit neut
#                 self.few_shot_example_indices.append(ind)
#                 n_imp += 1
#                 n_neut += 1
#             if n_imp == 6:
#                 print("filled all implicit samples")
#                 break
        
#         # Consider implicit/explicit
#         for ind in random_indices:
#             # implicit is labeled at idx 3
#             print("ind: " + str(ind))
#             if train_data[ind][3] == 0:  # false
#                 self.few_shot_example_indices.append(ind)
#                 n_exp += 1
#             if n_exp == 3:
#                 print("filled all explicit samples")
#                 break
                
#         for ind in random_indices:
#             # implicit is labeled at idx 3
#             print("ind: " + str(ind))
#             if train_data[ind][3] == 1:  # true
#                 self.few_shot_example_indices.append(ind)
#                 n_imp += 1
#             if n_imp == 3:
#                 print("filled all implicit samples")
#                 break
        
        # Consider polarity
        for ind in random_indices:
            # polarity is labeled at idx 2
            print("ind: " + str(ind))
            if train_data[ind][2] == 0 and train_data[ind][3] == 1:
                self.few_shot_example_indices.append(ind)
                n_pos += 1
            if n_pos == 2:
                print("filled all positive samples")
                break

        for ind in random_indices:
            print("ind: " + str(ind))
            if train_data[ind][2] == 1 and train_data[ind][3] == 1:
                self.few_shot_example_indices.append(ind)
                n_neg += 1
            if n_neg == 2:
                print("filled all negative samples")
                break

        for ind in random_indices:
            print("ind: " + str(ind))
            if train_data[ind][2] == 2 and train_data[ind][3] == 1:
                self.few_shot_example_indices.append(ind)
                n_neut += 1
            if n_neut == 2:
                print("filled all neutral samples")
                break

        print('self.few_show_example_indices: ', self.few_shot_example_indices)

        
        for index in self.few_shot_example_indices:
            self.few_shot_examples['sent'].append(train_data[index][0])
            self.few_shot_examples['target'].append(train_data[index][1])
            self.few_shot_examples['answer'].append(answer_l[train_data[index][2]])
            
        load_data = lambda dataset: DataLoader(MyDataset(dataset), num_workers=0, worker_init_fn=self.worker_init, \
                                               shuffle=self.config.shuffle, batch_size=self.config.batch_size,
                                               collate_fn=self.collate_fn_few_shot)
        train_loader, valid_loader, test_loader = map(load_data, [train_data, valid_data, test_data])
        train_loader.data_length, valid_loader.data_length, test_loader.data_length = math.ceil(
            len(train_data) / self.config.batch_size), \
            math.ceil(len(valid_data) / self.config.batch_size), \
            math.ceil(len(test_data) / self.config.batch_size)

        res = [train_loader, valid_loader, test_loader]

        return res, self.config

    def collate_fn(self, data):
        input_tokens, input_targets, input_labels, implicits = zip(*data)
        if self.config.reasoning == 'prompt':
            new_tokens = []
            for i, line in enumerate(input_tokens):
                line = ' '.join(line.split()[:self.config.max_length - 25])
                if self.config.zero_shot == True:
                    _, prompt = prompt_direct_inferring(line, input_targets[i])
                else:
                    _, prompt = prompt_direct_inferring_masked(line, input_targets[i])
                new_tokens.append(prompt)

            batch_input = self.tokenizer.batch_encode_plus(new_tokens, padding=True, return_tensors='pt',
                                                           max_length=self.config.max_length)
            batch_input = batch_input.data

            labels = [self.config.label_list[int(w)] for w in input_labels]
            batch_output = self.tokenizer.batch_encode_plus(labels, max_length=3, padding=True,
                                                            return_tensors="pt").data

            res = {
                'input_ids': batch_input['input_ids'],
                'input_masks': batch_input['attention_mask'],
                'output_ids': batch_output['input_ids'],
                'output_masks': batch_output['attention_mask'],
                'input_labels': torch.tensor(input_labels),
                'implicits': torch.tensor(implicits)
            }
            res = {k: v.to(self.config.device) for k, v in res.items()}
            return res

        elif self.config.reasoning == 'thor':

            new_tokens = []
            contexts_A = []
            for i, line in enumerate(input_tokens):
                line = ' '.join(line.split()[:self.config.max_length - 25])
                context_step1, prompt = prompt_for_aspect_inferring(line, input_targets[i])
                contexts_A.append(context_step1)
                new_tokens.append(prompt)

            batch_contexts_A = self.tokenizer.batch_encode_plus(contexts_A, padding=True, return_tensors='pt',
                                                                max_length=self.config.max_length)
            batch_contexts_A = batch_contexts_A.data
            batch_targets = self.tokenizer.batch_encode_plus(list(input_targets), padding=True, return_tensors='pt',
                                                             max_length=self.config.max_length)
            batch_targets = batch_targets.data
            batch_input = self.tokenizer.batch_encode_plus(new_tokens, padding=True, return_tensors='pt',
                                                           max_length=self.config.max_length)
            batch_input = batch_input.data

            labels = [self.config.label_list[int(w)] for w in input_labels]
            batch_output = self.tokenizer.batch_encode_plus(labels, max_length=3, padding=True,
                                                            return_tensors="pt").data

            res = {
                'input_ids': batch_input['input_ids'],
                'input_masks': batch_input['attention_mask'],
                'context_A_ids': batch_contexts_A['input_ids'],
                'target_ids': batch_targets['input_ids'],
                'output_ids': batch_output['input_ids'],
                'output_masks': batch_output['attention_mask'],
                'input_labels': torch.tensor(input_labels),
                'implicits': torch.tensor(implicits)
            }
            res = {k: v.to(self.config.device) for k, v in res.items()}
            return res

        else:
            raise 'choose correct reasoning mode: prompt or thor.'

    def collate_fn_twice(self, data):
        input_tokens, input_targets, input_labels, implicits = zip(*data)
        if self.config.reasoning == 'prompt_twice':
            new_tokens = []
            for i, line in enumerate(input_tokens):
                line = ' '.join(line.split()[:self.config.max_length - 25])
                if self.config.zero_shot == True:
                    _, prompt = prompt_direct_inferring_twice(line, input_targets[i], self.add_phrase)
                else:
                    _, prompt = prompt_direct_inferring_masked(line, input_targets[i])
                new_tokens.append(prompt)

            batch_input = self.tokenizer.batch_encode_plus(new_tokens, padding=True, return_tensors='pt',
                                                           max_length=self.config.max_length)
            batch_input = batch_input.data

            labels = [self.config.label_list[int(w)] for w in input_labels]
            batch_output = self.tokenizer.batch_encode_plus(labels, max_length=3, padding=True,
                                                            return_tensors="pt").data

            res = {
                'input_ids': batch_input['input_ids'],
                'input_masks': batch_input['attention_mask'],
                'output_ids': batch_output['input_ids'],
                'output_masks': batch_output['attention_mask'],
                'input_labels': torch.tensor(input_labels),
                'implicits': torch.tensor(implicits)
            }
            res = {k: v.to(self.config.device) for k, v in res.items()}
            return res

        elif self.config.reasoning == 'thor':

            new_tokens = []
            contexts_A = []
            for i, line in enumerate(input_tokens):
                line = ' '.join(line.split()[:self.config.max_length - 25])
                context_step1, prompt = prompt_for_aspect_inferring(line, input_targets[i])
                contexts_A.append(context_step1)
                new_tokens.append(prompt)

            batch_contexts_A = self.tokenizer.batch_encode_plus(contexts_A, padding=True, return_tensors='pt',
                                                                max_length=self.config.max_length)
            batch_contexts_A = batch_contexts_A.data
            batch_targets = self.tokenizer.batch_encode_plus(list(input_targets), padding=True, return_tensors='pt',
                                                             max_length=self.config.max_length)
            batch_targets = batch_targets.data
            batch_input = self.tokenizer.batch_encode_plus(new_tokens, padding=True, return_tensors='pt',
                                                           max_length=self.config.max_length)
            batch_input = batch_input.data

            labels = [self.config.label_list[int(w)] for w in input_labels]
            batch_output = self.tokenizer.batch_encode_plus(labels, max_length=3, padding=True,
                                                            return_tensors="pt").data

            res = {
                'input_ids': batch_input['input_ids'],
                'input_masks': batch_input['attention_mask'],
                'context_A_ids': batch_contexts_A['input_ids'],
                'target_ids': batch_targets['input_ids'],
                'output_ids': batch_output['input_ids'],
                'output_masks': batch_output['attention_mask'],
                'input_labels': torch.tensor(input_labels),
                'implicits': torch.tensor(implicits)
            }
            res = {k: v.to(self.config.device) for k, v in res.items()}
            return res

        else:
            raise 'choose correct reasoning mode: prompt or thor.'

    def collate_fn_few_shot(self, data):
        input_tokens, input_targets, input_labels, implicits = zip(*data)
        
        if self.config.reasoning == 'prompt_few_shot':
            new_tokens = []
            raw_texts = []
            show_prompt_once = False
            for i, line in enumerate(input_tokens):
                line = ' '.join(line.split()[:self.config.max_length - 25])
                raw_texts.append(line)
                if self.config.zero_shot == True:
                    prompt = prompt_direct_inferring_few_shot(line, input_targets[i], few_shot_examples = self.few_shot_examples,
                                                              show_prompt = show_prompt_once)
                    show_prompt_once = False
                else:
                    _, prompt = prompt_direct_inferring_masked(line, input_targets[i])
                new_tokens.append(prompt)

            batch_input = self.tokenizer.batch_encode_plus(new_tokens, padding=True, return_tensors='pt',
                                                           max_length=self.config.max_length)
            batch_input = batch_input.data

            labels = [self.config.label_list[int(w)] for w in input_labels]
            batch_output = self.tokenizer.batch_encode_plus(labels, max_length=3, padding=True,
                                                            return_tensors="pt").data
            ##new_tokens_tensor = torch.tensor(new_tokens)
            res = {
                'raw_texts': raw_texts,
                'input_ids': batch_input['input_ids'],
                'input_masks': batch_input['attention_mask'],
                'output_ids': batch_output['input_ids'],
                'output_masks': batch_output['attention_mask'],
                'input_labels': torch.tensor(input_labels),
                'implicits': torch.tensor(implicits)
            }
            #res = {k: v.to(self.config.device) for k, v in res.items()}
            res = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v for k, v in res.items()}
            return res
# -



class Preprocessor:
    def __init__(self, config):
        self.config = config

    def read_file(self):
        dataname = self.config.dataname
        train_file = os.path.join(self.config.data_dir, dataname,
                                  '{}_Train_v2_Implicit_Labeled_preprocess_finetune.pkl'.format(dataname.capitalize()))
        test_file = os.path.join(self.config.data_dir, dataname,
                                 '{}_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl'.format(dataname.capitalize()))
        train_data = pkl.load(open(train_file, 'rb'))
        test_data = pkl.load(open(test_file, 'rb'))
        ids = np.arange(len(train_data))
        np.random.shuffle(ids)
        lens = 150
        valid_data = {w: v[-lens:] for w, v in train_data.items()}
        train_data = {w: v[:-lens] for w, v in train_data.items()}

        return train_data, valid_data, test_data

    def transformer2indices(self, cur_data):
        res = []
        for i in range(len(cur_data['raw_texts'])):
            text = cur_data['raw_texts'][i]
            target = cur_data['raw_aspect_terms'][i]
            implicit = 0
            if 'implicits' in cur_data:
                implicit = cur_data['implicits'][i]
            label = cur_data['labels'][i]
            implicit = int(implicit)
            res.append([text, target, label, implicit])
        return res

    def forward(self):
        modes = 'train valid test'.split()
        dataset = self.read_file()
        res = []
        for i, mode in enumerate(modes):
            data = self.transformer2indices(dataset[i])
            res.append(data)
        return res
