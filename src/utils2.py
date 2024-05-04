import os
import random
import torch
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def prompt_direct_inferring(context, target):
    new_context = f'Given the sentence "{context}", '
    prompt = new_context + f'what is the sentiment polarity (positive, negative, or neutral) towards {target}?'
    return new_context, prompt

# BSF
def prompt_direct_inferring_twice(context, target, add_phrase):
    new_context = f'Given the sentence: "{context}", '
    prompt = new_context + f'what is the sentiment polarity (positive, negative, or neutral) towards {target}? ' + add_phrase
    return new_context, prompt

def prompt_direct_inferring_masked(context, target):
    new_context = f'Given the sentence "{context}", '
    prompt = new_context + f'the sentiment polarity towards {target} is [mask]'
    return new_context, prompt


def prompt_for_aspect_inferring(context, target):
    new_context = f'Given the sentence "{context}", '
    prompt = new_context + f'which specific aspect of {target} is possibly mentioned?'
    return new_context, prompt


def prompt_for_opinion_inferring(context, target, aspect_expr):
    new_context = context + ' The mentioned aspect is about ' + aspect_expr + '.'
    prompt = new_context + f' Based on the common sense, what is the implicit opinion towards the mentioned aspect of {target}, and why?'
    return new_context, prompt


def prompt_for_polarity_inferring(context, target, opinion_expr):
    new_context = context + f' The opinion towards the mentioned aspect of {target} is ' + opinion_expr + '.'
    prompt = new_context + f' Based on such opinion, what is the sentiment polarity towards {target}?'
    return new_context, prompt


def prompt_for_polarity_label(context, polarity_expr):
    prompt = context + f' The sentiment polarity is {polarity_expr}.' + ' Based on these contexts, summarize and return the sentiment polarity only, such as positive, neutral, or negative.'
    return prompt

def prompt_for_fewshot_examples(context, target, few_shot_examples, show_prompt = False):
    with open('./src/few_shot_prompts.txt') as f:
        instruction_prompt = f.read()

    instruction_prompt = instruction_prompt.replace('[sent1]', few_shot_examples['sent'][0]).replace('[target1]', few_shot_examples['target'][0]).replace('[answer1]', few_shot_examples['answer'][0])
    instruction_prompt = instruction_prompt.replace('[sent2]', few_shot_examples['sent'][1]).replace('[target2]', few_shot_examples['target'][1]).replace('[answer2]', few_shot_examples['answer'][1])
    instruction_prompt = instruction_prompt.replace('[sent3]', few_shot_examples['sent'][2]).replace('[target3]', few_shot_examples['target'][2]).replace('[answer3]', few_shot_examples['answer'][2])
    instruction_prompt = instruction_prompt.replace('[sent4]', few_shot_examples['sent'][3]).replace('[target4]', few_shot_examples['target'][3]).replace('[answer4]', few_shot_examples['answer'][3])
    instruction_prompt = instruction_prompt.replace('[sent5]', few_shot_examples['sent'][4]).replace('[target5]', few_shot_examples['target'][4]).replace('[answer5]', few_shot_examples['answer'][4])
    instruction_prompt = instruction_prompt.replace('[sent6]', few_shot_examples['sent'][5]).replace('[target6]', few_shot_examples['target'][5]).replace('[answer6]', few_shot_examples['answer'][5])
    
    instruction_prompt = instruction_prompt.replace('[sent_test]', context).replace('[target_test]', target)

    if show_prompt:
        print('-'*77)
        print(instruction_prompt)
    return instruction_prompt


def set_seed(seed):
    print('seed: ', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    # torch.set_deterministic(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_params_LLM(config, model, fold_data):
    no_decay = ['bias', 'LayerNorm.weight']
    named = (list(model.named_parameters()))
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named if not any(nd in n for nd in no_decay)],
         'lr': float(config.bert_lr),
         'weight_decay': float(config.weight_decay)},
        {'params': [p for n, p in named if any(nd in n for nd in no_decay)],
         'lr': float(config.bert_lr),
         'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=float(config.adam_epsilon))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=config.epoch_size * fold_data.__len__())
    config.score_manager = ScoreManager()
    config.optimizer = optimizer
    config.scheduler = scheduler
    return config


class ScoreManager:
    def __init__(self) -> None:
        self.score = []
        self.line = []

    def add_instance(self, score, res):
        self.score.append(score)
        self.line.append(res)

    def get_best(self):
        best_id = np.argmax(self.score)
        res = self.line[best_id]
        return res
