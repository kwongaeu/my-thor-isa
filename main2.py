import argparse
import yaml
import torch
from attrdict import AttrDict
import pandas as pd

from src.utils import set_seed, load_params_LLM
from loader2 import MyDataLoader
from src.model import LLMBackbone
from src.engine import PromptTrainer, ThorTrainer, FewshotTrainer
import time

class Template:
    def __init__(self, args):
        config = AttrDict(yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader))
        names = []
        for k, v in vars(args).items():
            setattr(config, k, v)
        config.dataname = config.data_name
        set_seed(config.seed)
        print('config.seed=' + config.seed)

        config.device = torch.device('cuda:{}'.format(config.cuda_index) if torch.cuda.is_available() else 'cpu')
        names = [config.model_size, config.dataname] + names
        config.save_name = '_'.join(list(map(str, names))) + '_{}.pth.tar'
        self.config = config

    def forward(self):
        if self.config.reasoning == 'prompt_twice':
            (self.trainLoader, self.validLoader, self.testLoader), self.config = MyDataLoader(self.config).get_data_twice()
        elif self.config.reasoning == 'fewshot':
            (self.trainLoader, self.validLoader, self.testLoader), self.config = MyDataLoader(self.config).get_data_fewshot()
        else: # 'prompt' or 'thor'
            (self.trainLoader, self.validLoader, self.testLoader), self.config = MyDataLoader(self.config).get_data()

        self.model = LLMBackbone(config=self.config).to(self.config.device)
        self.config = load_params_LLM(self.config, self.model, self.trainLoader)

        print(f"Running on the {self.config.data_name} data.")
        if self.config.reasoning in ['prompt', 'prompt_twice']:
            print("Choosing prompt one-step infer mode.")
            trainer = PromptTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        elif self.config.reasoning == 'thor':
            print("Choosing thor multi-step infer mode.")
            trainer = ThorTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        elif self.config.reasoning == 'fewshot':
            print("Choosing few-shot prompting infer mode.")
            trainer = FewshotTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        else:
            raise 'Should choose a correct reasoning mode: prompt or thor.'

        if self.config.zero_shot == True:
            print("Zero-shot mode for evaluation.")
            time_taken = time.time()
            r = trainer.evaluate_step(self.testLoader, 'test')
            print(r)
            print('-' * 77)
            print('Time took: ', time.time() - time_taken)
            print('-' * 77)
            return

        print("Fine-tuning mode for training.")
        trainer.train()
        lines = trainer.lines

        df = pd.DataFrame(lines)
        print(df.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda_index', default=0)
    parser.add_argument('-r', '--reasoning', default='thor', choices=['prompt', 'prompt_twice', 'thor', 'fewshot'],
                        help='with one-step prompt or multi-step thor reasoning')
    parser.add_argument('-z', '--zero_shot', action='store_true', default=True,
                        help='running under zero-shot mode or fine-tune mode')
    parser.add_argument('-d', '--data_name', default='laptops', choices=['restaurants', 'laptops'],
                        help='semeval data name')
    parser.add_argument('--add_phrase', default=None, type=str)
    parser.add_argument('--few_shot_example_indices',
                        type=lambda s: [int(item) for item in s.split(',')],
                        default = None,
                        help='few_shot_example_indices')
    parser.add_argument('-s', '--seed', default=42, help='seed', type=int)
    parser.add_argument('-f', '--config', default='./config/config.yaml', help='config file')
    args = parser.parse_args()
    template = Template(args)
    template.forward()
