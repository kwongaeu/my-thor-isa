import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration


class LLMBackbone(nn.Module):
    def __init__(self, config):
        super(LLMBackbone, self).__init__()
        self.config = config
        self.engine = T5ForConditionalGeneration.from_pretrained(config.model_path)
#         access_token = "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" # huggingface token
#         self.engine = AutoModelForCausalLM.from_pretrained(config.model_path, token=access_token) # for Llama
#         self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, token=access_token) # for Llama
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        

    def forward(self, **kwargs):
        input_ids, input_masks, output_ids, output_masks = [kwargs[w] for w in '\
        input_ids, input_masks, output_ids, output_masks'.strip().split(', ')]
        output_ids[output_ids[:, :] == self.tokenizer.pad_token_id] = -100
        output = self.engine(input_ids, attention_mask=input_masks, decoder_input_ids=None,
                             decoder_attention_mask=output_masks, labels=output_ids)
        loss = output[0]
        return loss

    def generate(self, **kwargs):
        input_ids, input_masks = [kwargs[w] for w in '\
        input_ids, input_masks'.strip().split(', ')]
        output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks,
                                      max_length=self.config.max_length)
        dec = [self.tokenizer.decode(ids) for ids in output]
        output = [context.replace('<pad>', '').replace('</s>', '').strip() for context in dec]
        return output

    def evaluate(self, **kwargs):
        input_ids, input_masks = [kwargs[w] for w in '\
        input_ids, input_masks'.strip().split(', ')]
        output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks, max_length=200)
        dec = [self.tokenizer.decode(ids) for ids in output]
        label_dict = {w: i for i, w in enumerate(self.config.label_list)}
        output = []
        for ii, w in enumerate(dec):
#             print(w)
            x = w.replace('<pad>', '').replace('</s>', '').strip().lower()
            if ii == 0:
                print(x)
            fs = x.split('. ')[-1]
            if 'is neutral' in fs or 'is "neutral"' in fs or 'is slightly neutral' in fs or 'neutral' in fs:
                pred = 'neutral'
            elif 'is positive' in fs or 'is "positive"' in fs or 'is slightly positive' in fs or 'positive' in fs:
                pred = 'positive'
            elif 'is negative' in fs or 'is "negative"' in fs or 'is slightly negative' in fs or 'negative' in fs:
                pred = 'negative'
            else:
                print('sth wrong with answer extraction')
                print(x)
                print('-'*77)
                pred = 'neutral'
                
            output.append(label_dict.get(pred, 0))
#         output = [label_dict.get(w.replace('<pad>', '').replace('</s>', '').strip().lower().split('the answer is ')[1].replace('.', ''), 0) for w in dec]
#         print('output: ' + str(output))
        return output
