# -*- coding: utf-8 -*-
'''
@author: Yanchen Huang
@contact: yanchenhuang@smail.nju.edu.cn
@time: 2025/3/23
@desc:
'''
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import json
import random
import torch
from tqdm import tqdm
from utils import Preprocessor, load_data
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, data, shuffle=False):
        self.data = data
        self.length = len(data)
        self.indexes = list(range(self.length))
        if shuffle:
            random.shuffle(self.indexes)

    def __getitem__(self, index):
        index = self.indexes[index]
        return self.data[index]

    def __len__(self):
        return self.length

class DataMaker(object):
    def __init__(self, tokenizer, add_special_tokens=True, define_type='idiom'):
        super().__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.preprocessor = Preprocessor(tokenizer, self.add_special_tokens)
        if define_type == 'idiom':
            definition_f = open('/home/yukino/Documents/Code/NLP/CIMD/DEBERT/Idiom/Idiom_Definition.json', 'r', encoding='utf-8')
        else:
            definition_f = open('../xiandao/Word_Definition', 'r', encoding='utf-8')
        lines = definition_f.readlines()
        self.definitions = {}
        for idx, line in tqdm(enumerate(lines), desc='scan definition file : '):
            # if idx > 10000: break
            try:
                line = json.loads(line)
            except: continue
            self.definitions[line['word']] = {'definition': line['definition'], 'sentiment': line['sentiment']}
        print(f'total collect {len(self.definitions)} definition items')

    def get_definition(self, word):
        return f'{word}:{self.definitions.get(word, "")}'

    def generate_inputs(self, datas, max_seq_len, ent2id):

        ent_type_size = len(ent2id)  # 实体类别

        all_inputs = []
        for sample in tqdm(datas, desc='in generate inputs function : '):
            inputs = self.tokenizer(
                sample["text"],
                max_length=max_seq_len,
                truncation=True,
            )
            # tmp_label, define_positions = None, None

            ent2token_spans = self.preprocessor.get_ent2token_spans(sample["text"], sample["entity_list"])       # [(1, 4, 'err_idiom')]
            define_positions = [[start, end] for start, end, _ in ent2token_spans]
            # tmp_label = [[ent2id[label], start, end] for start, end, label in ent2token_spans if label == 'err_idiom']
            tmp_label = [[ent2id[label], start, end] for start, end, label in ent2token_spans if label != 'cor_idiom']

            # 针对句子中出现的所有单词生成单词释义
            e_maxlen = 500 // max(1, len(sample['entity_list']))
            define_ids, define_type_ids = [], []
            for count, worditem in enumerate(sample['entity_list']):      # (start, end, k)
                word = sample["text"][worditem[0]:worditem[1]+1]
                definition = self.definitions.get(word, {'definition': '[PAD]'})['definition']
                if definition is None: definition = '[PAD]'
                define_output = self.tokenizer(word + ':' + definition, max_length=e_maxlen, truncation=True)
                if count == len(sample['entity_list'])-1:           # the last one:
                    now_ids = define_output['input_ids']
                else:
                    now_ids = define_output['input_ids'][:-1]
                define_type_ids.extend([count%2] * len(now_ids))
                define_ids.extend(now_ids)

            assert len(define_ids) == len(define_type_ids)
            define_cls = [idx for idx, id in enumerate(define_ids) if id in [101, 102]]
            assert len(define_ids) == 0 or (len(define_cls) == len(sample['entity_list'])+1 and define_ids[-1] == 102)

            # prepare the input items
            input_ids = inputs["input_ids"][:max_seq_len]
            assert inputs["token_type_ids"][-1] == 0, '句子划分错误'
            attention_mask = inputs["attention_mask"][:max_seq_len]
            token_type_ids = inputs["token_type_ids"][:max_seq_len]

            match_word_ids = [0] * len(input_ids)
            for i, posit in enumerate(define_positions):
                beg, end = posit
                match_word_ids[beg:end+1] = [i+1] * (end-beg+1)
            match_word_ids = match_word_ids[:max_seq_len]


            # sample_input = (input_ids, attention_mask, token_type_ids, define_ids, define_attention_mask, define_type_ids, define_positions, tmp_label)
            sample_input = {
                'input_ids' : input_ids, 'attention_mask' : attention_mask, 'token_type_ids' : token_type_ids,
                'define_ids' : define_ids, 'define_type_ids' : define_type_ids,
                'define_cls' : define_cls, 'match_word_ids' : match_word_ids,
                'tmp_label' : tmp_label,
            }
            all_inputs.append(sample_input)
        return all_inputs

    def generate_batch(self, batch_data, data_type="train"):
        input_ids = [batch['input_ids'] for batch in batch_data]
        attention_mask = [batch['attention_mask'] for batch in batch_data]
        token_type_ids = [batch['token_type_ids'] for batch in batch_data]
        define_ids = [batch['define_ids'] for batch in batch_data]
        define_type_ids = [batch['define_type_ids'] for batch in batch_data]
        define_cls = [batch['define_cls'] for batch in batch_data]
        match_word_ids = [batch['match_word_ids'] for batch in batch_data]
        tmp_label = [batch['tmp_label'] for batch in batch_data]

        len_input_ids = [len(item) for item in input_ids]
        len_define_ids = [len(item) for item in define_ids]
        len_label= [len(item) for item in tmp_label]

        input_ids = torch.tensor([item + [0] * (max(len_input_ids)-len_input_ids[i]) for i, item in enumerate(input_ids)]).long()
        attention_mask = torch.tensor([item + [0] * (max(len_input_ids)-len_input_ids[i]) for i, item in enumerate(attention_mask)]).long()
        token_type_ids = torch.tensor([item + [1] * (max(len_input_ids)-len_input_ids[i]) for i, item in enumerate(token_type_ids)]).long()
        match_word_ids = torch.tensor([item + [0] * (max(len_input_ids)-len_input_ids[i]) for i, item in enumerate(match_word_ids)]).long()

        define_ids = torch.tensor([item + [0] * (max(len_define_ids) - len_define_ids[i]) for i, item in enumerate(define_ids)]).long()
        define_type_ids = torch.tensor([item + [1] * (max(len_define_ids) - len_define_ids[i]) for i, item in enumerate(define_type_ids)]).long()
        # define_cls = [item for item in define_cls]

        t_labels = []
        for i in range(len(batch_data)):
            item = tmp_label[i]
            if max(len_label) - len_label[i]:
                item += [[-1, -1, -1] for _ in range(max(len_label) - len_label[i])]
            t_labels.append(item)

        labels = torch.tensor(t_labels).long()

        return (input_ids, attention_mask, token_type_ids, define_ids, define_type_ids, define_cls, match_word_ids, labels)

if __name__ == '__main__':
    train_data = load_data('/home/data_ti4_c/caoyc/ChID/code/GlobalPointer/datasets/ChID/sim_analysis.json', "train")[:100]
    ent2id = load_data('/home/data_ti4_c/caoyc/ChID/code/GlobalPointer/datasets/ChID/ent2id.json', "ent2id")

    from transformers import BertTokenizerFast, BertConfig
    tokenizer = BertTokenizerFast.from_pretrained("/home/data_ti4_c/caoyc/Pretrains/chinese-roberta-wwm-ext")
    data_maker = DataMaker(tokenizer)
    all_data = data_maker.generate_inputs(train_data, max_seq_len=300, ent2id=ent2id)
    dataloader = DataLoader(MyDataset(all_data, shuffle=False),
                            batch_size=2,
                            shuffle=False,
                            num_workers=0,
                            drop_last=False,
                            collate_fn=data_maker.generate_batch
                            )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from debert_modeling import DEBertModel
    config = BertConfig.from_pretrained("/home/data_ti4_c/caoyc/Pretrains/chinese-roberta-wwm-ext")
    define_mix_mode, define_encoder_max_layer = 'last', 5
    if define_mix_mode == 'last':
        add_layers = [define_encoder_max_layer]
    setattr(config, 'add_layers', add_layers)
    model = DEBertModel.from_pretrained("/home/data_ti4_c/caoyc/Pretrains/chinese-roberta-wwm-ext", config=config).to(device)

    for batid, batch_data in enumerate(dataloader):
        batch_data = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch_data]
        input_ids, attention_mask, token_type_ids, define_ids, define_type_ids, define_cls, match_word_ids, labels = batch_data

        append_params = {
            'define_ids': define_ids,
            'define_type_ids': define_type_ids,
            'match_word_ids': match_word_ids,
            'define_cls': define_cls,

            # params for model
            'define_encoder_max_layer': define_encoder_max_layer,
            'define_mix_mode': define_mix_mode,
        }
        output = model(input_ids, attention_mask, token_type_ids, append_params=append_params)
