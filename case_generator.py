# -*- coding: utf-8 -*-
'''
@author: Yongchang Cao
@contact: cyc990520@gmail.com
@file: trainer.py
@time: 2022/3/23 21:53
@desc:
'''
import json
import os
gpuid = '7'
print(gpuid)
os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

config = {
    'run_type': 'test',
    'reload_path':'/home/caoyc/ChID/DEBERT/cross_11_03-30_12:40:29',
    # 'test_data' : 'ran_v2_analysis.txt',          #
    'test_data' : 'gaokao_v2_analysis.json',          #
    'path_to_save_model': '/home/caoyc/ChID/DEBERT/ablation',
    "bert_path": "/home/data_ti4_c/caoyc/Pretrains/chinese-roberta-wwm-ext",
    'data_home': '/home/data_ti4_c/caoyc/ChID/code/GlobalPointer/datasets/ChID',
    'ent2id': 'ent2id.json',
    'valid_ran_data': 'ran_v2_analysis.txt',
    'valid_sim_data': 'sim_v2_analysis.txt',
    'max_seq_len': 300,
    'batch_size': 20,
    'epochs': 10,
    'lr': 3e-5,
    'define_encoder_max_layer': 10,
    'define_mix_mode': 'all',          # last, all, cross
    'max_train_size':60000,
    'use_half': True,
    'use_cache':True,
}
if config['run_type'] == 'test':
    config['define_mix_mode']=config['reload_path'].split('/')[-1].split('_')[0]
    config['define_encoder_max_layer']=int(config['reload_path'].split('/')[-1].split('_')[1])

def busy_gpu(min_memory = 0):
    wait_id = int(gpuid)
    infos = os.popen('nvidia-smi|grep %').read().split('\n')
    for i, info in enumerate(infos):
        if i != wait_id: continue
        if len(info) < 2: continue
        info = info.split('|')
        power = int(info[1].split()[-3][:-1])
        memory = int(info[2].split('/')[0].strip()[:-3])
        if memory <= min_memory: return False
    return True

import sys, time
wait_time = 30
i = 1
while True and config['run_type']=='train':
    i = i % 100
    symbol = 'monitoring: ' + '>' * i + ' ' * (100 - i - 1) + '|'
    if not busy_gpu(5000) and i > 1:
        time.sleep(wait_time)
    if not busy_gpu(5000): break
    sys.stdout.write('\r' + symbol)
    sys.stdout.flush()
    time.sleep(wait_time)
    i += 1

import torch
import random
from tqdm import tqdm
from transformers import BertTokenizerFast, BertConfig
from debert_dataloader import MyDataset, DataMaker
from utils import Logger, load_data, multilabel_categorical_crossentropy
from torch.utils.data import DataLoader
from GlobalPointer import MetricsCalculator, GlobalPointer
from torch.cuda.amp import autocast, GradScaler
if config['define_mix_mode'] == 'cross':
    from debert_cross_modeling import DEBertModel
else:
    from debert_modeling import DEBertModel

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config["num_workers"] = 2 if sys.platform.startswith("linux") else 0
seed = random.randint(0, 1e4)
torch.manual_seed(seed)  # pytorch random seed
torch.backends.cudnn.deterministic = True

tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=True, do_lower_case=False)
ent2id_path = os.path.join(config["data_home"], config["ent2id"])
ent2id = load_data(ent2id_path, "ent2id")
ent_type_size = len(ent2id)
test_data = []

class NewDataMaker(DataMaker):

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
            orispan = [sample["text"][a:b] for a, b in inputs[0].offsets]


            ent2token_spans = self.preprocessor.get_ent2token_spans(sample["text"], sample["entity_list"])       # [(1, 4, 'err_idiom')]
            define_positions = [[start, end] for start, end, _ in ent2token_spans]
            tmp_label = [[ent2id[label], start, end] for start, end, label in ent2token_spans if label == 'err_idiom']

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
            input_ids = inputs["input_ids"]
            assert inputs["token_type_ids"][-1] == 0, '句子划分错误'
            attention_mask = inputs["attention_mask"]
            token_type_ids = inputs["token_type_ids"]

            match_word_ids = [0] * len(input_ids)
            for i, posit in enumerate(define_positions):
                beg, end = posit
                match_word_ids[beg:end+1] = [i+1] * (end-beg+1)


            # sample_input = (input_ids, attention_mask, token_type_ids, define_ids, define_attention_mask, define_type_ids, define_positions, tmp_label)
            sample_input = {
                'input_ids' : input_ids, 'attention_mask' : attention_mask, 'token_type_ids' : token_type_ids,
                'define_ids' : define_ids, 'define_type_ids' : define_type_ids,
                'define_cls' : define_cls, 'match_word_ids' : match_word_ids,
                'tmp_label' : tmp_label, 'orispan':orispan,
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
        orispan = [batch['orispan'] for batch in batch_data]

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

        return (input_ids, attention_mask, token_type_ids, define_ids, define_type_ids, define_cls, match_word_ids, labels, orispan)

def data_generator(run_type='train', cache_path='/home/caoyc/ChID/DEBERT/data_cache.pt', use_cache=True):
    global test_data
    data_maker = NewDataMaker(tokenizer)
    if run_type != 'train':
        test_data_path = os.path.join(config["data_home"], config["test_data"])  ## train_data
        test_data = load_data(test_data_path, "valid")
        max_seq_len = config['max_seq_len']
        test_inputs = data_maker.generate_inputs(test_data, max_seq_len, ent2id)
        test_dataloader = DataLoader(MyDataset(test_inputs),
                                     batch_size=config["batch_size"],
                                     shuffle=False,
                                     num_workers=config["num_workers"],
                                     drop_last=False,
                                     collate_fn=data_maker.generate_batch
                                     )
        return test_dataloader

metrics = MetricsCalculator()
import numpy as np
def get_labeled_line(y_pred, y_true, length_info, orispan):
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    length = torch.sum(length_info > 0, dim=-1)
    pred = []
    true = []
    for b, l, start, end in zip(*np.where(y_pred > 0)):
        if end > length[b] or end <= start: continue
        pred.append((b, l, start, end))
    for b, l, start, end in zip(*np.where(y_true > 0)):
        true.append((b, l, start, end))

    pred.sort(key=lambda x:x[0])
    true.sort(key=lambda x:x[0])

    pred_lines = [[''.join(item), [], []] for item in orispan]
    for b, l, start, end in pred:
        pred_lines[b][1].append([int(start), int(end), ''.join(orispan[b][start:end+1])])
    for b, l, start, end in true:
        pred_lines[b][2].append([int(start), int(end), ''.join(orispan[b][start:end+1])])

    R = set(pred)
    T = set(true)
    X = len(R & T)
    Y = len(R)
    Z = len(T)
    return X, Y, Z, pred_lines

def valid_step(batch_valid, model):
    batch_data = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch_valid]
    input_ids, attention_mask, token_type_ids, define_ids, define_type_ids, define_cls, match_word_ids, labels, orispan = batch_data

    append_params = {
        'define_ids': define_ids,
        'define_type_ids': define_type_ids,
        'match_word_ids': match_word_ids,
        'define_cls': define_cls,
        'define_encoder_max_layer': config['define_encoder_max_layer'],
        'define_mix_mode': config['define_mix_mode'],
    }
    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids, append_params=append_params)

    new_batch_label = torch.zeros([labels.size()[0], 1, logits.size()[2], logits.size()[2]], device=labels.device)
    for batid in range(labels.size()[0]):
        for labelid in range(labels.size()[1]):
            if labels[batid, labelid, 0] == -1:
                break
            new_batch_label[batid, 0, labels[batid, labelid, 1].item(), labels[batid, labelid, 2].item()] = 1
    batch_labels = new_batch_label

    sample_f1, sample_precision, sample_recall, pred_lines = get_labeled_line(logits, batch_labels, attention_mask, orispan)
    return sample_f1, sample_precision, sample_recall, pred_lines

def valid(model, dataloader):
    model.eval()
    total_X, total_Y, total_Z = 0, 0, 0
    total_pred_line = []
    eval_start_time = time.time()
    for batch_data in tqdm(dataloader, desc="Validating"):
        X, Y, Z, pred_lines = valid_step(batch_data, model)
        total_X += X
        total_Y += Y
        total_Z += Z
        total_pred_line += pred_lines
    eval_end_time = time.time()
    print('eval epoch time : ')
    print(eval_end_time - eval_start_time)

    avg_f1 = 2 * total_X / (total_Y + total_Z + 1e-8)
    avg_precision = total_X / (total_Y + 1e-8)
    avg_recall = total_X / (total_Z + 1e-8)

    print("******************************************")
    print(f'avg_precision: {total_X} / {total_Y} = {avg_precision*100:.4f}, avg_recall: {total_X} / {total_Z} = {avg_recall*100:.4f}, avg_f1: {avg_f1*100:.4f}')
    print(f'{avg_precision*100:.4f}, {avg_recall*100:.4f}, {avg_f1*100:.4f}')
    print("******************************************")
    return avg_f1, total_pred_line

if __name__ == '__main__':
    if config['run_type'] != 'train':
        test_dataloader = data_generator(run_type='test')

    bertconfig = BertConfig.from_pretrained(config["bert_path"])
    if type(config['define_encoder_max_layer']) == type(list()):
        add_layers = config['define_encoder_max_layer']
    elif config['define_mix_mode'] == 'last':
        add_layers = [config['define_encoder_max_layer']]
    elif config['define_mix_mode'] == 'all':
        add_layers = [i for i in range(config['define_encoder_max_layer']+1)]
    elif config['define_mix_mode'] == 'cross':
        add_layers = [i for i in range(12)]

    setattr(bertconfig, 'add_layers', add_layers)
    encoder = DEBertModel.from_pretrained(config["bert_path"], config=bertconfig)
    model = GlobalPointer(encoder, ent_type_size, 64)
    model = model.to(device)
    if config['run_type'] != 'train':
        for filename in os.listdir(config['reload_path']):
            if filename.endswith('.pt'):
                reload_model_path = os.path.join(config['reload_path'], filename)
                model.load_state_dict(torch.load(reload_model_path), strict=False)
                test_f1, total_pred_line = valid(model, test_dataloader)
                with open(os.path.join(config['reload_path'], f'pred_4_{filename[:-3].split("_")[-1]}_{config["test_data"]}'), 'w', encoding='utf-8') as f:
                    for item in total_pred_line:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')