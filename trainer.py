# -*- coding: utf-8 -*-
'''
@author: Yongchang Cao
@contact: cyc990520@gmail.com
@file: trainer.py
@time: 2022/3/23 21:53
@desc:
'''
import os
gpuid = '0'
print(gpuid)
os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

from configs import config

if config['run_type'] == 'test':
    config['define_mix_mode']='cross'
    config['define_encoder_max_layer']=6

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
# wait_time = 5
# i = 1
# while True and config['run_type']=='train':
#     i = i % 100
#     symbol = 'monitoring: ' + '>' * i + ' ' * (100 - i - 1) + '|'
#     if not busy_gpu(5000) and i > 1:
#         time.sleep(wait_time)
#     if not busy_gpu(5000): break
#     sys.stdout.write('\r' + symbol)
#     sys.stdout.flush()
#     time.sleep(wait_time)
#     i += 1

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
# seed = 2321
torch.manual_seed(seed)  # pytorch random seed
torch.backends.cudnn.deterministic = True

model_state_dict_dir = os.path.join(config["path_to_save_model"],  config['define_mix_mode'] +'_'+ str(config['define_encoder_max_layer']) +'_'+ time.strftime("%m-%d_%H:%M:%S",time.gmtime()))
if config["run_type"] == "train" and not os.path.exists(model_state_dict_dir):
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)
    sys.stdout = Logger(os.path.join(model_state_dict_dir, 'syslog.txt'), sys.stdout)

    print(model_state_dict_dir)
    print(f'train params : {config}')
    print(f'Set global seed to {seed}')
    print('DEBERT训练程序, 使用可训练alpha')

tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], do_lower_case=False)
ent2id_path = os.path.join(config["data_home"], config["ent2id"])
ent2id = load_data(ent2id_path, "ent2id")
ent_type_size = len(ent2id)

def data_generator(run_type='train', cache_path='/home/yukino/Documents/Code/NLP/CIMD/DEBERT/data_cache.pt', use_cache=True):
    define_type = 'word' if 'ner' in config['ent2id'] else 'idiom'
    data_maker = DataMaker(tokenizer, define_type=define_type)
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

    if use_cache and os.path.exists(cache_path):
        data = torch.load(cache_path)
        train_inputs, valid_inputs, test_inputs = data[:]
    else:
        train_data_path = os.path.join(config["data_home"], config["train_data"])  ## train_data
        train_data = load_data(train_data_path, "train")
        valid_data_path_ran = os.path.join(config["data_home"], config["valid_ran_data"])  ## test_data
        valid_data_ran = load_data(valid_data_path_ran, "valid")
        valid_data_path_sim = os.path.join(config["data_home"], config["valid_sim_data"])
        valid_data_sim = load_data(valid_data_path_sim, "valid")

        max_seq_len = config['max_seq_len']

        if config['max_train_size']:
            train_data = train_data[:config['max_train_size']]

        train_inputs = data_maker.generate_inputs(train_data, max_seq_len, ent2id)
        valid_inputs = data_maker.generate_inputs(valid_data_ran, max_seq_len, ent2id)
        test_inputs = data_maker.generate_inputs(valid_data_sim, max_seq_len, ent2id)
        if use_cache:
            torch.save([train_inputs, valid_inputs, test_inputs], cache_path)

    if config['max_train_size']:
        train_inputs = train_inputs[:config['max_train_size']]

    train_dataloader = DataLoader(MyDataset(train_inputs),
                                  batch_size=config["batch_size"],
                                  shuffle=False,
                                  num_workers=config["num_workers"],
                                  drop_last=False,
                                  collate_fn=data_maker.generate_batch
                                  )
    valid_dataloader = DataLoader(MyDataset(valid_inputs),
                                  batch_size=config["batch_size"],
                                  shuffle=False,
                                  num_workers=config["num_workers"],
                                  drop_last=False,
                                  collate_fn=data_maker.generate_batch
                                  )
    test_dataloader = DataLoader(MyDataset(test_inputs),
                                  batch_size=config["batch_size"],
                                  shuffle=False,
                                  num_workers=config["num_workers"],
                                  drop_last=False,
                                  collate_fn=data_maker.generate_batch
                                  )

    return train_dataloader, valid_dataloader, test_dataloader

metrics = MetricsCalculator()

def train_step(batch_train, model, optimizer, criterion, scaler, get_metric=False):
    batch_data = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch_train]
    input_ids, attention_mask, token_type_ids, define_ids, define_type_ids, define_cls, match_word_ids, labels = batch_data

    append_params = {
        'define_ids': define_ids,
        'define_type_ids': define_type_ids,
        'match_word_ids': match_word_ids,
        'define_cls': define_cls,
        'define_encoder_max_layer': config['define_encoder_max_layer'],
        'define_mix_mode': config['define_mix_mode'],
    }
    if config['use_half']:
        with autocast():
            logits = model(input_ids, attention_mask, token_type_ids, append_params=append_params)
    else:
        logits = model(input_ids, attention_mask, token_type_ids, append_params=append_params)

    if logits.size(1) == 1:
        new_batch_label = torch.zeros([labels.size()[0], 1, logits.size()[2], logits.size()[2]], device=labels.device)
        for batid in range(labels.size()[0]):
            for labelid in range(labels.size()[1]):
                if labels[batid, labelid, 0] == -1 or labels[batid, labelid, 2] >= config['max_seq_len']:
                    break
                new_batch_label[batid, 0, labels[batid, labelid, 1].item(), labels[batid, labelid, 2].item()] = 1
    else:
        new_batch_label = torch.zeros([labels.size()[0], logits.size(1), logits.size()[2], logits.size()[2]], device=labels.device)
        for batid in range(labels.size(0)):
            for labelid in range(labels.size(1)):
                if labels[batid, labelid, 0] == -1 or labels[batid, labelid, 2] >= config['max_seq_len']:
                    break
                new_batch_label[batid, labels[batid, labelid, 0].item(), labels[batid, labelid, 1].item(), labels[batid, labelid, 2].item()] = 1

    batch_labels = new_batch_label
    loss = criterion(logits, batch_labels)  # batch_label ：[20, 1, 313, 313], 元素全0， 部分为1
    optimizer.zero_grad()

    if config['use_half']:
        scaler.scale(loss).backward()  # 为了梯度放大
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
    else:
        loss.backward()
        optimizer.step()

    # if config['strict']:
    #     batch_labels = []
    #     for bat in range(labels.size()[0]):
    #         for idx in range(labels.size()[1]):
    #             if labels[bat, idx, 0] == -1:
    #                 break
    #             batch_labels.append((bat, 0, labels[bat, idx, 1].item(), labels[bat, idx, 2].item()))

    if get_metric:
        sample_precision = metrics.get_sample_precision(logits, batch_labels)
        sample_f1 = metrics.get_sample_f1(logits, batch_labels)
        return loss.item(), sample_precision.item(), sample_f1.item()
    else:
        return loss.item(), 0.0, 0.0

def train(model, dataloader, epoch, optimizer, scheduler, scaler):
    model.train()

    # loss func
    def loss_fun(y_true, y_pred):
        """
        y_true:(batch_size, ent_type_size, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size, seq_len, seq_len)
        """
        batch_size, ent_type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
        loss = multilabel_categorical_crossentropy(y_true, y_pred)
        return loss

    total_loss, total_precision, total_f1 = 0., 0., 0.
    train_start_time = time.time()

    for batch_ind, batch_data in enumerate(dataloader):

        loss, precision, f1 = train_step(batch_data, model, optimizer, loss_fun, scaler, get_metric=batch_ind % config["log_interval"]==0)

        total_loss += loss
        total_precision += precision
        total_f1 += f1

        avg_loss = total_loss / (batch_ind + 1)
        scheduler.step()
        if config['use_half']:
            scaler.update()

        avg_precision = total_precision / (batch_ind + 1)
        avg_f1 = total_f1 / (batch_ind + 1)

        if batch_ind % config["log_interval"] == 0:
            print(f'{time.strftime("%H:%M:%S",time.gmtime())} Epoch: {epoch + 1}/{config["epochs"]}, Batch: {batch_ind + 1}/{len(dataloader)}, loss: {avg_loss}, precision: {avg_precision}, f1:{avg_f1}, lr: {optimizer.param_groups[0]["lr"]}')
    train_end_time = time.time()
    print('train epoch time : ')
    print(train_end_time-train_start_time)

def valid_step(batch_valid, model):
    batch_data = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch_valid]
    input_ids, attention_mask, token_type_ids, define_ids, define_type_ids, define_cls, match_word_ids, labels = batch_data

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

    if logits.size(1) == 1:
        new_batch_label = torch.zeros([labels.size()[0], 1, logits.size()[2], logits.size()[2]], device=labels.device)
        for batid in range(labels.size()[0]):
            for labelid in range(labels.size()[1]):
                if labels[batid, labelid, 0] == -1 or labels[batid, labelid, 2] >= config['max_seq_len']:
                    break
                new_batch_label[batid, 0, labels[batid, labelid, 1].item(), labels[batid, labelid, 2].item()] = 1
    else:
        new_batch_label = torch.zeros([labels.size()[0], logits.size(1), logits.size()[2], logits.size()[2]], device=labels.device)
        for batid in range(labels.size(0)):
            for labelid in range(labels.size(1)):
                if labels[batid, labelid, 0] == -1 or labels[batid, labelid, 2] >= config['max_seq_len']:
                    break
                new_batch_label[batid, labels[batid, labelid, 0].item(), labels[batid, labelid, 1].item(), labels[batid, labelid, 2].item()] = 1
    batch_labels = new_batch_label

    if config['strict']:
        batch_labels = []
        for bat in range(labels.size()[0]):
            for idx in range(labels.size()[1]):
                if labels[bat, idx, 0] == -1:
                    break
                batch_labels.append((bat, labels[bat, idx, 0].item(), labels[bat, idx, 1].item(), labels[bat, idx, 2].item()))

    if config['run_type'] == 'test' and 'medical' in config['test_data']:
        ori_text = [tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in input_ids[:]]
    else:
        ori_text = None
    sample_f1, sample_precision, sample_recall = metrics.get_evaluate_fpr(logits, batch_labels, attention_mask, text=ori_text)
    return sample_f1, sample_precision, sample_recall

def valid(model, dataloader):
    model.eval()
    total_X, total_Y, total_Z = 0, 0, 0

    eval_start_time = time.time()
    for batch_data in tqdm(dataloader, desc="Validating"):
        X, Y, Z = valid_step(batch_data, model)
        total_X += X
        total_Y += Y
        total_Z += Z
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
    return avg_f1

if __name__ == '__main__':
    if 'ner' in config['ent2id']:
        cache_path = f'/home/yukino/Documents/Code/NLP/CIMD/DEBERT/{config["ent2id"].split("_")[0]}_ner_data_cache.pt'
    else:
        cache_path = '/home/yukino/Documents/Code/NLP/CIMD/DEBERT/data_cache.pt'

    if config['run_type'] != 'train':
        test_dataloader = data_generator(run_type='test', cache_path=cache_path)

    else:
        train_dataloader, valid_dataloader_ran, valid_dataloader_sim = data_generator(use_cache=config['use_cache'], cache_path=cache_path)

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
                test_f1 = valid(model, test_dataloader)
        exit(0)

    # optimizer
    init_learning_rate = float(config["lr"])
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)

    # scheduler
    if config["scheduler"] == "CAWR":
        T_mult = config["T_mult"]
        rewarm_epoch_num = config["rewarm_epoch_num"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         len(train_dataloader) * rewarm_epoch_num,  # 重置学习率的周期
                                                                         T_mult)  # 是否重置后延长周期
    elif config["scheduler"] == "Step":
        decay_rate = config["decay_rate"]
        decay_steps = config["decay_steps"]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    elif config['scheduler'] == 'CosW':
        import math
        total_train_steps, num_warmup_steps = len(train_dataloader) * config['epochs'], int(len(train_dataloader) * config['epochs'] * 0.02)
        print(f'total_train_steps = {total_train_steps}, num_warmup_steps = {num_warmup_steps}')
        total_train_steps = len(train_dataloader) * (config['epochs'] + 1)
        # 0.5 * (cos(0-pi) + 1)
        warm_up_with_cosine_lr = lambda epoch: epoch / num_warmup_steps if epoch <= num_warmup_steps else 0.5 * (
                math.cos((epoch - num_warmup_steps) / (total_train_steps - num_warmup_steps) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    scaler = GradScaler()  # 训练前实例化一个GradScaler对象
    # max_sim_f1, max_ran_f1 = valid(model, valid_dataloader_sim), valid(model, valid_dataloader_ran)
    max_sim_f1, max_ran_f1 = 0, 0
    for epoch in range(config["epochs"]):
        train(model, train_dataloader, epoch, optimizer, scheduler, scaler)

        valid_f1_ran = valid(model, valid_dataloader_ran)
        if valid_f1_ran > max_ran_f1:
            max_ran_f1 = valid_f1_ran
            if valid_f1_ran > 0.88:  # save the best model
                # modle_state_num = len(glob.glob(model_state_dict_dir + "/model_state_dict_*.pt"))
                modle_state_num = 'ran'
                # torch.save(model.state_dict(), os.path.join(model_state_dict_dir, "model_state_dict_{}.pt".format(modle_state_num)))
            print(f"Best ran F1: {max_ran_f1}")
            print("******************************************")

        valid_f1_sim = valid(model, valid_dataloader_sim)
        if valid_f1_sim > max_sim_f1:
            max_sim_f1 = valid_f1_sim
            if valid_f1_sim > 0.78:  # save the best model
                # modle_state_num = len(glob.glob(model_state_dict_dir + "/model_state_dict_*.pt"))
                modle_state_num = 'sim'
                torch.save(model.state_dict(), os.path.join(model_state_dict_dir, "model_state_dict_{}.pt".format(modle_state_num)))
            print(f"Best sim F1: {max_sim_f1}")
            print("******************************************")