# -*- coding: utf-8 -*-
'''
@author: Yanchen Huang
@contact: yanchenhuang@smail.nju.edu.cn
@time: 2025/3/23
@desc:
'''

base_config = {
    'run_type': 'train',
    # 'reload_path':'/home/caoyc/ChID/DEBERT/cross_6_04-01_07:22:49',
    'test_data' : 'gaokao_v2_analysis.json',

    'path_to_save_model': '/home/yukino/Documents/Code/NLP/CIMD/DEBERT/models',
    "bert_path": "/home/yukino/Documents/Code/NLP/CIMD/ychuang/Pretrains/chinese-roberta-wwm-ext",
    'data_home': '/home/yukino/Documents/Code/NLP/CIMD/GlobalPointer/datasets/ChID/',
    'ent2id': 'ent2id.json',
    'train_data': 'train_big.json',             # 'train_big.json'
    'valid_ran_data': 'ran_v2_analysis.txt',    # 'ran_v2_analysis.txt',
    'valid_sim_data': 'sim_v2_analysis.txt',    # 'sim_v2_analysis.txt',
    'max_seq_len': 300,
    'batch_size': 8,
    'epochs': 10,
    'lr': 3e-5,
    'define_encoder_max_layer': 6,
    'define_mix_mode': 'all',          # last, all, cross
    'max_train_size': 500000,             # 60000

    'scheduler': 'Step',                # CAWR, CosW, Step
    'T_mult': 1,
    'rewarm_epoch_num': 1,
    'log_interval': 1000,
    "decay_rate": 0.99,
    "decay_steps": 100,
    'use_half': True,
    'use_cache':True,
    'strict': True,
}


# financial_config = {
#     'run_type': 'train',
#     # 'reload_path':'/home/caoyc/ChID/DEBERT/cross_6_04-01_07:22:49',
#     'test_data' : 'financial_ner_train.json',

#     'path_to_save_model': 'C:\Users\13414\Desktop\NLP\CIMD\DEBERT',
#     "bert_path": "C:\Users\13414\Desktop\NLP\CIMD\ychuang\Pretrains\chinese-roberta-wwm-ext",
#     'data_home': 'C:\Users\13414\Desktop\NLP\CIMD\GlobalPointer\datasets\DomainNER',
#     'ent2id': 'financial_ner_ent2id.json',
#     'train_data': 'financial_ner_train_v2.json',             # 'train_big.json',
#     'valid_ran_data': 'financial_ner_dev_v2.json',    # 'ran_v2_analysis.txt',
#     'valid_sim_data': 'financial_ner_test_v2.json',    # 'sim_v2_analysis.txt',
#     'max_seq_len': 128,
#     'batch_size': 32,
#     'epochs': 10,
#     'lr': 3e-5,
#     'define_encoder_max_layer': 12,
#     'define_mix_mode': 'all',              # last, all, cross
#     'max_train_size': 300000,             # 60000

#     'scheduler': 'CosW',                # CAWR, CosW, Step
#     'T_mult': 1,
#     'rewarm_epoch_num': 1,
#     'log_interval': 1000,
#     "decay_rate": 0.99,
#     "decay_steps": 100,
#     'use_half': True,
#     'use_cache':True,
#     'strict': True,
# }

# medical_config = {
#     'run_type': 'test',         # train  test
#     # 'reload_path':'/home/caoyc/ChID/DEBERT/all_12_09-28_07:02:02',
#     'test_data' : 'medical_ner_test_v2.json',

#     'path_to_save_model': 'C:\Users\13414\Desktop\NLP\CIMD\DEBERT',
#     "bert_path": "C:\Users\13414\Desktop\NLP\CIMD\ychuang\Pretrains\chinese-roberta-wwm-ext",
#     'data_home': 'C:\Users\13414\Desktop\NLP\CIMD\GlobalPointer\datasets\DomainNER',
#     'ent2id': 'medical_ner_ent2id.json',
#     'train_data': 'medical_ner_train_v2.json',             # 'train_big.json',
#     'valid_ran_data': 'medical_ner_dev_v2.json',    # 'ran_v2_analysis.txt',
#     'valid_sim_data': 'medical_ner_test_v2.json',    # 'sim_v2_analysis.txt',
#     'max_seq_len': 250,
#     'batch_size': 16,
#     'epochs': 20,
#     'lr': 3e-5,
#     'define_encoder_max_layer': 10,
#     'define_mix_mode': 'all',              # last, all, cross
#     'max_train_size': 300000,             # 60000

#     'scheduler': 'CosW',                # CAWR, CosW, Step
#     'T_mult': 1,
#     'rewarm_epoch_num': 1,
#     'log_interval': 1000,
#     "decay_rate": 0.99,
#     "decay_steps": 100,
#     'use_half': False,
#     'use_cache': False,
#     'strict': True,
# }


config = base_config

