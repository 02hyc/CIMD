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
    'max_seq_len': 450,
    'batch_size': 16,
    'epochs': 10,
    'lr': 2e-5,                        # 降低初始学习率以提高稳定性
    'define_encoder_max_layer': 6,
    'define_mix_mode': 'all',          # last, all, cross
    'max_train_size': 300000,          # 减少训练集大小，避免过拟合
    
    # 修改学习率调度策略
    'scheduler': 'CosW',               # 改用简单的余弦退火，不使用重启
    'T_mult': 1,
    'rewarm_epoch_num': 1,
    'log_interval': 500,               # 增加日志频率以便更好地监控
    "decay_rate": 0.98,                # 调整衰减率
    "decay_steps": 200,                # 增加衰减步长
    
    # 添加正则化参数
    'weight_decay': 0.01,              # 添加权重衰减（L2正则化）
    'dropout_rate': 0.1,               # 添加dropout率
    'gradient_clip_val': 1.0,          # 添加梯度裁剪值
    
    'use_half': False,                 # 关闭半精度训练，提高数值稳定性
    'use_cache': True,
    'strict': True,
    
    # 添加早停策略
    'early_stopping': True,            # 启用早停
    'patience': 3,                     # 连续3个epoch验证集性能不提升则停止
    'min_delta': 0.001,                # 最小改进阈值
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
