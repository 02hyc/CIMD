# -*- coding: utf-8 -*-
'''
@author: Yongchang Cao
@contact: cyc990520@gmail.com
@file: debert_main.py
@time: 2022/3/22 16:06
@desc:
'''

'''
append_params:
    define_ids : [bat, seq]
    define_type_ids : [bat, seq]
    # define_positions = [[[1,2][5, 6]], [[6, 9], [-1, -1]]]     # [bat, rid, rnum]
    match_word_ids : [[0,0,1,1,0,0,2,2,0,0,0], [0,0,1,1,0,0,0,0,0,0,0]]
    
    
    define_encoder_max_layer : 0-12: int
    define_mix_mode: 'last', 'all', '[0, 3]'
    define_cls : [[0, 2, 3, 5], [0, 2]]
    
    # generate params:
    define_embedding_output : [bat, seq, dim]
    define_hidden_states: [bat, seq, dim]
'''

import torch
define_ids_embed = torch.range(1, 20).view(2, 10, 1).repeat(1, 1, 5)
print(define_ids_embed)     # [bat, seq, dim]

CLS_indexs_list = [[0, 2, 5], [0, 4]]
CLS_indexs_list_length = [len(CLS_indexs_item) for CLS_indexs_item in CLS_indexs_list]
max_CLS_index = max(CLS_indexs_list_length)
pad_define_embeds = torch.zeros([2, max_CLS_index+1, 5])

print('*' * 50)
print('select CLS embed:')
for i in range(2):
    print(define_ids_embed[i, CLS_indexs_list[i]])

for i in range(2):
    pad_define_embeds[i, 1:CLS_indexs_list_length[i]+1] = define_ids_embed[i, CLS_indexs_list[i]]
print('*' * 50)
print('select pad_define_embeds')
print(pad_define_embeds)

match_word_ids = [[0,0,1,1,0,0,2,2,0,0],
                  [0,0,0,0,1,1,0,0,0,0]]
match_word_ids = torch.tensor(match_word_ids).long()
match_word_embed = torch.zeros(2, 10, 5)
print('*' * 50)
print('match_word_embed')
for i in range(2):
    match_word_embed[i] = (torch.index_select(pad_define_embeds[i], dim=0, index=match_word_ids[i]))
print(match_word_embed.size(), match_word_embed)


