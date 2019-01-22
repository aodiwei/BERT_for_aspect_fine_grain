#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '18-11-22'
# 
"""
import os
import eval_pred

cmd = """python3 run_classifier.py --task_name=aspect_fine_grain
--do_train={}
--do_eval={}
--do_predict={}
--data_dir=/home/adw/workspace/study/dataset/aspect_bert
--vocab_file=/home/adw/workspace/ref/chinese_L-12_H-768_A-12/vocab.txt
--bert_config_file=/home/adw/workspace/ref/chinese_L-12_H-768_A-12/bert_config.json
--init_checkpoint=/home/adw/workspace/ref/chinese_L-12_H-768_A-12/bert_model.ckpt
--max_seq_length=512
--train_batch_size=6
--learning_rate=2e-5
--num_train_epochs={}
--output_dir=/tmp/aspect_output/"""

# cmd_e = cmd.format('false', 'false', 'true', 1)
# cmd_e = ' '.join(cmd_e.split('\n'))
# print('starting train epoch {}'.format(1))
# os.system(cmd_e)

max = 0
for i in range(1, 11):
    cmd_e = cmd.format('true', 'false', 'true', i)
    cmd_e = ' '.join(cmd_e.split('\n'))
    print('starting train epoch {}'.format(i))
    os.system(cmd_e)
    ret = eval_pred.eval()
    if ret > max:
        print('better result {} at epoch {}'.format(ret, i))
        max = ret




