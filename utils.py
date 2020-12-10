import sys
import os
import numpy as np
import random
from configparser import ConfigParser


from collections import OrderedDict
import pickle
import datetime
import json
from tqdm import tqdm
from recordclass import recordclass  
import math
from model import BERT_Seq2SeqModel
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.backends.cudnn.deterministic = True
n_gpu = torch.cuda.device_count()

def get_relations(file_name):  
    nameToIdx = OrderedDict()
    idxToName = OrderedDict()
    reader = open(file_name)
    lines = reader.readlines()
    reader.close()
    nameToIdx['<PAD>'] = 0
    idxToName[0] = '<PAD>'
    idx = 1
    for line in lines:
        nameToIdx[line.strip()] = idx
        idxToName[idx] = line.strip()
        idx += 1
    return nameToIdx, idxToName

rel_file = os.path.join('./', 'relations.txt')
relnameToIdx, relIdxToName = get_relations(rel_file)

def get_data(src_lines, trg_lines, datatype):
    samples = []
    uid = 1
    for i in range(0, len(src_lines)):
        src_line = src_lines[i].strip()
        trg_line = trg_lines[i].strip()
        src_words = src_line.split()

        trg_rels = []
        trg_pointers = []
        parts = trg_line.split('|')
        if datatype == 1:
            random.shuffle(parts)

        for part in parts:
            elements = part.strip().split()
            trg_rels.append(relnameToIdx[elements[4]])
            trg_pointers.append((int(elements[0]), int(elements[1]), int(elements[2]), int(elements[3])))

        if datatype == 1 and (len(src_words) > max_src_len or len(trg_rels) > max_trg_len):
            continue

        sample = Sample(Id=uid, SrcLen=len(src_words), SrcWords=src_words, TrgLen=len(trg_rels), TrgRels=trg_rels,
                        TrgPointers=trg_pointers) 
        samples.append(sample)
        uid += 1
    return samples


def get_model(model_id,bertconfig,config):
    if model_id == 1:
        return BERT_Seq2SeqModel(bertconfig,config)


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 1:
        torch.cuda.manual_seed_all(seed)


def shuffle_data(data,config):  
    batch_size=config.per_gpu_train_batch_size
    data.sort(key=lambda x: x.SrcLen)
    num_batch = int(len(data) / batch_size)
    rand_idx = random.sample(range(num_batch), num_batch)
    new_data = []
    for idx in rand_idx:
        new_data += data[batch_size * idx: batch_size * (idx + 1)]
    if len(new_data) < len(data):
        new_data += data[num_batch * batch_size:]
    return new_data


def get_max_len(sample_batch):  
    src_max_len = len(sample_batch[0].SrcWords)
    for idx in range(1, len(sample_batch)):
        if len(sample_batch[idx].SrcWords) > src_max_len:
            src_max_len = len(sample_batch[idx].SrcWords)

    trg_max_len = len(sample_batch[0].TrgRels)
    for idx in range(1, len(sample_batch)):
        if len(sample_batch[idx].TrgRels) > trg_max_len:
            trg_max_len = len(sample_batch[idx].TrgRels)

    return src_max_len, trg_max_len

def get_padded_pointers(pointers, pidx, max_len):
    idx_list = []
    for p in pointers:
        idx_list.append(p[pidx])
    pad_len = max_len + 1 - len(pointers)
    for i in range(0, pad_len):
        idx_list.append(-1)
    return idx_list

def get_padded_mask(cur_len, max_len):
    mask_seq = list()
    for i in range(0, cur_len):
        mask_seq.append(0)
    pad_len = max_len - cur_len
    for i in range(0, pad_len):
        mask_seq.append(1)
    return mask_seq


def get_padded_relations(rels, max_len):
    rel_list = []
    for r in rels:
        rel_list.append(r)
    rel_list.append(relnameToIdx['NA'])
    pad_len = max_len + 1 - len(rel_list)
    for i in range(0, pad_len):
        rel_list.append(relnameToIdx['<PAD>'])
    return rel_list


def get_relation_index_seq(rel_ids, max_len):
    seq = list()
    for r in rel_ids:
        seq.append(r)
    seq.append(relnameToIdx['NA'])
    pad_len = max_len + 1 - len(seq)
    for i in range(0, pad_len):
        seq.append(relnameToIdx['<PAD>'])
    return seq


def get_entity_masks(pointers, src_max, trg_max):
    arg1_masks = []
    arg2_masks = []
    for p in pointers:
        arg1_mask = [1 for i in range(src_max)]
        arg1_mask[p[0]] = 0
        arg1_mask[p[1]] = 0

        arg2_mask = [1 for i in range(src_max)]
        arg2_mask[p[2]] = 0
        arg2_mask[p[3]] = 0

        arg1_masks.append(arg1_mask)
        arg2_masks.append(arg2_mask)

    pad_len = trg_max + 1 -len(pointers)
    for i in range(0, pad_len):
        arg1_mask = [1 for i in range(src_max)]
        arg2_mask = [1 for i in range(src_max)]
        arg1_masks.append(arg1_mask)
        arg2_masks.append(arg2_mask)
    return arg1_masks, arg2_masks


def get_span_pos(spanTag,max_seq_length):
    span_pos = np.zeros((max_seq_length, max_seq_length), dtype=int)
    for i in range(0,len(spanTag),2):
        span_pos[int(spanTag[i])+1][int(spanTag[i+1])+1]=1
    return span_pos

def get_batch_data(cur_samples, is_training=False):
    """
    Returns the training samples and labels as numpy array
    """
    batch_src_max_len, batch_trg_max_len = get_max_len(cur_samples)
    batch_trg_max_len += 1
    src_words_list = list()
    src_words_mask_list = list()
    ori_src_words_mask_list = list()
    src_seg_list=list()
    decoder_input_list = list()
    ner_list = []

    rel_seq = list()
    arg1_start_seq = list()
    arg1_end_seq = list()
    arg2_start_seq = list()
    arg2_end_seq = list()
    arg1_mask_seq = []
    arg2_mask_seq = []
    ner_feature_list = []

    for sample in cur_samples:
        src_words_list.append(sample.input_ids)
        src_words_mask_list.append(sample.input_mask)
        src_seg_list.append(sample.segment_ids)
        ner_list.append(sample.nerTag)
        ori_src_words_mask_list.append(get_padded_mask(sample.SrcLen, len(sample.input_ids)))
        ner_feature_list.append(get_span_pos(sample.spanTag,len(sample.input_ids)))
        if is_training:
            arg1_start_seq.append(get_padded_pointers(sample.TrgPointers, 0, batch_trg_max_len))
            arg1_end_seq.append(get_padded_pointers(sample.TrgPointers, 1, batch_trg_max_len))
            arg2_start_seq.append(get_padded_pointers(sample.TrgPointers, 2, batch_trg_max_len))
            arg2_end_seq.append(get_padded_pointers(sample.TrgPointers, 3, batch_trg_max_len))
            rel_seq.append(get_padded_relations(sample.TrgRels, batch_trg_max_len))
            decoder_input_list.append(get_relation_index_seq(sample.TrgRels, batch_trg_max_len))
            arg1_mask, arg2_mask = get_entity_masks(sample.TrgPointers, batch_src_max_len, batch_trg_max_len)
            arg1_mask_seq.append(arg1_mask)
            arg2_mask_seq.append(arg2_mask)
            
        else:
            decoder_input_list.append(get_relation_index_seq([], 1))

    return {'src_words': np.array(src_words_list, dtype=np.float32),
            'src_words_mask': np.array(src_words_mask_list),
            'ori_src_words_mask':np.array(ori_src_words_mask_list),
            'nerTag':np.array(ner_list),
            'src_segment':np.array(src_seg_list),
            'decoder_input': np.array(decoder_input_list),
            'rel': np.array(rel_seq),
            'arg1_start':np.array(arg1_start_seq),
            'arg1_end': np.array(arg1_end_seq),
            'arg2_start': np.array(arg2_start_seq),
            'arg2_end': np.array(arg2_end_seq),
            'arg1_mask': np.array(arg1_mask_seq),
            'arg2_mask': np.array(arg2_mask_seq),
            'ner_feature':np.array(ner_feature_list)}

def is_full_match(triplet, triplets): 
    for t in triplets:
        if t[0] == triplet[0] and t[1] == triplet[1] and t[2] == triplet[2]:
            return True
    return False


def get_answer_pointers(arg1start_preds, arg1end_preds, arg2start_preds, arg2end_preds, sent_len):
    arg1_prob = -1.0
    arg1start = -1
    arg1end = -1
    max_ent_len = 5
    for i in range(0, sent_len):
        for j in range(i, min(sent_len, i + max_ent_len)):
            if arg1start_preds[i] * arg1end_preds[j] > arg1_prob:
                arg1_prob = arg1start_preds[i] * arg1end_preds[j]
                arg1start = i
                arg1end = j

    arg2_prob = -1.0
    arg2start = -1
    arg2end = -1
    for i in range(0, arg1start):
        for j in range(i, min(arg1start, i + max_ent_len)):
            if arg2start_preds[i] * arg2end_preds[j] > arg2_prob:
                arg2_prob = arg2start_preds[i] * arg2end_preds[j]
                arg2start = i
                arg2end = j
    for i in range(arg1end + 1, sent_len):
        for j in range(i, min(sent_len, i + max_ent_len)):
            if arg2start_preds[i] * arg2end_preds[j] > arg2_prob:
                arg2_prob = arg2start_preds[i] * arg2end_preds[j]
                arg2start = i
                arg2end = j

    arg2_prob1 = -1.0
    arg2start1 = -1
    arg2end1 = -1
    for i in range(0, sent_len):
        for j in range(i, min(sent_len, i + max_ent_len)):
            if arg2start_preds[i] * arg2end_preds[j] > arg2_prob1:
                arg2_prob1 = arg2start_preds[i] * arg2end_preds[j]
                arg2start1 = i
                arg2end1 = j

    arg1_prob1 = -1.0
    arg1start1 = -1
    arg1end1 = -1
    for i in range(0, arg2start1):
        for j in range(i, min(arg2start1, i + max_ent_len)):
            if arg1start_preds[i] * arg1end_preds[j] > arg1_prob1:
                arg1_prob1 = arg1start_preds[i] * arg1end_preds[j]
                arg1start1 = i
                arg1end1 = j
    for i in range(arg2end1 + 1, sent_len):
        for j in range(i, min(sent_len, i + max_ent_len)):
            if arg1start_preds[i] * arg1end_preds[j] > arg1_prob1:
                arg1_prob1 = arg1start_preds[i] * arg1end_preds[j]
                arg1start1 = i
                arg1end1 = j
    if arg1_prob * arg2_prob > arg1_prob1 * arg2_prob1:
        return arg1start, arg1end, arg2start, arg2end
    else:
        return arg1start1, arg1end1, arg2start1, arg2end1


def get_gt_triples(src_words, rels, pointers):  
    triples = []
    i = 0
    for r in rels:
        arg1 = ''.join(src_words[pointers[i][0]:pointers[i][1] + 1])
        arg2 = ''.join(src_words[pointers[i][2]:pointers[i][3] + 1])
        triplet = (arg1.strip(), arg2.strip(), relIdxToName[r])
        if not is_full_match(triplet, triples):
            triples.append(triplet)
        i += 1
    return triples


def get_pred_triples(rel, arg1s, arg1e, arg2s, arg2e, src_words):  
    triples = []
    all_triples = []
    
    for i in range(0, len(rel)):
        r = np.argmax(rel[i][1:]) + 1
        if r == relnameToIdx['NA']:
            break
        s1, e1, s2, e2 = get_answer_pointers(arg1s[i], arg1e[i], arg2s[i], arg2e[i], len(src_words))
        arg1 = ''.join(src_words[s1: e1 + 1 ])  
        arg2 = ''.join(src_words[s2: e2 + 1])
        arg1 = arg1.strip()
        arg2 = arg2.strip()
        if arg1 == arg2:  
            continue
        triplet = (arg1, arg2, relIdxToName[r])
        all_triples.append(triplet)
        if not is_full_match(triplet, triples): 
            triples.append(triplet)
    return triples, all_triples  


def get_F1(data, preds): 
    gt_pos = 0
    pred_pos = 0
    total_pred_pos = 0
    correct_pos = 0
    
    for i in range(0, len(data)):
        gt_triples = get_gt_triples(data[i].SrcWords, data[i].TrgRels, data[i].TrgPointers)
        pred_triples, all_pred_triples = get_pred_triples(preds[0][i], preds[1][i], preds[2][i], preds[3][i],
                                                          preds[4][i], data[i].SrcWords)
        total_pred_pos += len(all_pred_triples)
        gt_pos += len(gt_triples)
        pred_pos += len(pred_triples)
        for gt_triple in gt_triples:
            if is_full_match(gt_triple, pred_triples):
                correct_pos += 1
    return pred_pos, gt_pos, correct_pos

def write_test_res(data, preds, outfile):  
    writer = open(outfile, 'w')
    for i in range(0, len(data)):
        pred_triples, _ = get_pred_triples(preds[0][i], preds[1][i], preds[2][i], preds[3][i], preds[4][i],
                                        data[i].SrcWords)
        pred_triples_str = []
        for pt in pred_triples:
            pred_triples_str.append(pt[0] + ' ; ' + pt[1] + ' ; ' + pt[2])
        writer.write(' | '.join(pred_triples_str) + '\n')
    writer.close()


class Config(ConfigParser):
    def __init__(self, config_file):
        raw_config = ConfigParser()
        raw_config.read(config_file)
        self.cast_values(raw_config)
        

    def cast_values(self, raw_config):
        for section in raw_config.sections():
            for key, value in raw_config.items(section):
                val = None
                
                if type(value) is str and value.startswith("[") and value.endswith("]"):
                    val = eval(value)
                    setattr(self, key, val)
                    continue
                for attr in ["getint", "getfloat", "getboolean"]:
                    try:
                        val = getattr(raw_config[section], attr)(key)
                        break
                    except:
                        val = value
                    
                setattr(self, key, val)

