import torch
import torch.nn as nn
from pytorch_transformers import (WEIGHTS_NAME, BertTokenizer,BertModel, BertPreTrainedModel, BertConfig)
from transformers import BertLayer
import torch.autograd as autograd
import torch.nn.functional as F
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.linear_ctx = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.linear_query = nn.Linear(self.input_dim, self.input_dim, bias=True)
        self.v = nn.Linear(self.input_dim, 1)

    def forward(self, s_prev, enc_hs, src_mask):
        uh = self.linear_ctx(enc_hs)
        wq = self.linear_query(s_prev)
        wquh = torch.tanh(wq + uh)
        attn_weights = self.v(wquh).squeeze()
        attn_weights.data.masked_fill_(src_mask.data, -float('inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        ctx = torch.bmm(attn_weights.unsqueeze(1), enc_hs).squeeze()
        return ctx, attn_weights



class Decoder(nn.Module):  
    def __init__(self, input_dim, hidden_dim, layers, drop_out_rate, max_length,att_type,rel_size):
        super(Decoder, self).__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = layers
        self.drop_rate = int(drop_out_rate)
        self.max_length = int(max_length)

        if att_type == 0:
            self.attention = Attention(input_dim)
            self.lstm = nn.LSTMCell(10 * self.input_dim, self.hidden_dim)
        elif att_type == 1:
            self.attention = Attention(input_dim)
            self.lstm = nn.LSTMCell(10 * self.input_dim, self.hidden_dim)
        else:
            self.attention1 = Attention(input_dim)
            self.attention2 = Attention(input_dim)
            self.lstm = nn.LSTMCell(11 * self.input_dim, self.hidden_dim)

        self.e1_pointer_lstm = nn.LSTM(2 * self.input_dim, self.input_dim, 1, batch_first=True,
                                       bidirectional=True)
        self.e2_pointer_lstm = nn.LSTM(4 * self.input_dim, self.input_dim, 1, batch_first=True,
                                       bidirectional=True)

        self.entity1s_lin = nn.Linear(2 * self.input_dim, 1)
        self.entity1e_lin = nn.Linear(2 * self.input_dim, 1)
        self.entity2s_lin = nn.Linear(2 * self.input_dim, 1)
        self.entity2e_lin = nn.Linear(2 * self.input_dim, 1)
        self.rel_lin = nn.Linear(9 * self.input_dim, rel_size)
        self.dropout = nn.Dropout(self.drop_rate)
        self.w = nn.Linear(9 * self.input_dim, self.input_dim)

    def forward(self, y_prev, prev_tuples, h_prev, enc_hs, src_mask, entity1, entity2, entity1_mask, entity2_mask,att_type,
                is_training=False):
        src_time_steps = enc_hs.size()[1]
        if att_type == 0:
            ctx, attn_weights = self.attention(h_prev[0].squeeze().unsqueeze(1).repeat(1, src_time_steps, 1),
                                                enc_hs, src_mask)
        elif att_type == 1:
            reduce_prev_tuples = self.w(prev_tuples)
            ctx, attn_weights = self.attention(reduce_prev_tuples.unsqueeze(1).repeat(1, src_time_steps, 1),
                                                enc_hs, src_mask)
        else:
            ctx1, attn_weights1 = self.attention1(h_prev[0].squeeze().unsqueeze(1).repeat(1, src_time_steps, 1),
                                               enc_hs, src_mask)
            reduce_prev_tuples = self.w(prev_tuples)
            ctx2, attn_weights2 = self.attention2(reduce_prev_tuples.unsqueeze(1).repeat(1, src_time_steps, 1),
                                               enc_hs, src_mask)
            ctx = torch.cat((ctx1, ctx2), -1)
            attn_weights = (attn_weights1 + attn_weights2) / 2

        s_cur = torch.cat((prev_tuples, ctx), 1)
        hidden, cell_state = self.lstm(s_cur, h_prev)
        hidden = self.dropout(hidden)
        
        e1_pointer_lstm_input = torch.cat((enc_hs, hidden.unsqueeze(1).repeat(1, src_time_steps, 1)), 2)
        e1_pointer_lstm_out, phc = self.e1_pointer_lstm(e1_pointer_lstm_input)
        e1_pointer_lstm_out = self.dropout(e1_pointer_lstm_out)

        e2_pointer_lstm_input = torch.cat((e1_pointer_lstm_input, e1_pointer_lstm_out), 2)
        e2_pointer_lstm_out, phc = self.e2_pointer_lstm(e2_pointer_lstm_input)
        e2_pointer_lstm_out = self.dropout(e2_pointer_lstm_out)

        entity1s = self.entity1s_lin (e1_pointer_lstm_out).squeeze()
        entity1s.data.masked_fill_(src_mask.data, -float('inf'))

        entity1e = self.entity1e_lin (e1_pointer_lstm_out).squeeze()
        entity1e.data.masked_fill_(src_mask.data, -float('inf'))

        entity2s = self.entity2s_lin (e2_pointer_lstm_out).squeeze()
        entity2s.data.masked_fill_(src_mask.data, -float('inf'))

        entity2e = self.entity2e_lin (e2_pointer_lstm_out).squeeze()
        entity2e.data.masked_fill_(src_mask.data, -float('inf'))

        entity1sweights = F.softmax(entity1s, dim=-1)
        entity1eweights = F.softmax(entity1e, dim=-1)

        entity1sv = torch.bmm(entity1eweights.unsqueeze(1), e1_pointer_lstm_out).squeeze()
        entity1ev = torch.bmm(entity1sweights.unsqueeze(1), e1_pointer_lstm_out).squeeze()
        entity1 = self.dropout(torch.cat((entity1sv, entity1ev), -1))

        entity2sweights = F.softmax(entity2s, dim=-1)
        entity2eweights = F.softmax(entity2e, dim=-1)

        entity2sv = torch.bmm(entity2eweights.unsqueeze(1), e2_pointer_lstm_out).squeeze()
        entity2ev = torch.bmm(entity2sweights.unsqueeze(1), e2_pointer_lstm_out).squeeze()
        entity2 = self.dropout(torch.cat((entity2sv, entity2ev), -1))
        
        rel = self.rel_lin(torch.cat((hidden, entity1, entity2), -1))

        if is_training:
            entity1s = F.log_softmax(entity1s, dim=-1)
            entity1e = F.log_softmax(entity1e, dim=-1)
            entity2s = F.log_softmax(entity2s, dim=-1)
            entity2e = F.log_softmax(entity2e, dim=-1)
            rel = F.log_softmax(rel, dim=-1)

            return rel.unsqueeze(1), entity1s.unsqueeze(1), entity1e.unsqueeze(1), entity2s.unsqueeze(1), \
                entity2e.unsqueeze(1), (hidden, cell_state), entity1, entity2
        else:
            entity1s = F.softmax(entity1s, dim=-1)
            entity1e = F.softmax(entity1e, dim=-1)
            entity2s = F.softmax(entity2s, dim=-1)
            entity2e = F.softmax(entity2e, dim=-1)
            rel = F.softmax(rel, dim=-1)
            return rel.unsqueeze(1), entity1s.unsqueeze(1), entity1e.unsqueeze(1), entity2s.unsqueeze(1), entity2e.unsqueeze(1), \
                   (hidden, cell_state), entity1, entity2


class BERT_Seq2SeqModel(nn.Module):
    def __init__(self,bertconfig,config):
        super(BERT_Seq2SeqModel, self).__init__()
        self.encoder = BertModel.from_pretrained( config.model_path, config=bertconfig)
        self.num_labels = bertconfig.num_labels
        self.l2_reg_lambda = bertconfig.l2_reg_lambda
        self.dropout = nn.Dropout(bertconfig.hidden_dropout_prob)
        vocab_size=config.vocab_size
        self.ner_classifier=nn.Linear(config.enc_hidden_size, vocab_size)

        self.span_layer = BertLayer(config=bertconfig)
        self.w = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.gamma = nn.Parameter(torch.ones(1))

        dec_att_type=int(config.dec_att_type)
        self.rel_size=config.rel_size
        self.decoder = Decoder(config.dec_inp_size, config.dec_hidden_size, 1, config.drop_rate, config.max_trg_len,dec_att_type,self.rel_size)
        self.relation_embeddings = nn.Embedding(config.rel_size, config.dec_inp_size)
        self.dropout_di = nn.Dropout(config.drop_rate)  


    def forward(self, src_words_seq, src_mask, ori_src_words_mask, src_segment, trg_words_seq,  trg_rel_cnt, 
                entity1_mask, entity2_mask,  dec_hidden_size, att_type, input_span_mask, is_training=False):   
        if is_training: 
            trg_word_embeds = self.dropout_di(self.relation_embeddings(trg_words_seq))  
            self.encoder.train()
        else:
            self.encoder.eval()
        
        batch_len = src_words_seq.size()[0]
        src_time_steps = src_words_seq.size()[1]  
        time_steps = trg_rel_cnt
        
        outputs=self.encoder(src_words_seq,attention_mask=src_mask,token_type_ids=src_segment)
        pooled_output = outputs[1]
        enc_hs = outputs[0]

        ner_logits = self.ner_classifier(enc_hs)
        
        extended_span_attention_mask = input_span_mask.unsqueeze(1)
        extended_span_attention_mask = (1.0 - extended_span_attention_mask) * -10000.0
        span_sequence_output= self.span_layer(enc_hs, extended_span_attention_mask)
        w = F.softmax(self.w)
        enc_hs = self.gamma * (w[0] * enc_hs + w[1] * span_sequence_output[0])

        
        h0 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, dec_hidden_size))).cuda()
        c0 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, dec_hidden_size))).cuda()
        dec_hid = (h0, c0)

        dec_inp = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, dec_hidden_size))).cuda()
        entity1 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, 4 * dec_hidden_size))).cuda()
        entity2 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, 4 * dec_hidden_size))).cuda()

        prev_tuples = torch.cat((entity1, entity2, dec_inp), -1)
        if is_training:
            dec_outs = self.decoder(dec_inp, prev_tuples, dec_hid, enc_hs, ori_src_words_mask, entity1, entity2,
                                    entity1_mask[:, 0, :].squeeze(), entity2_mask[:, 0, :].squeeze(),att_type, is_training)
        else:
            dec_outs = self.decoder(dec_inp, prev_tuples, dec_hid, enc_hs, ori_src_words_mask, entity1, entity2, None, None,att_type,
                                    is_training)
        rel = dec_outs[0]
        entity1s = dec_outs[1]
        entity1e = dec_outs[2]
        entity2s = dec_outs[3]
        entity2e = dec_outs[4]
        dec_hid = dec_outs[5]
        entity1 = dec_outs[6]
        entity2 = dec_outs[7]

        topv, topi = rel[:, :, 1:].topk(1)
        topi = torch.add(topi, 1)

        for t in range(1, time_steps):
            if is_training:
                dec_inp = trg_word_embeds[:, t - 1, :].squeeze()
                prev_tuples = torch.cat((entity1, entity2, dec_inp), -1) + prev_tuples
                dec_outs = self.decoder(dec_inp, prev_tuples, dec_hid, enc_hs, ori_src_words_mask, entity1, entity2,
                                        entity1_mask[:, t, :].squeeze(), entity2_mask[:, t, :].squeeze(), att_type, is_training)
            else:
                dec_inp = self.relation_embeddings(topi.squeeze().detach()).squeeze()
                prev_tuples = torch.cat((entity1, entity2, dec_inp), -1) + prev_tuples
                dec_outs = self.decoder(dec_inp, prev_tuples, dec_hid, enc_hs, ori_src_words_mask, entity1, entity2, None, None,att_type, 
                                        is_training)

            cur_rel = dec_outs[0]
            cur_entity1s = dec_outs[1]
            cur_entity1e = dec_outs[2]
            cur_entity2s = dec_outs[3]
            cur_entity2e = dec_outs[4]
            dec_hid = dec_outs[5]
            entity1 = dec_outs[6]
            entity2 = dec_outs[7]

            rel = torch.cat((rel, cur_rel), 1)
            entity1s = torch.cat((entity1s, cur_entity1s), 1)
            entity1e = torch.cat((entity1e, cur_entity1e), 1)
            entity2s = torch.cat((entity2s, cur_entity2s), 1)
            entity2e = torch.cat((entity2e, cur_entity2e), 1)

            topv, topi = cur_rel[:, :, 1:].topk(1)
            topi = torch.add(topi, 1)
        if is_training:
            rel = rel.view(-1, self.rel_size)
            entity1s = entity1s.view(-1, src_time_steps)
            entity1e = entity1e.view(-1, src_time_steps)
            entity2s = entity2s.view(-1, src_time_steps)
            entity2e = entity2e.view(-1, src_time_steps)
        return rel, entity1s, entity1e, entity2s, entity2e, ner_logits

