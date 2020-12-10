from utils import *
from data_load import *
from pytorch_transformers import (WEIGHTS_NAME, BertTokenizer,BertModel, BertPreTrainedModel, BertConfig)
from pytorch_transformers import AdamW, WarmupLinearSchedule
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
def custom_print(*msg):  
    for i in range(0, len(msg)):
        if i == len(msg) - 1:
            print(msg[i])
            logger.write(str(msg[i]) + '\n')  
        else:
            print(msg[i], ' ', end='')
            logger.write(str(msg[i]))


def train_model(model_id, train_samples, dev_samples, best_model_file,bertconfig,config):
    train_size = len(train_samples)
    batch_size=config.per_gpu_train_batch_size
    
    batch_count = int(math.ceil(train_size/batch_size))
    move_last_batch = False
    if len(train_samples) - batch_size * (batch_count - 1) == 1:
        move_last_batch = True
        batch_count -= 1
    
    
    model = get_model(model_id,bertconfig,config)  
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    custom_print('Parameters size:', pytorch_total_params)

    if torch.cuda.is_available():
        model.cuda()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model,device_ids=[0])
    

    rel_criterion = nn.NLLLoss(ignore_index=0)
    entity_criterion = nn.NLLLoss(ignore_index=-1)
    ner_criterion = nn.CrossEntropyLoss(ignore_index=0)

    wf = 1.0
    wn = 1.0
    custom_print('weight factor:', wf, 'weight ner:', wn)


    if config.max_steps > 0:
        t_total = config.max_steps
        config.num_train_epochs = config.max_steps // (
            batch_count // config.gradient_accumulation_steps) + 1
    else:
        t_total = batch_count  // config.gradient_accumulation_steps * config.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=config.learning_rate, eps=config.adam_epsilon)
    custom_print(optimizer)

    best_dev_acc = -1.0
    best_epoch_idx = -1
    best_epoch_seed = -1
    for epoch_idx in range(0, config.num_train_epochs):
        
        custom_print('Epoch:', epoch_idx + 1)
        cur_seed = random_seed + epoch_idx + 1

        set_random_seeds(cur_seed)
        cur_shuffled_train_data = shuffle_data(train_samples,config)
        start_time = datetime.datetime.now()
        train_loss_val = 0.0

        for batch_idx in range(0, batch_count):
            model.train()
            model.zero_grad()
            batch_start = batch_idx * batch_size
            batch_end = min(len(cur_shuffled_train_data), batch_start + batch_size)
            if batch_idx == batch_count - 1 and move_last_batch:
                batch_end = len(cur_shuffled_train_data)

            cur_batch = cur_shuffled_train_data[batch_start:batch_end]
            cur_samples_input = get_batch_data(cur_batch, True)

            src_words_seq = torch.from_numpy(cur_samples_input['src_words'].astype('long'))
            src_segment=torch.from_numpy(cur_samples_input['src_segment'].astype('long'))
            src_words_mask = torch.from_numpy(cur_samples_input['src_words_mask'].astype('long'))
            ori_src_words_mask=torch.from_numpy(cur_samples_input['ori_src_words_mask'].astype('bool'))
            trg_words_seq = torch.from_numpy(cur_samples_input['decoder_input'].astype('long'))
            ner=torch.from_numpy(cur_samples_input['nerTag'].astype('long'))

            rel = torch.from_numpy(cur_samples_input['rel'].astype('long'))
            arg1s = torch.from_numpy(cur_samples_input['arg1_start'].astype('long'))
            arg1e = torch.from_numpy(cur_samples_input['arg1_end'].astype('long'))
            arg2s = torch.from_numpy(cur_samples_input['arg2_start'].astype('long'))
            arg2e = torch.from_numpy(cur_samples_input['arg2_end'].astype('long'))
            arg1_mask = torch.from_numpy(cur_samples_input['arg1_mask'].astype('bool'))
            arg2_mask = torch.from_numpy(cur_samples_input['arg2_mask'].astype('bool'))

            input_span_mask=torch.from_numpy(cur_samples_input['ner_feature'].astype('long'))

            if torch.cuda.is_available():
                src_words_seq = src_words_seq.cuda()
                src_words_mask = src_words_mask.cuda()
                ori_src_words_mask = ori_src_words_mask.cuda()
                src_segment=src_segment.cuda()
                trg_words_seq = trg_words_seq.cuda()
                ner = ner.cuda()

                rel = rel.cuda()
                arg1s = arg1s.cuda()
                arg1e = arg1e.cuda()
                arg2s = arg2s.cuda()
                arg2e = arg2e.cuda()

                arg1_mask = arg1_mask.cuda()
                arg2_mask = arg2_mask.cuda()

                input_span_mask=input_span_mask.cuda()

            src_words_seq = autograd.Variable(src_words_seq)
            src_words_mask = autograd.Variable(src_words_mask)
            ori_src_words_mask = autograd.Variable(ori_src_words_mask)
            src_segment = autograd.Variable(src_segment)
            trg_words_seq = autograd.Variable(trg_words_seq)
            ner=autograd.Variable(ner)

            rel = autograd.Variable(rel)
            arg1s = autograd.Variable(arg1s)
            arg1e = autograd.Variable(arg1e)
            arg2s = autograd.Variable(arg2s)
            arg2e = autograd.Variable(arg2e)

            arg1_mask = autograd.Variable(arg1_mask)
            arg2_mask = autograd.Variable(arg2_mask)
            input_span_mask=autograd.Variable(input_span_mask)

            if model_id == 1:
                outputs = model(src_words_seq, src_words_mask, ori_src_words_mask, src_segment, trg_words_seq,
                                rel.size()[1],  arg1_mask, arg2_mask, config.dec_hidden_size, config.dec_att_type, input_span_mask,True) #positional_seq,  src_chars_seq,  adj,
            ner_logit=outputs[5].view(-1, outputs[5].shape[-1])
            rel = rel.view(-1, 1).squeeze()
            ner=ner. view(-1).squeeze()
            arg1s = arg1s.view(-1, 1).squeeze()
            arg1e = arg1e.view(-1, 1).squeeze()
            arg2s = arg2s.view(-1, 1).squeeze()
            arg2e = arg2e.view(-1, 1).squeeze()
            loss = rel_criterion(outputs[0], rel) + \
                   wn * ner_criterion(ner_logit,ner)+\
                   wf * (entity_criterion(outputs[1], arg1s) + entity_criterion(outputs[2], arg1e)) + \
                   wf * (entity_criterion(outputs[3], arg2s) + entity_criterion(outputs[4], arg2e))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if (batch_idx + 1) % config.update_freq == 0:
                optimizer.step()
                model.zero_grad()
            train_loss_val += loss.item()

        train_loss_val /= batch_count
        end_time = datetime.datetime.now()
        custom_print('Training loss:', train_loss_val)
        custom_print('Training time:', end_time - start_time)

        custom_print('\nDev Results\n')
        set_random_seeds(random_seed)
        dev_preds = predict(dev_samples, model, model_id,config)

        pred_pos, gt_pos, correct_pos = get_F1(dev_samples, dev_preds)
        custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        dev_acc = (2 * p * r) / (p + r + 1e-8)
        custom_print('F1:', dev_acc)
        print('epoch',epoch_idx,'p',p,'r',r,'f1',dev_acc)
        if dev_acc >= best_dev_acc:
            best_epoch_idx = epoch_idx + 1
            best_epoch_seed = cur_seed
            custom_print('model saved......')
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), best_model_file)

        custom_print('\n\n')
        if epoch_idx + 1 - best_epoch_idx >= config.early_stop_cnt:
            break
    
    custom_print('*******')
    custom_print('Best Epoch:', best_epoch_idx)
    custom_print('Best Epoch Seed:', best_epoch_seed)


def predict(samples, model, model_id,config):
    pred_batch_size = config.test_batch_size
    batch_count = math.ceil(len(samples) / pred_batch_size)  
   
    move_last_batch = False
    if len(samples) - pred_batch_size * (batch_count - 1) == 1:
        move_last_batch = True
        batch_count -= 1
    rel = list()
    arg1s = list()
    arg1e = list()
    arg2s = list()
    arg2e = list()
    model.eval()
    set_random_seeds(config.seed)
    start_time = datetime.datetime.now()
    for batch_idx in range(0, batch_count):
        batch_start = batch_idx * pred_batch_size
        batch_end = min(len(samples), batch_start + pred_batch_size)
        if batch_idx == batch_count - 1 and move_last_batch:
            batch_end = len(samples)

        cur_batch = samples[batch_start:batch_end]
        cur_samples_input = get_batch_data(cur_batch, False)

        src_words_seq = torch.from_numpy(cur_samples_input['src_words'].astype('long'))
        src_segment=torch.from_numpy(cur_samples_input['src_segment'].astype('long'))
        src_words_mask = torch.from_numpy(cur_samples_input['src_words_mask'].astype('long'))
        ori_src_words_mask=torch.from_numpy(cur_samples_input['ori_src_words_mask'].astype('bool'))
        trg_words_seq = torch.from_numpy(cur_samples_input['decoder_input'].astype('long'))
        input_span_mask=torch.from_numpy(cur_samples_input['ner_feature'].astype('long'))

        if torch.cuda.is_available():
            src_words_seq = src_words_seq.cuda()
            src_words_mask = src_words_mask.cuda()
            ori_src_words_mask = ori_src_words_mask.cuda()
            src_segment=src_segment.cuda()
            trg_words_seq = trg_words_seq.cuda()
            input_span_mask = input_span_mask.cuda()

        src_words_seq = autograd.Variable(src_words_seq)
        src_words_mask = autograd.Variable(src_words_mask)
        ori_src_words_mask = autograd.Variable(ori_src_words_mask)
        src_segment = autograd.Variable(src_segment)
        trg_words_seq = autograd.Variable(trg_words_seq)
        input_span_mask = autograd.Variable(input_span_mask)
        
        with torch.no_grad():
            if model_id == 1:  
                outputs = model(src_words_seq, src_words_mask, ori_src_words_mask, src_segment, trg_words_seq,
                                config.max_trg_len,  None, None, config.dec_hidden_size, config.dec_att_type, input_span_mask,False) #positional_seq,  src_chars_seq,  adj,

        rel += list(outputs[0].data.cpu().numpy())
        arg1s += list(outputs[1].data.cpu().numpy())
        arg1e += list(outputs[2].data.cpu().numpy())
        arg2s += list(outputs[3].data.cpu().numpy())
        arg2e += list(outputs[4].data.cpu().numpy())
        model.zero_grad()
    end_time = datetime.datetime.now()
    custom_print('Prediction time:', end_time - start_time)
    return rel, arg1s, arg1e, arg2s, arg2e  



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest='config',type=str, default="./my_config.ini")
    parser.add_argument("--srcpath",type=str, default='data/')
    parser.add_argument("--trgpath",type=str, default='data/test')
    parser.add_argument("--mode",type=str, default='train',choices=['train','test'])

    args = parser.parse_args()
    config = Config(args.config) 
    config.vocab_size=len(VOCAB)
    
    random_seed = config.seed
    n_gpu = torch.cuda.device_count()
    set_random_seeds(random_seed)

    src_data_folder = args.srcpath
    trg_data_folder = args.trgpath
    if not os.path.exists(trg_data_folder):
        os.mkdir(trg_data_folder)
    model_name = 1
    job_mode = args.mode
    model_path = config.model_path
    processor=relationsTextProcessor(src_data_folder)

    bertconfig = BertConfig.from_pretrained(       
         model_path, finetuning_task=config.task_name,output_hidden_states=True)
    
    bertconfig.l2_reg_lambda = config.l2_reg_lambda
    bertconfig.is_decoder=False

    tokenizer = BertTokenizer.from_pretrained(
         model_path, do_lower_case=True)  

    # train a model
    if job_mode == 'train':
        logger = open(os.path.join(trg_data_folder, 'training.log'), 'w')  
        datatype=1  #train 1    dev 2   test 3
    
        custom_print(sys.argv)
        custom_print(config)
        custom_print('loading data......')
        model_file_name = os.path.join(trg_data_folder, 'BERT_for_joint_model.bin')
        train_examples = processor.get_train_examples('train.sent','train.pointer','train.ner',datatype=datatype)
        train_data = convert_examples_to_features(
            train_examples, config.max_seq_len, tokenizer)
        dev_examples= processor.get_dev_examples('dev.sent','dev.pointer','dev.ner',datatype=2)
        dev_data = convert_examples_to_features(
        dev_examples, config.max_seq_len, tokenizer)

        custom_print('Training data size:', len(train_data))
        custom_print('Development data size:', len(dev_data))


        custom_print("Training started......")
        train_model(model_name, train_data, dev_data, model_file_name,bertconfig,config)
        logger.close()

    if job_mode == 'test':
        logger = open(os.path.join(trg_data_folder, 'test.log'), 'w')
        datatype=3
        custom_print(sys.argv)
        model_file = os.path.join(trg_data_folder, 'BERT_for_joint_model.bin')

        best_model = get_model(model_name,bertconfig,config) 
        custom_print(best_model)
        if torch.cuda.is_available():
            best_model.cuda()
        if n_gpu > 1:
            best_model = torch.nn.DataParallel(best_model,device_ids=[0])
        best_model.load_state_dict(torch.load(model_file))

        custom_print('\nTest Results\n')
        test_examples = processor.get_test_examples('test.sent','test.pointer','test.ner',datatype=datatype)
        test_data = convert_examples_to_features(
            test_examples, config.max_seq_len, tokenizer)

        reader = open(os.path.join(src_data_folder, 'test.tup'))
        test_gt_lines = reader.readlines()
        reader.close()

        custom_print('Test size:', len(test_data))
        set_random_seeds(random_seed)
        test_preds = predict(test_data, best_model, model_name, config)
        
        pred_pos, gt_pos, correct_pos = get_F1(test_data, test_preds)
        custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)  
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        test_acc = (2 * p * r) / (p + r + 1e-8)
        custom_print('P:', round(p, 3))
        custom_print('R:', round(r, 3))
        custom_print('F1:', round(test_acc, 3))
        write_test_res(test_data, test_preds, os.path.join(trg_data_folder, 'test.out'))
        
        logger.close()


