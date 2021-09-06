# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_iterator
from tqdm import tqdm
import json

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()

def load_dataset(path, pad_size=32):
        PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents

if __name__ == '__main__':
    # inference
    dataset = 'THUCNews'
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    model = x.Model(config).to(config.device)
    #print("Config batch size: ",config.batch_size)

    start_time = time.time()
    print("Loading data...")
    textfile = open(dataset+'/data/test_pred.txt', "w", encoding="utf-8")
    with open(dataset+'/data/test2.txt', encoding='utf-8') as f:
        for line in f:
            textfile.write(line.replace('\n','')+'\t'+str(-1)+'\n')
    textfile.close()

    test_data = load_dataset(dataset+'/data/test_pred.txt', config.pad_size)
    test_iter = build_iterator(test_data, config)

    model.eval()
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in test_iter:
            outputs = model(texts)
            #loss = F.cross_entropy(outputs, labels)
            #loss_total += loss
            #labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            #labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    print('prediction result is: ', predict_all)
    # Save label result into labels.json
    classes = {}
    with open('THUCNews/data'+'/labels.json', encoding='utf-8') as f:
        for i,line in enumerate(f):
            classes[str(i)] = json.loads(line)['label_des']

    textfile = open('THUCNews/data'+"/test_pred.json", "w",encoding="utf-8")
    with open('THUCNews/data'+'/test2.txt', encoding='utf-8') as f:
        for i,line in enumerate(f):
            textfile.write('{"id": '+str(i)+', "label": '+str(predict_all[i])+', "label_des": '+classes[str(predict_all[i])]+', "sentence": '+line.replace("\n","")+'}\n')
    textfile.close()