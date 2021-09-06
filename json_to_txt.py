import pandas as pd
import json
import argparse
from datasets import load_dataset

### json file format change to text file format for training and testing
### assume train.json, dev.json, test.json, labels.json in data path

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--datapath', type=str, default= 'THUCNews/data', help='select datafile path')
args = parser.parse_args()


if __name__ == '__main__':
    # load json files into one dataset
    dataset = load_dataset('json', data_files={
                    'train':args.datapath+"/train.json",
                    'validation':args.datapath+"/dev.json"
                })

    # train.txt
    sents = dataset['train']['sentence']
    labels = dataset['train']['label']
    textfile = open(args.datapath+"/train.txt", "w",encoding="utf-8")
    for i, sent in enumerate(dataset['train']['sentence']):
        textfile.write(sent + '\t' + labels[i] + "\n")
    textfile.close()

    # valid.txt & test.txt are the same in training process
    sents = dataset['validation']['sentence']
    labels = dataset['validation']['label']
    textfile = open(args.datapath+"/validation.txt", "w",encoding="utf-8")
    textfile2 = open(args.datapath+"/test.txt", "w",encoding="utf-8")
    for i, sent in enumerate(dataset['validation']['sentence']):
        textfile.write(sent + '\t' + labels[i] + "\n")
        textfile2.write(sent + '\t' + labels[i] + "\n")
    textfile.close()
    textfile2.close()

     # test2.txt is the real text file that used for inference
    textfile = open(args.datapath+"/test2.txt", "w",encoding="utf-8")
    with open(args.datapath+'/test.json', encoding='utf-8') as f:
        for line in f:
            textfile.write(json.loads(line)['sentence'] + "\n")
    textfile.close()

    # class.txt
    textfile = open(args.datapath+"/class.txt", "w",encoding="utf-8")
    with open(args.datapath+'/labels.json', encoding='utf-8') as f:
        for line in f:
            textfile.write(json.loads(line)['label_des'] + "\n")
    textfile.close()

