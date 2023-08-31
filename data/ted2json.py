import json
import os
from tqdm import tqdm

def read_txt(txt_path_en, txt_path_ch):
    with open(txt_path_en, 'r', encoding='utf-8') as f:
        en_all = f.readlines()

    with open(txt_path_ch, 'r', encoding='utf-8') as f:
        ch_all = f.readlines()

    list_of_data = []

    for en, zh in tqdm(zip(en_all, ch_all)):
        temp = [en.strip(), zh.strip()]
        list_of_data.append(temp)
    return list_of_data

def write_json(content, json_path, file_name):
    json_data = json.dumps(content, ensure_ascii=False, indent=4).encode('utf-8')
    with open(f'{json_path}/{file_name}', 'wb') as file:
        file.write(json_data)

def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data


if __name__ == '__main__':
    # list_of_data = read_txt('.')
    train = read_txt('./data/ted_raw/ted_train_en-zh.raw.en', './data/ted_raw/ted_train_en-zh.raw.zh')
    dev = read_txt('./data/ted_raw/ted_dev_en-zh.raw.en', './data/ted_raw/ted_dev_en-zh.raw.zh')
    test = read_txt('./data/ted_raw/ted_test1_en-zh.raw.en', './data/ted_raw/ted_test1_en-zh.raw.zh')
    write_json(train, './data/json_ted', 'train.json')
    write_json(dev, './data/json_ted', 'dev.json')
    write_json(test, './data/json_ted', 'test.json')
    