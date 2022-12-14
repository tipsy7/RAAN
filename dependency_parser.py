
"""Data preprocessing with semantic parsing"""

from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
import json
import argparse

nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2018-10-05')


def parse_a_sentence(sent):
    return nlp.dependency_parse(sent)


def read_files(src_path,  tar_path):
    print('src_path', src_path)
    print('tar_path', tar_path)
    data = []
    reader = open(src_path, 'r')

    for line in tqdm(reader.readlines()):
        line = line.strip()
        parse = parse_a_sentence(line)
        # store semantically dependent word ids
        sent = [(i[1:3]) for i in parse]
        data.append(sent)

    nlp.close()
    reader.close()

    with open(tar_path, "w") as f:
        json.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocessing for text data')
    parser.add_argument(
        '--src_path', type=str, help='path of data phases to be processed', required=True)
    parser.add_argument(
        '--tar_path', type=str, help='dest path of processed data', required=True)

    opt = parser.parse_args()
    read_files(opt.src_path, opt.tar_path)

    print('Done')
