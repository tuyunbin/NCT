import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import json
import argparse
import random
import h5py
import numpy as np
import pickle
import re

from collections import defaultdict
from utils.preprocess import tokenize, encode, build_vocab, build_dep_vocab
parser = argparse.ArgumentParser()
parser.add_argument('--input_captions_json', default="./edit_data/change_captions.json", help='input captions json file to process into hdf5')

parser.add_argument('--input_image_dir', default="./edit_data/img0", help='input image directory')
parser.add_argument('--input_vocab_json', default='')
parser.add_argument('--input_dep_vocab_json', default='')
parser.add_argument('--input_captions_dep_json', default="./edit_data/change_captions_dep.json", help='input captions json file to process into hdf5')
parser.add_argument('--split_json', default="./edit_data/splits.json", help='json file that contains the dataset splits')
parser.add_argument('--output_vocab_json', default="./edit_data/transformer_vocab.json", help='output vocab file')
parser.add_argument('--output_dep_vocab_json', default="./edit_data/transformer_dep_vocab.json", help='output vocab file')
parser.add_argument('--output_h5', default="./edit_data/transformer_labels.h5", help='output h5 file')

parser.add_argument('--word_count_threshold', default=1, type=int)
parser.add_argument('--allow_unk', default=1, type=int)

def main(args):
    print('Loading captions')
    with open(args.input_captions_json, 'r') as f:
        captions = json.load(f)
    with open(args.input_captions_dep_json, 'r') as f:
        deps = json.load(f)
    # for k, v in deps.items():
    #     for j in v:
    #         j = ''.join(j)

    with open(args.split_json, 'r') as f:
        splits = json.load(f)
    all_imgs = sorted(os.listdir(args.input_image_dir))
    captioned_imgs = list(captions.keys())
    all_captions = []
    for img, caps in captions.items():
        all_captions.extend(caps)

    all_deps = []
    for img, dep in deps.items():
        all_deps.extend(dep)

    # cap_keys = sorted(list(captions.keys()))
    # for img in cap_keys:
    #     caps = captions[img]
    #     dep = deps[img]
    #     for i, j in zip(caps, dep):
    #         sen = i.split()
    #         assert len(sen) == len(j)


    # Extract train data points
    train_split = splits['train']
    train_imgs = [all_imgs[idx] for idx in train_split]
    train_captions = []

    for img in train_imgs:
        r = '[’!"#$%&\'()*+-./:<=>?@[\\]^_`{|}~\n。！，]+'

        cap = captions[img]
        # for c in cap:
        #     c = re.sub(r, '', c)

        train_captions.extend(cap)


    N = len(all_imgs)
    N_captioned = len(captions)
    M = len(all_captions)
    print('Total images: %d' % N)
    print('Total captioned images: %d' % N_captioned)
    print('Total captions: %d' % M)
    print('Total train images: %d' % len(train_imgs))
    print('Total train captions: %d' % len(train_captions))


    # Either create the vocab or load it from disk
    if args.input_vocab_json == '':
        print('Building vocab')
        word_to_idx = build_vocab(
            train_captions,
            min_token_count=args.word_count_threshold,
            # punct_to_keep=[';', ','], punct_to_remove=['?', '.']
        )
    else:
        print('Loading vocab')
        with open(args.input_vocab_json, 'r') as f:
            word_to_idx = json.load(f)
    if args.output_vocab_json != '':
        with open(args.output_vocab_json, 'w') as f:
            json.dump(word_to_idx, f)

    if args.input_dep_vocab_json == '':
        print('Building dependency vocab')
        dep_to_idx = build_dep_vocab()
    else:
        print('Loading dependency vocab')
        with open(args.input_dep_vocab_json, 'r') as f:
            dep_to_idx = json.load(f)
    if args.output_dep_vocab_json != '':
        with open(args.output_dep_vocab_json, 'w') as f:
            json.dump(dep_to_idx, f)

    # Encode all captions
    # First, figure out max length of captions
    all_cap_tokens = []
    max_length = -1
    cap_keys = sorted(list(captions.keys()))
    for img in cap_keys:
        caps = captions[img]
        n = len(caps)
        assert n > 0, 'error: some image has no caption'
        tokens_list = []
        for cap in caps:
            r = '[’!"#$%&\'()*+-./:<=>?@[\\]^_`{|}~\n。！，]+'
            cap = re.sub(r, '', cap)
            # cap_tokens = tokenize(cap,
            #                       add_start_token=True,
            #                       add_end_token=True,
            #                       # punct_to_keep=[';', ','],
            #                       # punct_to_remove=['?', '.']
            #                       )
            cap_tokens = cap.split()
            cap_tokens.insert(0, '<START>')
            cap_tokens.append('<END>')

            tokens_list.append(cap_tokens)
            max_length = max(max_length, len(cap_tokens))
        all_cap_tokens.append((img, tokens_list))

        ## dep tokenize
    all_dep_tokens = []
    max_dep_length = -1
    dep_keys = sorted(list(deps.keys()))
    for img in dep_keys:
        dep = deps[img]
        n = len(dep)
        assert n > 0, 'error: some image has no caption'
        tokens_list = []
        for d in dep:
            d.insert(0, '<START>')
            d.append('<END>')
            tokens_list.append(d)
            max_dep_length = max(max_dep_length, len(d))
        all_dep_tokens.append((img, tokens_list))

    print('Encoding captions')
    label_arrays = []
    dep_arrays = []
    label_start_idx = -np.ones(N, dtype=np.int64)
    label_end_idx = -np.ones(N, dtype=np.int64)
    label_length = np.zeros(M, dtype=np.int64)
    dep_length = np.zeros(M, dtype=np.int64)
    caption_counter = 0
    dep_counter = 0
    counter = 0

    # Then encode
    for img, tokens_list in all_cap_tokens:
        i = int(img.split('.')[0].split('_')[-1])
        n = len(tokens_list)
        Li = np.zeros((n, max_length), dtype=np.int64)
        for j, tokens in enumerate(tokens_list):
            label_length[caption_counter] = len(tokens)
            caption_counter += 1
            tokens_encoded = encode(tokens,
                                    word_to_idx,
                                    allow_unk=args.allow_unk == 1)
            for k, w in enumerate(tokens_encoded):
                Li[j, k] = w
        # captions are padded with zeros
        label_arrays.append(Li)
        label_start_idx[i] = counter
        label_end_idx[i] = counter + n - 1

        counter += n

    for img, dep_tokens_list in all_dep_tokens:
        i = int(img.split('.')[0].split('_')[-1])
        n = len(dep_tokens_list)
        Di = np.zeros((n, max_dep_length), dtype=np.int64)
        for j, tokens in enumerate(dep_tokens_list):
            dep_length[dep_counter] = len(tokens)
            dep_counter += 1
            dep_tokens_encoded = encode(tokens,
                                        dep_to_idx,
                                        allow_unk=args.allow_unk == 1)
            for k, w in enumerate(dep_tokens_encoded):
                Di[j, k] = w
        # captions are padded with zeros
        dep_arrays.append(Di)
    D = np.concatenate(dep_arrays, axis=0)
    L = np.concatenate(label_arrays, axis=0)  # put all labels together
    assert L.shape[0] == M, "lengths don't match?"
    assert np.all(label_length > 0), 'error: some captions have no word?'
    assert np.all(label_length == dep_length)


    # Create h5 file
    print('Writing output')
    print('Encoded captions array size: ', L.shape)
    with h5py.File(args.output_h5, 'w') as f:
        f.create_dataset('labels', data=L)
        f.create_dataset('deps', data=D)
        f.create_dataset('label_start_idx', data=label_start_idx)
        f.create_dataset('label_end_idx', data=label_end_idx)
        f.create_dataset('label_length', data=label_length)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
