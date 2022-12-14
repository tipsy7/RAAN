
"""Data provider"""


import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import json as jsonmod
import pandas as pd
import json
import nltk
import pdb
import pickle

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'

        # Captions
        self.captions = []
        with open(loc + '%s_precaps_stan.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc + '%s_ims.npy' % data_split, allow_pickle=True)

        with open(loc + '%s_word_frequency.pickle' % data_split, 'rb') as file:
            self.word_freq = pickle.load(file)


        self.length = len(self.captions)

        self.bbox = np.load(loc + '%s_ims_bbx.npy' % data_split, allow_pickle=True)
        self.sizes = np.load(loc + '%s_ims_size.npy' % data_split, allow_pickle=True)

        with open(loc + '%s_caps.json' % data_split) as f:
            self.depends = json.load(f)

        print('image shape', self.images.shape)
        print('text shape', len(self.captions))

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index / self.im_div
        # obj_id = index / 5
        image = torch.Tensor(self.images[int(img_id)])
        caption = self.captions[index]
        vocab = self.vocab
        depend = self.depends[index]

        # Convert caption (string) to word ids.
        caps = []
        caps.extend(caption.split(b','))
        caps = list(map(int, caps))

        # Load bbox and its size
        bboxes = self.bbox[int(img_id)]

        word_freqs = self.word_freq[index]

        imsize = self.sizes[int(img_id)]
        # k sample
        k = image.shape[0]
        assert k == 36

        for i in range(k):
            bbox = bboxes[i]
            bbox[0] /= imsize['image_w']
            bbox[1] /= imsize['image_h']
            bbox[2] /= imsize['image_w']
            bbox[3] /= imsize['image_h']
            bboxes[i] = bbox

        captions = torch.Tensor(caps)
        bboxes = torch.Tensor(bboxes)

        return image, captions, bboxes, depend, index, img_id , word_freqs

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, bboxes, depends, ids, img_ids, word_freqs = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    bboxes = torch.stack(bboxes, 0)
    # word_freqs = torch.stack(word_freqs, 0)
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()

    word_freqs_new = torch.zeros(len(word_freqs), max(lengths)).float()

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    for i ,wd_fr in enumerate(word_freqs):
        end = lengths[i]
        word_freqs_new[i, :end] = torch.tensor(wd_fr[:end])

    word_freqs_new = word_freqs_new.unsqueeze(-1).contiguous()
    return images, targets, bboxes, depends, lengths, ids, word_freqs_new


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader
