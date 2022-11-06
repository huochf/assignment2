import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import pickle
import os
import sys
import nltk
import cv2
import configparser
from torchvision import transforms
from pycocotools.coco import COCO

from datasets.coco_data_loader import get_loader
from modeling.base_model import EncoderCNN, DecoderRNN
from utils.build_vocab import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def show_attention(image, attn_weights, caption):
    h, w = 224, 224
    pad_h = 30

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).reshape(3, 1, 1)
    image = ((image[0] * std + mean) ) * 255.
    image = image[ [2, 1, 0], :, :]
    image = image.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

    img_show = np.ones(((h + pad_h) * 4, w * 4, 3), dtype=np.uint8) * 255.
    for idx, word in enumerate(caption):
        if idx >= 16:
            break
        attn_weight = attn_weights[idx][:-1].reshape(7, 7).detach().cpu().numpy()
        attn_weight = cv2.resize(attn_weight, (224, 224))
        attn_weight = attn_weight / attn_weight.max() * 255.
        attn_weight = attn_weight.astype(np.uint8)

        attn_weight = cv2.applyColorMap(attn_weight, cv2.COLORMAP_JET)
        alpha = 0.5
        heat_image = cv2.addWeighted(image.copy(), alpha, attn_weight, 1 - alpha, 0)
        i, j = idx // 4, idx % 4
        img_show[(h + pad_h) * i:(h + pad_h) * i + 224, w * j: w * j + 224] = heat_image
        img_show = cv2.putText(img_show, caption[idx] + '({:.2f})'.format(attn_weights[idx][-1]), 
            (w * j + 90, (h + pad_h) * i + 224 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return img_show


def show_caption(image, caption, caption_gt):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).reshape(3, 1, 1)
    image = ((image[0] * std + mean) ) * 255.
    image = image[ [2, 1, 0], :, :]
    image = image.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    image = cv2.resize(image, (600, 600))
    image = np.concatenate([image, 255 * np.ones((64, 600, 3))], axis=0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(image, 'prediction: ' + caption, (10, 615), font, 0.5, (0, 0, 0), 2)
    image = cv2.putText(image, 'GT: ' + caption_gt, (10, 640), font, 0.5, (0, 0, 255), 2)
    return image


def main(args):
    config = configparser.ConfigParser()
    config.read(args.config)
    params = config['EVAL']
    model_path = params['model_path']
    crop_size = int(params['crop_size'])
    vocab_path = params['vocab_path']
    image_dir = params['image_dir']
    caption_path = params['caption_path']
    checkpoint_name = str(params['checkpoint_name'])

    embed_size = int(params['embed_size'])
    hidden_size = int(params['hidden_size'])
    num_layers = int(params['num_layers'])

    batch_size = int(params['batch_size'])
    num_workers = int(params['num_workers'])
    e2e = params.getboolean('e2e')

    # Image preprocessing
    transform = transforms.Compose([ 
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build the models
    if params['encoder'] == 'sequence':
        encoder = EncoderCNN(embed_size, e2e).eval()
    else:
        raise NotImplementedError()
    encoder = encoder.to(device)

    if params['decoder'] == 'lstm':
        decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers, ).eval()
    else:
        raise NotImplementedError()
    decoder = decoder.to(device)

    # Load pretrained model
    if os.path.exists(params['checkpoint']):
        print('Loading pretrained models from {}...'.format(params['checkpoint']))
        state_dict = torch.load(str(params['checkpoint']))
        encoder.load_state_dict(state_dict['encoder'])
        decoder.load_state_dict(state_dict['decoder'])
        epoch = state_dict['epoch']
    else:
        print('pretrained models not found!')
        exit(1)

    data_loader = get_loader(image_dir, caption_path, vocab, 
                             transform, 1, 
                             shuffle=True, num_workers=1)

    def id_to_word(si):
        s = []
        for word_id in si:
            word = vocab.idx2word[word_id]
            if word == '<end>':
                break
            s.append(word)
        return s

    for idx, (img_ids, images, captions, lengths) in enumerate(data_loader):
        if idx > args.vis_image_num:
            break

        images = images.to(device)
        feature = encoder(images)
        sampled_ids, attn_weights = decoder.sample(feature)
        gen_cap = id_to_word(sampled_ids[0].detach().cpu().numpy())
        gt_cap = id_to_word(captions[0].detach().cpu().numpy())

        image_vis = show_caption(images, ' '.join(gen_cap), ' '.join(gt_cap[1:]))

        img_dir = os.path.join(model_path, 'visualization')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        cv2.imwrite(os.path.join(img_dir, '{}_vis.jpg'.format(img_ids[0])), image_vis)
        if params['decoder'] == 'attention':
            image_attn = show_attention(images, attn_weights[0], gen_cap)
            cv2.imwrite(os.path.join(img_dir, '{}_vis_attn.jpg'.format(img_ids[0])), image_attn)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_base.ini')
    parser.add_argument('--vis_image_num', type=int, default=32)
    args = parser.parse_args()
    main(args)
