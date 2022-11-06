import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import pickle
import os
import sys
import nltk
import configparser
from torchvision import transforms
from pycocotools.coco import COCO

from datasets.coco_data_loader import get_loader
from modeling.base_model import EncoderCNN, DecoderRNN
from utils.build_vocab import Vocabulary

from utils.pycocoevalcap.eval import COCOEvalCap

c = nltk.translate.bleu_score.SmoothingFunction()
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_model(encoder, decoder, data_loader, vocab, caption_path, use_beam_search=False, device=torch.device('cuda')):
    encoder.eval()
    decoder.eval()
    def id_to_word(si):
        s = []
        for word_id in si:
            word = vocab.idx2word[word_id]
            if word == '<end>':
                break
            s.append(word)
        return(s)

    predictions = []
    processed = []
    coco = COCO(caption_path)
    ids = coco.getImgIds()
    for i, (img_ids, images, captions, lengths) in enumerate(data_loader):
        # Generate an caption from the image
        images = images.to(device)
        feature = encoder(images)
        if not use_beam_search:
            sampled_ids = decoder.sample(feature)
        else:
            sampled_ids = decoder.sample_beam_search(feature)
        sampled_ids = sampled_ids.cpu().numpy()

        for idx, img_id in enumerate(img_ids):
            if img_id not in processed and img_id in ids:
                gen_cap = id_to_word(sampled_ids[idx])
                predictions.append({'image_id': img_id, 'caption': ' '.join(gen_cap)})
                processed.append(img_id)

        if i % 10 == 0:
            print('{}/{}'.format(i, len(data_loader)))
            sys.stdout.flush()
    print(len(predictions))
    coco = COCO(caption_path)
    cocoRes = coco.loadRes(predictions)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    return cocoEval.eval


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

    use_beam_search = params.getboolean('use_beam_search')

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
                             transform, batch_size, 
                             shuffle=False, num_workers=num_workers)

    eval_metrics = test_model(encoder, decoder, data_loader, vocab, caption_path, use_beam_search=use_beam_search)
    for metric, score in eval_metrics.items():
        print('%s: %.3f'%(metric, score))
    eval_metrics['epoch'] = epoch
    with open(os.path.join(model_path, '{}_test_metric.json'.format(checkpoint_name)), 'a') as f:
        f.write(json.dumps(eval_metrics) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_base.ini')

    args = parser.parse_args()
    main(args)    
