import argparse
import numpy as np
import os
import sys
import pickle
import configparser
import json

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from datasets.coco_data_loader import get_loader
from utils.build_vocab import Vocabulary
from modeling.base_model import EncoderCNN, DecoderRNN
from eval import test_model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    config = configparser.ConfigParser()
    config.read(args.config)
    params = config['TRAIN']
    model_path = params['model_path']
    crop_size = int(params['crop_size'])
    vocab_path = params['vocab_path']
    image_dir = params['image_dir']
    caption_path = params['caption_path']

    batch_size = int(params['batch_size'])
    num_workers = int(params['num_workers'])
    learning_rate = float(params['learning_rate'])
    num_epochs = int(params['num_epochs'])
    log_step = int(params['log_step'])
    save_step = int(params['save_step'])
    checkpoint_name = str(params['checkpoint_name'])

    embed_size = int(params['embed_size'])
    hidden_size = int(params['hidden_size'])
    num_layers = int(params['num_layers'])
    e2e = params.getboolean('e2e')
    encoder_dropout_ratio = float(params['encoder_dropout_ratio'])
    decoder_dropout_ratio = float(params['decoder_dropout_ratio'])

    # Create model directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    data_loader = get_loader(image_dir, caption_path, vocab, 
                             transform, batch_size, 
                             shuffle=True, num_workers=num_workers)
    eval_loader = get_loader(config['EVAL']['image_dir'], config['EVAL']['caption_path'], vocab, 
                             transform, batch_size, 
                             shuffle=True, num_workers=num_workers)

    # Build the models
    if params['encoder'] == 'single_vector':
        encoder = EncoderCNN(embed_size, e2e, dropout_ratio=encoder_dropout_ratio).cuda()
    else:
        raise NotImplementedError()

    if params['decoder'] == 'lstm':
        decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers, dropout_ratio=decoder_dropout_ratio).cuda()
    else:
        raise NotImplementedError()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if not e2e:
        model_params = list(decoder.parameters()) + list(encoder.feature_embed.parameters())
    else:
        model_params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(model_params, lr=learning_rate)

    # Load pretrained model
    begin_epoch = 0
    if os.path.exists(params['checkpoint']):
        print('Loading pretrained models from {}...'.format(params['checkpoint']))
        state_dict = torch.load(str(params['checkpoint']))
        encoder.load_state_dict(state_dict['encoder'])
        decoder.load_state_dict(state_dict['decoder'])
        optimizer.load_state_dict(state_dict['optimizer'])
        if params.getboolean('drop_lr'):
            print('drop learning rate...')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        if not params.getboolean('resume'):
            begin_epoch = state_dict['epoch']

    # Train the models
    total_step = len(data_loader)
    # test_model(encoder, decoder, eval_loader, vocab, config['EVAL']['caption_path'], )
    for epoch in range(begin_epoch, num_epochs):
        encoder.train()
        decoder.train()
        for i, (img_ids, images, captions, lengths) in enumerate(data_loader):

            # Set mini-batch dataset
            images = images.cuda()
            captions = captions.cuda()
            targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]

            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                    .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                sys.stdout.flush()

        # Save the model checkpoints
        if (epoch + 1) % save_step == 0:
            state_dict = {
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            torch.save(state_dict, os.path.join(model_path, '{}.pth'.format(checkpoint_name)))
            eval_metrics = test_model(encoder, decoder, eval_loader, vocab, config['EVAL']['caption_path'], )
            for metric, score in eval_metrics.items():
                print('%s: %.3f'%(metric, score))
            eval_metrics['epoch'] = epoch
            with open(os.path.join(model_path, '{}_eval_metric.json'.format(checkpoint_name)), 'a') as f:
                f.write(json.dumps(eval_metrics) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_base.ini')

    args = parser.parse_args()
    main(args)
