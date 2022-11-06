import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):

    def __init__(self, embed_size, e2e=False, dropout_ratio=0.):
        """Load the pretrained ResNet and replace top fc layers."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.feature_embed = nn.Sequential(
            nn.Linear(resnet.fc.in_features, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
        )
        self.e2e = e2e

    def forward(self, images):
        """Extract feature vectors from input images."""
        if self.e2e:
            features = self.resnet(images)
        else:
            with torch.no_grad():
                features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.feature_embed(features)
        return features


class DecoderRNN(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20, vocab=None, dropout_ratio=0.):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
        )
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length
        self.rnn = nn.LSTMCell(embed_size, hidden_size)
        self.rnn_drop = nn.Dropout(dropout_ratio)

        if vocab is not None:
            self.pad_id = vocab('<pad>')
            self.unk_id = vocab('<unk>')
            self.sos_id = vocab('<start>')
            self.eos_id = vocab('<end>')
        else:
            self.pad_id = 0
            self.unk_id = 3
            self.sos_id = 1
            self.eos_id = 2

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        if isinstance(features, tuple):
            features = features[1]
        state = self.rnn(features)

        outputs = []
        for i in range(captions.size(1) - 1):
            inputs = self.word_embed(captions[:, i])
            state = self.rnn(inputs, state)
            output = self.rnn_drop(state[0])
            output = self.linear(output)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        if isinstance(features, tuple):
            features = features[1]
        b = features.shape[0]
        sampled_ids = []
        states = self.rnn(features)
        start = features.new_zeros(b, dtype=torch.long).fill_(self.sos_id)
        inputs = self.word_embed(start)

        for i in range(self.max_seq_length):
            states = self.rnn(inputs, states)
            outputs = self.linear(states[0])
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.word_embed(predicted)  
        sampled_ids = torch.stack(sampled_ids, 1) 
        return sampled_ids

    def sample_beam_search(self, features, states=None, beam_size=3):
        raise NotImplementedError()

        return sampled_ids

