################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import torch
from torch import nn
from torchvision import models as model_zoo
from vocab import Vocabulary

TODO = object()

# Build and return the model here based on the configuration.
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_sequence.html#torch.nn.utils.rnn.pack_sequence
# https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence
# https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch


class Encoder(nn.Module):
    def __init__(self, model: nn.Module):
        self.model = model
    
    def forward(self, images):
        return self.model(images)
    

class Decoder(nn.Module):
    def __init__(self, model: nn.Module, embedding: nn.Module):
        self.model = model
        self.embedding = embedding

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0) # feature rows are stacked up vertically
        return self.model(embeddings)

    def generate_capions(self, image):
        
        
# MODEL
class ExperimentModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, embedding: nn.Embedding, vocab: Vocabulary):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.vocab = vocab

    def forward(self, images, captions):
        encoded = self.encoder(images)
        outputs = self.decoder(encoded, captions) # LSTM takes in 1. current feature 2. hidden + cell state
        return outputs


def get_model(config_data, vocab):
    """
    High Level Factory
    """
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    dropout = config_data['model']['dropout']
    nonlinearity = config_data['model'].get('nonlinearity') or 'tanh'

    embedding = get_embedding(len(vocab), embedding_size)
    encoder = get_encoder(output_size=embedding_size, fine_tune=False)
    if model_type == 'baseline':
        decoder = get_lstm(input_size=embedding_size, hidden_size=hidden_size, num_layers=1, dropout=dropout)
    elif model_type == 'baseline_variant_rnn':
        decoder = get_rnn(input_size=embedding_size, hidden_size=hidden_size, num_layers=1, dropout=dropout, nonlinearity=nonlinearity)
    else:
        raise NotImplementedError(f'Unknown model type {model_type}')

    model = ExperimentModel(encoder, decoder, embedding)

    return model


# Low Level Factories

# Embedding
def get_embedding(vocab_size, embed_size):
    return nn.Embedding(vocab_size, embed_size) #

# Encoder: CNN resnet50 Model
def get_encoder(type_: str = 'resnet50', output_size: int = None, fine_tune: str = None, progress=False) -> model_zoo.ResNet:
    if type_ == 'resnet50':
        model = model_zoo.resnet50(pretrained=True, progress=progress)
    else:
        raise ValueError(f'Invalid encoder type: {type}')
    # Freeze the model
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
    else:
        raise NotImplementedError('Fine Tuning is not implemented')

    if not output_size:
        raise ValueError('Missing output size')

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=output_size, bias=True)
    return Encoder(model)


# Decoder: LSTM
def get_lstm(input_size: int = None, hidden_size: int = None, num_layers: int = None, dropout: float = None) -> nn.LSTM:
    if not input_size:
        raise ValueError
    if not hidden_size:
        raise ValueError
    if not num_layers:
        raise ValueError
    if dropout is None:
        raise ValueError
    return nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=True, dropout=dropout)
    
# Decoder: RNN Vanilla
def get_rnn(input_size: int = None, hidden_size: int = None, num_layers: int = None, dropout: float = None, nonlinearity: str=None) -> nn.RNN:
    if not input_size:
        raise ValueError
    if not hidden_size:
        raise ValueError
    if not num_layers:
        raise ValueError
    if dropout is None:
        raise ValueError
    if nonlinearity is None:
        raise ValueError
    return nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=True, dropout=dropout, nonlinearity=nonlinearity)
    
# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size, train_CNN=False):
#         super(EncoderCNN, self).__init__()
#         self.train_CNN = train_CNN
#         self.inception = models.inception_v3(pretrained=True, aux_logits=False)
#         self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
#         self.relu = nn.ReLU()
#         self.times = []
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, images):
#         features = self.inception(images)
#         return self.dropout(self.relu(features))


# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
#         super(DecoderRNN, self).__init__()
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
#         self.linear = nn.Linear(hidden_size, vocab_size)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, features, captions):
#         embeddings = self.dropout(self.embed(captions))
#         embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
#         hiddens, _ = self.lstm(embeddings)
#         outputs = self.linear(hiddens)
#         return outputs


# class CNNtoRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
#         super(CNNtoRNN, self).__init__()
#         self.encoderCNN = EncoderCNN(embed_size)
#         self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

#     def forward(self, images, captions):
#         features = self.encoderCNN(images)
#         outputs = self.decoderRNN(features, captions)
#         return outputs

#     def caption_image(self, image, vocabulary, max_length=50):
#         result_caption = []

#         with torch.no_grad():
#             x = self.encoderCNN(image).unsqueeze(0)
#             states = None

#             for _ in range(max_length):
#                 hiddens, states = self.decoderRNN.lstm(x, states)
#                 output = self.decoderRNN.linear(hiddens.squeeze(0))
#                 predicted = output.argmax(1)
#                 result_caption.append(predicted.item())
#                 x = self.decoderRNN.embed(predicted).unsqueeze(0)

#                 if vocabulary.itos[predicted.item()] == "<EOS>":
#                     break

#         return [vocabulary.itos[idx] for idx in result_caption]