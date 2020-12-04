################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import torch
from torch import nn
from torchvision import models as model_zoo

from .vocab import Vocabulary

TODO = object()

# Build and return the model here based on the configuration.
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_sequence.html#torch.nn.utils.rnn.pack_sequence
# https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence
# https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch


class Encoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, images):
        return self.model(images)
    

class Decoder(nn.Module):
    def __init__(self, model: nn.Module, embedding: nn.Module):
        super().__init__()
        self.model = model
        self.embedding = embedding

    def forward(self, features, captions):
        embeddings = self.embedding(captions)  # 64xVAR_LENxEMBED_DIM
        # ---------------------
        #                      \ 64x1xEMBED_DIMS
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)  # 64x(VAR_LEN+1)xEMBED_DIMS
        return self.model(embeddings)

        
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
        outputs = self.decoder(encoded, captions)  # LSTM takes in 1. current feature 2. hidden + cell state
        return outputs

    # WIP: Stochastic
    def apply_generation(self, outputs, deterministic: bool =True, temperature: float = 1):
        print("outputs size: ", outputs.size() )
        if deterministic:
            return outputs.argmax(1)
        else:
            # Stochastic
            temp_outputs = torch.div(outputs, temperature)
            temp_outputs = nn.Softmax(temp_outputs)
            picked_idx = torch.multinomial(input=temp_outputs, num_samples=1)
            return outputs[picked_idx]
            # raise NotImplementedError()
    
    
    def generate_captions(self, images):
        captions = []
        max_length = 300  # Where can we get this number. ==> embedding size = 300
        input_ = self.encoder(images).unsqueeze(0)
        states = None
        for _ in range(max_length):
            hidden_outputs, states = self.decoder.model(input_, states)
            prediction = self.apply_generation(hidden_output)
            captions.append(prediction)
            input_ = self.decoder.embedding(prediction).unsqueeze(0)
        return torch.Tensor(captions)




# def get_model(config_data, vocab):
#     """
#     High Level Factory
#     """
#     hidden_size = config_data['model']['hidden_size']
#     embedding_size = config_data['model']['embedding_size']
#     model_type = config_data['model']['model_type']
#     dropout = config_data['model']['dropout']
#     deterministic = config_data['model'].get('deterministic') or True
#     nonlinearity = config_data['model'].get('nonlinearity') or 'tanh'

#     embedding = get_embedding(len(vocab), embedding_size)
#     encoder = get_encoder(output_size=embedding_size, fine_tune=False)
#     if model_type == 'baseline':
#         decoder = get_lstm(input_size=embedding_size, hidden_size=hidden_size, num_layers=1, dropout=dropout)
#     elif model_type == 'baseline_variant_rnn':
#         decoder = get_rnn(input_size=embedding_size, hidden_size=hidden_size, num_layers=1, dropout=dropout, nonlinearity=nonlinearity)
#     else:
#         raise NotImplementedError(f'Unknown model type {model_type}')

#     model = ExperimentModel(encoder, decoder, embedding, deterministic)

#     return model


def get_model(config_data, vocab):
    """
    High Level Factory
    """
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    dropout = config_data['model'].get('dropout', 0)
    nonlinearity = config_data['model'].get('nonlinearity') or 'tanh'

    embedding = get_embedding(len(vocab), embedding_size)
    encoder = get_encoder(output_size=embedding_size, fine_tune=False)
    if model_type == 'baseline':
        decoder = Decoder(
            get_lstm(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=1,
                dropout=dropout,
            ),
            embedding
        )
    elif model_type == 'baseline_variant_rnn':
        decoder = Decoder( 
            get_rnn(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=1,
                dropout=dropout,
                nonlinearity=nonlinearity,
            ),
            embedding
        )
    else:
        raise NotImplementedError(f'Unknown model type {model_type}')

    model = ExperimentModel(encoder, decoder, embedding, vocab)

    return model


# Low Level Factories

# Embedding
def get_embedding(vocab_size, embed_size):
    return nn.Embedding(vocab_size, embed_size)  #


# Encoder: CNN resnet50 Model
def get_encoder(
    type_: str = 'resnet50',
    output_size: int = None,
    fine_tune: str = None,
    progress=False,
) -> model_zoo.ResNet:
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
def get_lstm(
    input_size: int = None,
    hidden_size: int = None,
    num_layers: int = None,
    dropout: float = None,
) -> nn.LSTM:

    if not input_size:
        raise ValueError
    if not hidden_size:
        raise ValueError
    if not num_layers:
        raise ValueError
    if dropout is None:
        raise ValueError
        
    return nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        ##bias=True,
        dropout=dropout,
        batch_first=True,
    )


# Decoder: RNN Vanilla
def get_rnn(
    input_size: int = None,
    hidden_size: int = None,
    num_layers: int = None,
    dropout: float = None,
    nonlinearity: str = None,
) -> nn.RNN:
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
    return nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        dropout=dropout,
        nonlinearity=nonlinearity,
        batch_first=True,
    )