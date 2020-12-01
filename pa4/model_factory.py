################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

from torch import nn
from torchvision import models as model_zoo

TODO = object()

# Build and return the model here based on the configuration.
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_sequence.html#torch.nn.utils.rnn.pack_sequence
# https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence
# https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch

class ExperimentModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, embedding: nn.Embedding):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding

    def forward(self, images, captions):
        batch_size = image.size(0)
        latent_state = self.encoder(images)
        packed_output, (ht, ct) = self.decoder(latent_state)
        output = self.embedding()
        return out


def get_model(config_data, vocab):
    """
    High Level Factory
    """
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    dropout = config_data['model']['dropout']
    nonlinearity = config_data['model'].get('nonlinearity') or 'tanh'

    embedding = get_embedding()
    encoder = get_encoder(output_size=embedding_size, fine_tune=False)
    if model_type == 'baseline':
        decoder = get_lstm(input_size=embedding_size, hidden_size=hidden_size, num_layers=1, dropout=dropout)
    elif model_type == 'baseline_variant_rnn':
        # encoder = get_encoder(output_size=embedding_size, fine_tune=False)
        decoder = get_rnn(input_size=embedding_size, hidden_size=hidden_size, num_layers=1, dropout=dropout, nonlinearity=nonlinearity)
    else:
        raise NotImplementedError(f'Unknown model type {model_type}')

    model = ExperimentModel(encoder, decoder, embedding)

    return model


# Low Level Factories

# Embedding
def get_embedding():
    # TODO
    ...

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
    return model

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