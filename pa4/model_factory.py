################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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
    def __init__(self, model: nn.Module, hidden_size: int, output_size: int):
        super().__init__()
        self.model = model
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, features, lengths=None, state=None, return_state=False):
        if lengths is not None:
            packed_features = pack_padded_sequence(features, lengths, batch_first=True)
            packed_output, state = self.model(packed_features, state)
            output = pad_packed_sequence(packed_output)
        else:
            output, state = self.model(features, state)

        raw = self.linear(output)

        if return_state:
            return raw, state
        return raw


        
# MODEL
class ExperimentModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, embedding: nn.Embedding, vocab: Vocabulary, max_length: int, deterministic: bool, temperature: float):
        super().__init__()
        self.vocab = vocab

        self.encoder = encoder
        self.embedding = embedding
        self.decoder = decoder
        self.max_length = max_length
        self.deterministic = deterministic
        self.temperature = temperature

        if not self.deterministic and self.temperature is None:
            raise EnviromentError("Non deterministic missing temperature on model initialization")

    def forward(self, images, captions, lengths):
        encoded = self.encoder(images)
        embeddings = self.embedding(captions)  # 64xVAR_LENxEMBED_DIM
        # ---------------------
        #                      \ 64x1xEMBED_DIMS
        features = torch.cat((encoded.unsqueeze(1), embeddings[:, :-1]), dim=1)  # 64x(VAR_LEN+1)xEMBED_DIMS

        outputs = self.decoder(features, lengths)  # LSTM takes in 1. current feature 2. hidden + cell state
        return outputs
    
    def forward_generate(self, images):
        input_ = self.encoder(images).unsqueeze(1)
        state = None
        batch_size = images.shape[0]
        captions = torch.LongTensor(batch_size, self.max_length)
    
        for i in range(self.max_length):
            output, state = self.decoder(input_, state, return_state=True)
            token = self.apply_generation(output.squeeze(1))
            captions[:, i] = token
            input_ = self.embedding(token).unsqueeze(1)

        return captions

    def apply_generation(self, outputs):
        # outputs: 64xVOCAB_SIZE
        if self.deterministic:
            return outputs.argmax(1)

        # Stochastic
        temp_outputs = nn.functional.softmax(outputs / self.temperature)
        picked_idx = torch.multinomial(temp_outputs, num_samples=1)
        return outputs[picked_idx]

class ExperimentModelVariant2(ExperimentModel):
    def forward(self, images, captions, lengths):
        seq_len = captions.size(1)

        latent = self.encoder(images).unsqueeze(1)  # Shape: BATCHx1xLATENT_DIMS
        latent_stacked = latent.expand(-1, seq_len, -1)  # Shape: BATCHxSEQ_LENxLATENT_DIMS

        captions = torch.cat(torch.zeros(1), captions)  # Shape: BATCHx(SEQ_LEN+1)
        embeddings = self.embedding(captions)  # Shape: BATCHx(SEQ_LEN+1)xEMBED_DIMS

        features = torch.cat((latent_stacked, embeddings[:, :-1]), dim=2)  # Shape: BATCHxSEQ_LENx(LATENT_DIMS+EMBED_DIMS)

        outputs = self.decoder(features)
        return outputs

    def forward_generate(self, images):
        LATENT_DIMS = images.size(-1)
        latent = self.encoder(images).unsqueeze(1)  # Shape: BATCHx1xLATENT_DIMS
        pad = self.embedding(torch.zeros(64, 1))  # Shape: BATCHx1xEMBED_DIMS
        input_ = torch.cat((latent, pad))

        for i in range(self.max_length):
            output, state = self.decoder(input_, state, return_state=True)
            token = self.apply_generation(output.squeeze(1))
            captions[:, i] = token
            embedded = self.embedding(token).unsqueeze(1)
            input_[LATENT_DIMS:] = embedded

        return captions


'''
1. Run encoder
2. Pass encoder output into LSTM with empty state
3. Get prediction from LSTM state, and store prediction
4. Pass prediction (re-embedded) into LSTM
'''


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
    max_length = config_data['generation']['max_length']
    deterministic = config_data['generation']['deterministic']
    temperature = config_data['generation'].get('temperature')

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
            hidden_size=hidden_size,
            output_size=len(vocab),
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
            hidden_size=hidden_size,
            output_size=len(vocab),
        )
    else:
        raise NotImplementedError(f'Unknown model type {model_type}')

    model = ExperimentModel(encoder, decoder, embedding, vocab, max_length, deterministic, temperature)

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
        bias=True,
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