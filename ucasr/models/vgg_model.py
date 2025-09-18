import torch

import torch.nn as nn
import torch.nn.functional as F

from ucasr.models.cca_layer import CCALayer
from ucasr.models.custom_modules import LogSpectrogramModule, ScoreAugModule


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='linear')


class ModelBlock(nn.Module):
    """ Small convolutional block for the main model """

    def __init__(self, input_channels, output_channels, filter_size=(3, 3), padding=(1, 1), mp_size=2,
                 padding_mode='zeros'):
        super(ModelBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=output_channels,
                               kernel_size=filter_size,
                               padding=padding,
                               padding_mode=padding_mode,
                               bias=False)

        self.conv2 = nn.Conv2d(in_channels=output_channels,
                               out_channels=output_channels,
                               kernel_size=filter_size,
                               padding=padding,
                               padding_mode=padding_mode,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(num_features=output_channels)
        self.bn2 = nn.BatchNorm2d(num_features=output_channels)

        self.max_pool = nn.MaxPool2d(kernel_size=mp_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)

        x = self.max_pool(x)

        return x


class AttentionBlock(nn.Module):
    """ Attention convolutional block used before the audio input """

    def __init__(self, args, num_filters=24, padding_mode='zeros'):
        super(AttentionBlock, self).__init__()

        # y filter map dimensions pre-cca embedding layer
        spec_context = int(args.snippet_len[args.audio_context] * args.fps)
        y_fm = (args.spec_bins // 16, spec_context // 16)

        # four main convolutional blocks...
        self.block1 = ModelBlock(input_channels=1,
                                 output_channels=num_filters,
                                 padding_mode=padding_mode)

        self.block2 = ModelBlock(input_channels=num_filters,
                                 output_channels=num_filters * 2,
                                 padding_mode=padding_mode)

        self.block3 = ModelBlock(input_channels=num_filters * 2,
                                 output_channels=num_filters * 4,
                                 padding_mode=padding_mode)

        self.block4 = ModelBlock(input_channels=num_filters * 4,
                                 output_channels=num_filters * 4,
                                 padding_mode=padding_mode)

        # ...followed by a convolutional layer with (spec_context) filter maps...
        self.conv = nn.Conv2d(in_channels=num_filters * 4,
                              out_channels=spec_context,
                              kernel_size=(1, 1),
                              padding=(0, 0))

        # ...then the average of each filter map is computed to generate a (audio_context, 1, 1) tensor
        self.gap = nn.AvgPool2d(kernel_size=y_fm)

    def forward(self, y):
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)

        y = self.conv(y)
        y = self.gap(y)

        y = F.softmax(y, dim=1)

        return y


class VGGEncoder(nn.Module):

    def __init__(self, args, num_filters=24):
        super(VGGEncoder, self).__init__()

        self.block1 = ModelBlock(input_channels=1,
                                 output_channels=num_filters)

        self.block2 = ModelBlock(input_channels=num_filters,
                                 output_channels=num_filters * 2)

        self.block3 = ModelBlock(input_channels=num_filters * 2,
                                 output_channels=num_filters * 4)

        self.block4 = ModelBlock(input_channels=num_filters * 4,
                                 output_channels=num_filters * 4)

        self.conv = nn.Conv2d(in_channels=num_filters * 4,
                              out_channels=32,  # hard-coded for the moment
                              kernel_size=(1, 1),
                              padding=(0, 0),
                              bias=False)

        self.bn = nn.BatchNorm2d(num_features=32)  # hard-coded for the moment

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.bn(self.conv(x))
        x = x.view(x.shape[0], -1)

        return x


class NormalizeModule(nn.Module):
    def __init__(self):
        super(NormalizeModule, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


class MLPHead(nn.Module):
    """ Multilayer Perceptron projection network """

    def __init__(self, args, encoder_embedding_size, normalize_embeddings=True):
        """ initialize model  """
        super(MLPHead, self).__init__()

        encoder_size = encoder_embedding_size
        hidden_size = args.mlp_hidden_size
        emb_dim = args.emb_dim

        self.net = nn.Sequential(nn.Linear(encoder_size, hidden_size),
                                 nn.BatchNorm1d(hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_size, emb_dim))

        self.net.append(NormalizeModule()) if normalize_embeddings else None

    def forward(self, x):
        """
        Forward pass
        x -> latent variable
        """

        x = self.net(x)
        return x


class VGGPathModel(nn.Module):
    def __init__(self, args, num_filters=24, is_audio=True, pretrain=False):
        super(VGGPathModel, self).__init__()

        self.frontend_layer = None
        if pretrain:
            if is_audio:
                self.frontend_layer = LogSpectrogramModule(args)
            else:
                self.frontend_layer = ScoreAugModule(args)

        if is_audio:
            spec_context = int(args.snippet_len[args.audio_context] * args.fps)
            input_size = 32 * (args.spec_bins // 16) * (spec_context // 16)  # 32 is hard-coded
        else:
            input_size = 32 * (args.staff_height // 32) * (args.sheet_context // 32)  # 32 is hard-coded

        self.encoder = VGGEncoder(args, num_filters)

        if pretrain:
            self.mlp_head = MLPHead(args, encoder_embedding_size=input_size, normalize_embeddings=pretrain)
        else:
            self.mlp_head = nn.Linear(in_features=input_size, out_features=args.emb_dim)

        if pretrain:
            self.apply(init_weights)

    def forward(self, x):

        if self.frontend_layer:
            x = self.frontend_layer(x)
        x = self.encoder(x)
        x = self.mlp_head(x)
        return x


class VGGModel(nn.Module):
    """ two-pathway convolution model for cross-modal embedding learning """

    def __init__(self, args, use_cca=True, pretrain=False):
        """ initialize model  """
        super(VGGModel, self).__init__()

        num_filters = 24

        # convolutional net for sheet path (x)
        self.x_net = VGGPathModel(args, num_filters=num_filters, is_audio=False, pretrain=pretrain)

        # convolutional net for audio path (y)
        self.y_net = VGGPathModel(args, num_filters=num_filters, is_audio=True, pretrain=pretrain)

        # initializing the attention branch in case it is present in the model
        self.use_att = args.use_att

        if self.use_att:
            self.att_layer = AttentionBlock(args, num_filters=num_filters, padding_mode='zeros')

        # initializing the cca embedding layer
        self.use_cca = use_cca
        if self.use_cca:
            self.cca_layer = CCALayer(in_dim=args.emb_dim)

        # uses He uniform initialization for the convolutional and linear layers
        self.apply(init_weights)

    def forward(self, x, y, return_pre_cca=False, return_att=False):
        """
        Forward pass
        x -> sheet music snippet
        y -> spectrogram excerpt
        """

        # -- view 1 - sheet
        x = self.x_net(x)

        # -- view 2 - spectrogram
        # getting the attention mask if present in the model and applying it to the audio input
        if self.use_att:
            att = self.att_layer(y)
            att = att.permute(0, 3, 2, 1)
            y = torch.mul(y, att) * y.shape[-1]

        y = self.y_net(y)

        # returns pre-cca latent representations for refining the model after training
        if return_pre_cca:
            return x, y

        # merge modalities by cca projection
        if self.use_cca:
            x, y = self.cca_layer(x, y)

        # normalizing the output final embeddings to length 1.0
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)

        if return_att and self.use_att:
            return x, y, att.squeeze()
        return x, y
