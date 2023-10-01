import torch
from torch import nn
import math
import torch.nn.init as init
from torch.autograd import Function
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from .resnet18 import resnet18


# from utils import weights_init

def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


##################################################
##########  BACKBONE NETWORKS  ###################
##################################################

########## CNN #############################
class cnn_regressor(nn.Module):
    def __init__(self, configs):
        super(cnn_regressor, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat



class regressor(nn.Module):
    def __init__(self, configs):
        super(regressor, self).__init__()
        self.logits = nn.Linear(configs.features_len * configs.final_out_channels, 1)
        self.configs = configs

    def forward(self, x):

        predictions = self.logits(x)

        return predictions



########## TCN #############################

torch.backends.cudnn.benchmark = True  # might be required to fasten TCN

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class tcn_regressor(nn.Module):
    def __init__(self, configs):
        super(tcn_regressor, self).__init__()

        in_channels0 = configs.input_channels
        out_channels0 = configs.tcn_layers[1]
        kernel_size = configs.tcn_kernel_size
        stride = 1
        dilation0 = 1
        padding0 = (kernel_size - 1) * dilation0

        self.net0 = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(out_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
        )

        self.downsample0 = nn.Conv1d(in_channels0, out_channels0, 1) if in_channels0 != out_channels0 else None
        self.relu = nn.ReLU()

        in_channels1 = configs.tcn_layers[0]
        out_channels1 = configs.tcn_layers[1]
        dilation1 = 2
        padding1 = (kernel_size - 1) * dilation1
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
            nn.Conv1d(out_channels1, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
        )
        self.downsample1 = nn.Conv1d(out_channels1, out_channels1, 1) if in_channels1 != out_channels1 else None

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False, padding=padding0,
                      dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),

            nn.Conv1d(out_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding0, dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(out_channels0, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),

            nn.Conv1d(out_channels1, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),
        )

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x0 = self.conv_block1(inputs)
        res0 = inputs if self.downsample0 is None else self.downsample0(inputs)
        out_0 = self.relu(x0 + res0)

        x1 = self.conv_block2(out_0)
        res1 = out_0 if self.downsample1 is None else self.downsample1(out_0)
        out_1 = self.relu(x1 + res1)

        out = out_1[:, :, -1]
        return out


######## ResNet ##############################################

class resnet18_regressor(nn.Module):
    def __init__(self, configs):
        super(resnet18_regressor, self).__init__()
        self.resnet = resnet18(configs)
    def forward(self, x_in):
        x = self.resnet(x_in)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out

######## GRU ##############################################

class gru_regressor(nn.Module):
    def __init__(self, configs):
        super(gru_regressor, self).__init__()

        self.gru = nn.GRU(input_size=configs.input_size, hidden_size=configs.hidden_size,
                          num_layers=configs.num_layers, batch_first=True, dropout=configs.dropout)

        self.fc = nn.Linear(configs.hidden_size, configs.output_size)

    def forward(self, x):
        output, h_n = self.gru(x)

        output = self.fc(output[:, -1, :])

        return output

######## Inception ##############################################

class inception_regressor(nn.Module):
    def __init__(self, configs):
        super(inception_regressor, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels[0], kernel_size=1),
            nn.BatchNorm1d(configs.mid_channels[0]),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels[1], kernel_size=1),
            nn.BatchNorm1d(configs.mid_channels[1]),
            nn.ReLU(),
            nn.Conv1d(configs.mid_channels[1], configs.mid_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(configs.mid_channels[2]),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels[3], kernel_size=1),
            nn.BatchNorm1d(configs.mid_channels[3]),
            nn.ReLU(),
            nn.Conv1d(configs.mid_channels[3], configs.mid_channels[4], kernel_size=5, padding=2),
            nn.BatchNorm1d(configs.mid_channels[4]),
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(configs.input_channels, configs.mid_channels[5], kernel_size=1),
            nn.BatchNorm1d(configs.mid_channels[5]),
            nn.ReLU()
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)
        branch4_output = self.branch4(x)

        outputs = [branch1_output, branch2_output, branch3_output, branch4_output]
        x = torch.cat(outputs, dim=1)

        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)

        return x_flat

######## Xception ##############################################

class xception_regressor(nn.Module):
    def __init__(self, configs):
        super(xception_regressor, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.mid_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(configs.mid_channels * 4),
            nn.ReLU()
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 4, configs.mid_channels * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(configs.mid_channels * 8),
            nn.ReLU()
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 8, configs.mid_channels * 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(configs.mid_channels * 16),
            nn.ReLU()
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat

######## mWDN ##############################################

class mwdn_regressor(nn.Module):
    def __init__(self, configs):
        super(mwdn_regressor, self).__init__()

        self.wavelet_layers = nn.ModuleList()

        for i in range(configs.num_levels):
            wavelet_block = nn.Sequential(
                nn.Conv1d(in_channels=configs.input_channels, out_channels=configs.mid_channels,
                          kernel_size=configs.kernel_size, stride=configs.stride, padding=(configs.kernel_size // 2)),
                nn.ReLU(),
                nn.Conv1d(in_channels=configs.mid_channels, out_channels=configs.output_channels,
                          kernel_size=configs.kernel_size, stride=configs.stride, padding=(configs.kernel_size // 2)),
                nn.ReLU()
            )
            self.wavelet_layers.append(wavelet_block)

        self.pooling = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x):
        wavelet_outputs = []

        for wavelet_layer in self.wavelet_layers:
            x = wavelet_layer(x)
            wavelet_outputs.append(x)

        fused_output = torch.cat(wavelet_outputs, dim=1)
        pooled_output = self.pooling(fused_output)

        return pooled_output


######## GRU-FCN ##############################################

class RCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RCN, self).__init__()

        # Define the structure of the RCN model
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Sequence modeling using GRU, taking only the
        # last moment of the hidden state as an output
        _, h_n = self.gru(x)
        # Pass implicit state into fully connected layer for classification
        x = self.fc(h_n.squeeze(0))
        return x

class FCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FCN, self).__init__()

        # Define the structure of the FCN model
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class gru_fcn_regressor(nn.Module):
    def __init__(self, rcn_input_size, rcn_hidden_size, rcn_output_size,
                 fcn_in_channels, fcn_out_channels, fcn_kernel_size):
        super(gru_fcn_regressor, self).__init__()

        # Define the structure of the GRU-FCN model
        self.rcn = RCN(rcn_input_size, rcn_hidden_size, rcn_output_size)
        self.fcn = FCN(fcn_in_channels, fcn_out_channels, fcn_kernel_size)

    def forward(self, x):
        x = self.rcn(x)
        x = self.fcn(x)
        return x


######## Transformer ##############################################

class transformer_regressor(nn.Module):
    def __init__(self, configs):
        super(transformer_regressor, self).__init__()

        self.embedding = nn.Embedding(configs.vocab_size, configs.embedding_dim)
        self.positional_encoding = PositionalEncoding(configs.embedding_dim, configs.max_seq_length)

        encoder_layer = nn.TransformerEncoderLayer(d_model=configs.embedding_dim, nhead=configs.num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.num_layers)

        self.fc = nn.Linear(configs.embedding_dim, configs.output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        positional_encoded = self.positional_encoding(embedded)
        transformer_output = self.transformer_encoder(positional_encoded)
        flatten = transformer_output.mean(dim=1)
        output = self.fc(flatten)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_length):
        super(PositionalEncoding, self).__init__()
        self.encoding = self.positional_encoding(embedding_dim, max_seq_length)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

    def positional_encoding(self, n_dims, max_seq_length):
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_dims, 2) * (-math.log(10000.0) / n_dims))
        encoding = torch.zeros(1, max_seq_length, n_dims)
        encoding[0, :, 0::2] = torch.sin(position * div_term)
        encoding[0, :, 1::2] = torch.cos(position * div_term)
        return encoding

######## KNN ##############################################

from sklearn.neighbors import KNeighborsRegressor

class knn_regressor(nn.Module):
    def __init__(self, configs):
        super(knn_regressor, self).__init__()

        self.knn = KNeighborsRegressor(n_neighbors=configs.n_neighbors)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        output = self.knn.predict(x_flat.detach().numpy())

        return torch.from_numpy(output)

######## SVR ##############################################

from sklearn.svm import SVR

class svr_regressor(nn.Module):
    def __init__(self, configs):
        super(svr_regressor, self).__init__()

        self.svr = SVR(kernel=configs.kernel, C=configs.C, epsilon=configs.epsilon)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        output = self.svr.predict(x_flat.detach().numpy())

        return torch.from_numpy(output)

######## RF ##############################################

from sklearn.ensemble import RandomForestRegressor

class rf_regressor(nn.Module):
    def __init__(self, configs):
        super(rf_regressor, self).__init__()

        self.random_forest = RandomForestRegressor(n_estimators=configs.n_estimators,
                                                   max_depth=configs.max_depth)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        output = self.random_forest.predict(x_flat.detach().numpy())

        return torch.from_numpy(output)


######## XGBoost ##############################################

import xgboost as xgb

class xgb_regressor(nn.Module):
    def __init__(self, configs):
        super(xgb_regressor, self).__init__()

        self.xgb = xgb.XGBRegressor(n_estimators=configs.n_estimators,
                                    max_depth=configs.max_depth,
                                    learning_rate=configs.learning_rate)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        dmatrix = xgb.DMatrix(x_flat.detach().numpy())
        output = self.xgb.predict(dmatrix)

        return torch.from_numpy(output)

######## Stacking ##############################################

from sklearn.ensemble import StackingRegressor

class stacking_regressor(nn.Module):
    def __init__(self, configs):
        super(stacking_regressor, self).__init__()

        self.stacking = StackingRegressor(estimators=configs.estimators,
                                          final_estimator=configs.final_estimator)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        output = self.stacking.predict(x_flat.detach().numpy())

        return torch.from_numpy(output)

######## MLP ##############################################

class mlp_regressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(mlp_regressor, self).__init__()

        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        output = self.mlp(x)
        return output

class codats_regressor(nn.Module):
    def __init__(self, configs):
        super(codats_regressor, self).__init__()
        model_output_dim = configs.features_len
        self.hidden_dim = configs.hidden_dim
        self.logits = nn.Sequential(
            nn.Linear(model_output_dim * configs.final_out_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, configs.num_classes))

    def forward(self, x_in):
        predictions = self.logits(x_in)
        return predictions


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out

#### Codes required by DANN ##############
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


#### Codes required by CDAN ##############
class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
