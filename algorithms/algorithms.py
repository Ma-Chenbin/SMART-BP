import torch
import torch.nn as nn
import numpy as np
import itertools    

from models.models import regressor, cnn_regressor, gru_regressor, resnet18_regressor, tcn_regressor, \
    inception_regressor, Discriminator, xception_regressor, mwdn_regressor, gru_fcn_regressor, \
    transformer_regressor, codats_regressor, mlp_regressor, ReverseLayerF, RandomLayer
from models.loss import MMD_loss, CORAL, VAT, LMMD_loss, HoMM_loss, NTXentLoss, SupConLoss
from models.ema import EMA
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import torch.nn. functional as F


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs, backbone):
        super(Algorithm, self).__init__()
        self.configs = configs

        self.mse_loss = nn.MSELoss()
        self.feature_extractor = backbone(configs)
        self.regressor = regressor(configs)
        self.network = nn.Sequential(self.feature_extractor, self.regressor)


    # update function is common to all algorithms
    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            
            # training loop 
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_reg_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_reg_loss'].avg
                best_model = deepcopy(self.network.state_dict())


            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        
        last_model = self.network.state_dict()

        return last_model, best_model
    
    # train loop vary from one method to another
    def training_epoch(self, *args, **kwargs):
        raise NotImplementedError


########################  CNN  ########################

class CNN(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.regressor = cnn_regressor(configs)
        self.network = nn.Sequential(self.feature_extractor, self.regressor)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        self.hparams = hparams
        self.device = device

        self.domain_regressor = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_regressor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            src_reg_loss = self.mse_loss(src_pred.squeeze(), src_y)

            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_regressor(src_feat_reversed)
            src_domain_loss = self.mse_loss(src_domain_pred, domain_label_src.long())

            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_regressor(trg_feat_reversed)
            trg_domain_loss = self.mse_loss(trg_domain_pred, domain_label_trg.long())

            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_reg_loss_wt"] * src_reg_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses = {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_reg_loss': src_reg_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  GRU  ########################

class GRU(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.regressor = gru_regressor(configs)
        self.network = nn.Sequential(self.feature_extractor, self.regressor)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        self.hparams = hparams
        self.device = device

        self.domain_regressor = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_regressor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            src_reg_loss = self.mse_loss(src_pred.squeeze(), src_y)

            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_regressor(src_feat_reversed)
            src_domain_loss = self.mse_loss(src_domain_pred, domain_label_src.long())

            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_regressor(trg_feat_reversed)
            trg_domain_loss = self.mse_loss(trg_domain_pred, domain_label_trg.long())

            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_reg_loss_wt"] * src_reg_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses = {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_reg_loss': src_reg_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  ResNet  ########################

class ResNet(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.regressor = resnet18_regressor(configs)
        self.network = nn.Sequential(self.feature_extractor, self.regressor)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        self.hparams = hparams
        self.device = device

        self.domain_regressor = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_regressor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            src_reg_loss = self.mse_loss(src_pred.squeeze(), src_y)

            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_regressor(src_feat_reversed)
            src_domain_loss = self.mse_loss(src_domain_pred, domain_label_src.long())

            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_regressor(trg_feat_reversed)
            trg_domain_loss = self.mse_loss(trg_domain_pred, domain_label_trg.long())

            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_reg_loss_wt"] * src_reg_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses = {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_reg_loss': src_reg_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  TCN  ########################

class TCN(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.regressor = tcn_regressor(configs)
        self.network = nn.Sequential(self.feature_extractor, self.regressor)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        self.hparams = hparams
        self.device = device

        self.domain_regressor = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_regressor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            src_reg_loss = self.mse_loss(src_pred.squeeze(), src_y)

            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_regressor(src_feat_reversed)
            src_domain_loss = self.mse_loss(src_domain_pred, domain_label_src.long())

            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_regressor(trg_feat_reversed)
            trg_domain_loss = self.mse_loss(trg_domain_pred, domain_label_trg.long())

            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_reg_loss_wt"] * src_reg_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses = {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_reg_loss': src_reg_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  Inception  ########################

class Inception(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.regressor = inception_regressor(configs)
        self.network = nn.Sequential(self.feature_extractor, self.regressor)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        self.hparams = hparams
        self.device = device

        self.domain_regressor = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_regressor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            src_reg_loss = self.mse_loss(src_pred.squeeze(), src_y)

            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_regressor(src_feat_reversed)
            src_domain_loss = self.mse_loss(src_domain_pred, domain_label_src.long())

            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_regressor(trg_feat_reversed)
            trg_domain_loss = self.mse_loss(trg_domain_pred, domain_label_trg.long())

            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_reg_loss_wt"] * src_reg_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses = {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_reg_loss': src_reg_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  Xception  ########################

class Xception(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.regressor = xception_regressor(configs)
        self.network = nn.Sequential(self.feature_extractor, self.regressor)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        self.hparams = hparams
        self.device = device

        self.domain_regressor = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_regressor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            src_reg_loss = self.mse_loss(src_pred.squeeze(), src_y)

            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_regressor(src_feat_reversed)
            src_domain_loss = self.mse_loss(src_domain_pred, domain_label_src.long())

            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_regressor(trg_feat_reversed)
            trg_domain_loss = self.mse_loss(trg_domain_pred, domain_label_trg.long())

            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_reg_loss_wt"] * src_reg_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses = {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_reg_loss': src_reg_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  mWDN  ########################

class mWDN(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.regressor = mwdn_regressor(configs)
        self.optimizer = torch.optim.Adam(
            self.regressor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        self.hparams = hparams
        self.device = device

        self.domain_regressor = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_regressor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_pred = self.regressor(src_x)

            trg_pred = self.regressor(trg_x)

            src_reg_loss = self.mse_loss(src_pred.squeeze(), src_y)

            src_feat_reversed = ReverseLayerF.apply(src_pred, alpha)
            src_domain_pred = self.domain_regressor(src_feat_reversed)
            src_domain_loss = self.mse_loss(src_domain_pred, domain_label_src.long())

            trg_feat_reversed = ReverseLayerF.apply(trg_pred, alpha)
            trg_domain_pred = self.domain_regressor(trg_feat_reversed)
            trg_domain_loss = self.mse_loss(trg_domain_pred, domain_label_trg.long())

            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_reg_loss_wt"] * src_reg_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses = {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_reg_loss': src_reg_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  GRU-FCN  ########################

class GRU_RCN(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.model = gru_fcn_regressor(configs["rcn_input_size"], configs["rcn_hidden_size"], configs["rcn_output_size"],
                             configs["fcn_in_channels"], configs["fcn_out_channels"], configs["fcn_kernel_size"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams["learning_rate"])
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=hparams['step_size'],
                                                            gamma=hparams['lr_decay'])

        self.hparams = hparams
        self.device = device

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        for step, (src_x, src_y) in enumerate(src_loader):
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            self.optimizer.zero_grad()

            src_pred = self.model(src_x)

            src_loss = self.mse_loss(src_pred.squeeze(), src_y)

            src_loss.backward()
            self.optimizer.step()

            avg_meter['Loss'].update(src_loss.item(), src_x.size(0))

        self.lr_scheduler.step()

########################  Transformer  ########################

class Transformer(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.regressor = transformer_regressor(configs)
        self.network = nn.Sequential(self.feature_extractor, self.regressor)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        self.hparams = hparams
        self.device = device

        self.domain_regressor = Discriminator(configs)

        self.optimizer_disc = torch.optim.Adam(
            self.domain_regressor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            src_reg_loss = self.mse_loss(src_pred.squeeze(), src_y)

            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_regressor(src_feat_reversed)
            src_domain_loss = self.mse_loss(src_domain_pred, domain_label_src.long())

            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_regressor(trg_feat_reversed)
            trg_domain_loss = self.mse_loss(trg_domain_pred, domain_label_trg.long())

            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_reg_loss_wt"] * src_reg_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses = {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_reg_loss': src_reg_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  KNN  ########################

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

class KNNRegressor:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        return rmse


########################  SVR  ########################

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

class SVRRegressor:
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        return rmse


########################  RF  ########################

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class RFRegressor:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestRegressor(n_estimators=self.n_estimators,
                                           max_depth=self.max_depth)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        return rmse

########################  XGBoost  ########################

import xgboost as xgb
from sklearn.metrics import mean_squared_error

class XGBRegressor:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = xgb.XGBRegressor(n_estimators=self.n_estimators,
                                      max_depth=self.max_depth)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        return rmse


########################  Stacking  ########################

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

class StackingRegressor:
    def __init__(self):
        self.models = [
            ('lr', LinearRegression()),
            ('knn', KNeighborsRegressor()),
            ('dt', DecisionTreeRegressor())
        ]
        self.stacking_regressor = StackingRegressor(estimators=self.models,
                                                   final_estimator=LinearRegression())

    def train(self, X_train, y_train):
        self.stacking_regressor.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.stacking_regressor.predict(X_test)
        return y_pred

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
        return rmse


########################  MLP  ########################

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate, weight_decay, step_size, lr_decay, device):
        self.model = mlp_regressor(input_dim, hidden_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.lr_scheduler = StepLR(self.optimizer, step_size=step_size, gamma=lr_decay)
        self.device = device

    def train(self, X_train, y_train, num_epochs, batch_size):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            self.model.train()

            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            self.lr_scheduler.step()

    def predict(self, X_test):
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test).cpu().numpy()
        return y_pred


########################  MLP  ########################

class NO_ADAPT(Algorithm):
    """
    Lower bound: train on source and test on target.
    """
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):
        for src_x, src_y in src_loader:
            
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)
            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            src_reg_loss = self.mse_loss(src_pred, src_y)

            loss = src_reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Src_reg_loss': src_reg_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()
    

########################  TARGET_ONLY  ########################

class TARGET_ONLY(Algorithm):
    """
    Upper bound: train on target and test on target.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        for trg_x, trg_y in trg_loader:

            trg_x, trg_y = trg_x.to(self.device), trg_y.to(self.device)

            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.regressor(trg_feat)

            trg_reg_loss = self.mse_loss(trg_pred, trg_y)

            loss = trg_reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Trg_reg_loss': trg_reg_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  Deep_Coral  ########################

class Deep_Coral(Algorithm):
    """
    Sun B, Saenko K. Deep coral: Correlation alignment for deep domain adaptation[C]//Computer
    Vision–ECCV 2016 Workshops: Amsterdam, The Netherlands, October 8-10 and 15-16, 2016,
    Proceedings, Part III 14. Springer International Publishing, 2016: 443-450.
    """
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # correlation alignment loss
        self.coral = CORAL()


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        # add if statement

        if len(src_loader) > len(trg_loader):
            joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        else:
            joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            src_reg_loss = self.mse_loss(src_pred, src_y)

            trg_feat = self.feature_extractor(trg_x)

            coral_loss = self.coral(src_feat, trg_feat)

            loss = self.hparams["coral_wt"] * coral_loss + \
                self.hparams["src_reg_loss_wt"] * src_reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Total_loss': loss.item(), 'Src_reg_loss': src_reg_loss.item(),
                    'coral_loss': coral_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  MMDA  ########################

class MMDA(Algorithm):
    """
    Rahman M M, Fookes C, Baktashmotlagh M, et al. On minimum
    discrepancy estimation for deep domain adaptation[J].
    Domain Adaptation for Visual Understanding, 2020: 81-94.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd = MMD_loss()
        self.coral = CORAL()


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            src_reg_loss = self.mse_loss(src_pred, src_y)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            src_reg_loss_ = self.mse_loss(src_pred, src_y)

            trg_feat = self.feature_extractor(trg_x)

            coral_loss = self.coral(src_feat, trg_feat)
            mmd_loss = self.mmd(src_feat, trg_feat)
            cond_ent_loss = self.cond_ent(trg_feat)

            loss = self.hparams["coral_wt"] * coral_loss + \
                self.hparams["mmd_wt"] * mmd_loss + \
                self.hparams["cond_ent_wt"] * cond_ent_loss + \
                self.hparams["src_reg_loss_wt"] * src_reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'Coral_loss': coral_loss.item(), 'MMD_loss': mmd_loss.item(),
                    'cond_ent_wt': cond_ent_loss.item(), 'Src_reg_loss': src_reg_loss.item(),
                    'Src_reg_loss_': src_reg_loss_.item()}
            
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  DANN  ########################

class DANN(Algorithm):
    """
    Ganin Y, Ustinova E, Ajakan H, et al. Domain-adversarial training of neural networks[J].
    The journal of machine learning research, 2016, 17(1): 2096-2030.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        
        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Domain Discriminator
        self.domain_regressor = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_regressor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):
        # Combine dataloaders
        # Method 1 (min len of both domains)
        # joint_loader = enumerate(zip(src_loader, trg_loader))

        # Method 2 (max len of both domains)
        # joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            
            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            # Task regression  Loss
            src_reg_loss = self.mse_loss(src_pred.squeeze(), src_y)

            # Domain regression loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_regressor(src_feat_reversed)
            src_domain_loss = self.mse_loss(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_regressor(trg_feat_reversed)
            trg_domain_loss = self.mse_loss(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_reg_loss_wt"] * src_reg_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_reg_loss': src_reg_loss.item()}
           
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  CDAN  ########################

class CDAN(Algorithm):
    """
    Long M, Cao Z, Wang J, et al. Conditional adversarial domain adaptation[J].
    Advances in neural information processing systems, 2018, 31.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment Losses

        self.domain_regressor = Discriminator(configs)
        self.random_layer = RandomLayer([configs.features_len * configs.final_out_channels, configs.num_classes],
                                        configs.features_len * configs.final_out_channels)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_regressor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"])

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            # prepare true domain labels
            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

            # source features and predictions
            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            # target features and predictions
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.regressor(trg_feat)

            # concatenate features and predictions
            feat_concat = torch.cat((src_feat, trg_feat), dim=0)
            pred_concat = torch.cat((src_pred, trg_pred), dim=0)

            # Domain regression loss
            feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1)).detach()
            disc_prediction = self.domain_regressor(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
            disc_loss = self.mse_loss(disc_prediction, domain_label_concat)

            # update Domain regression
            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()

            # prepare fake domain labels for training the feature extractor
            domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
            domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

            # Repeat predictions after updating discriminator
            feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1))
            disc_prediction = self.domain_regressor(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))

            # loss of domain discriminator according to fake labels
            domain_loss = self.mse_loss(disc_prediction, domain_label_concat)

            # Task regression  Loss
            src_reg_loss = self.mse_loss(src_pred.squeeze(), src_y)

            # Regression loss.
            loss_trg_cent = self.criterion_cond(trg_pred)

            # total loss
            loss = self.hparams["src_reg_loss_wt"] * src_reg_loss + self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["cond_ent_wt"] * loss_trg_cent

            # update feature extractor
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_reg_loss': src_reg_loss.item(),
                    'cond_ent_loss': loss_trg_cent.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)
        self.lr_scheduler.step()


########################  DIRT  ########################

class DIRT(Algorithm):
    """
    Shu R, Bui H H, Narui H, et al. A dirt-t approach to unsupervised domain adaptation[J].
    arXiv preprint arXiv:1802.08735, 2018.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.vat_loss = VAT(self.network, device).to(device)
        self.ema = EMA(0.998)
        self.ema.register(self.network)

        # Discriminator
        self.domain_regressor = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_regressor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
       
    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            # prepare true domain labels
            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            # target features and predictions
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.regressor(trg_feat)

            # concatenate features and predictions
            feat_concat = torch.cat((src_feat, trg_feat), dim=0)

            # Domain regression loss
            disc_prediction = self.domain_regressor(feat_concat.detach())
            disc_loss = self.mse_loss(disc_prediction, domain_label_concat)

            # update Domain regression
            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()

            # prepare fake domain labels for training the feature extractor
            domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
            domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

            # Repeat predictions after updating discriminator
            disc_prediction = self.domain_regressor(feat_concat)

            # loss of domain discriminator according to fake labels
            domain_loss = self.mse_loss(disc_prediction, domain_label_concat)

            # Task regression  Loss
            src_reg_loss = self.mse_loss(src_pred.squeeze(), src_y)

            # Regression loss.
            loss_trg_cent = self.criterion_cond(trg_pred)

            # Virual advariarial training loss
            loss_src_vat = self.vat_loss(src_x, src_pred)
            loss_trg_vat = self.vat_loss(trg_x, trg_pred)
            total_vat = loss_src_vat + loss_trg_vat
            # total loss
            loss = self.hparams["src_reg_loss_wt"] * src_reg_loss + self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["cond_ent_wt"] * loss_trg_cent + self.hparams["vat_loss_wt"] * total_vat

            # update exponential moving average
            self.ema(self.network)

            # update feature extractor
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_reg_loss': src_reg_loss.item(),
                    'cond_ent_loss': loss_trg_cent.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  DSAN  ########################

class DSAN(Algorithm):
    """
    Zhu Y, Zhuang F, Wang J, et al. Deep subdomain adaptation network for image classification[J].
    IEEE transactions on neural networks and learning systems, 2020, 32(4): 1713-1722.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Alignment losses
        self.loss_LMMD = LMMD_loss(device=device, class_num=configs.num_classes).to(device)

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            # extract source features
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.regressor(trg_feat)

            # calculate lmmd loss
            domain_loss = self.loss_LMMD.get_loss(src_feat, trg_feat, src_y, torch.nn.functional.softmax(trg_pred, dim=1))

            # calculate source regression loss
            src_reg_loss = self.mse_loss(src_pred, src_y)

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_reg_loss_wt"] * src_reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Total_loss': loss.item(), 'LMMD_loss': domain_loss.item(), 'Src_reg_loss': src_reg_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  HoMM  ########################

class HoMM(Algorithm):
    """
    Chen C, Fu Z, Chen Z, et al. Homm: Higher-order moment matching for unsupervised domain
    adaptation[C]//Proceedings of the AAAI conference on artificial intelligence.
    2020, 34(04): 3422-3429.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # aligment losses
        self.coral = CORAL()
        self.HoMM_loss = HoMM_loss()

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            # extract source features
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            
            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.regressor(trg_feat)

            # calculate source regression loss
            src_reg_loss = self.mse_loss(src_pred, src_y)

            # calculate lmmd loss
            domain_loss = self.HoMM_loss(src_feat, trg_feat)

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_reg_loss_wt"] * src_reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'HoMM_loss': domain_loss.item(), 'Src_reg_loss': src_reg_loss.item()}
            
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  DDC  ########################

class DDC(Algorithm):
    """
    Tzeng E, Hoffman J, Zhang N, et al. Deep domain confusion: Maximizing for domain invariance[J].
    arXiv preprint arXiv:1412.3474, 2014.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd_loss = MMD_loss()

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
            # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)

            # calculate source regression loss
            src_reg_loss = self.mse_loss(src_pred, src_y)

            # calculate mmd loss
            domain_loss = self.mmd_loss(src_feat, trg_feat)

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_reg_loss_wt"] * src_reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Total_loss': loss.item(), 'MMD_loss': domain_loss.item(), 'Src_reg_loss': src_reg_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  CoDATS  ########################

class CoDATS(Algorithm):
    """
    Wilson G, Doppa J R, Cook D J. Multi-source deep domain adaptation with weak supervision
    for time-series sensor data[C]//Proceedings of the 26th ACM SIGKDD international conference
    on knowledge discovery & data mining. 2020: 1768-1778.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # we replace the original classifier with codats the regressor
        # remember to use same name of self.regressor, as we use it for the model evaluation
        self.regressor = codats_regressor(configs)
        self.network = nn.Sequential(self.feature_extractor, self.regressor)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


        # Domain regressor
        self.domain_regressor = Discriminator(configs)

        self.optimizer_disc = torch.optim.Adam(
            self.domain_regressor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
        
            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            # Task regression  Loss
            src_reg_loss = self.mse_loss(src_pred.squeeze(), src_y)

            # Domain regression loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_regressor(src_feat_reversed)
            src_domain_loss = self.mse_loss(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_regressor(trg_feat_reversed)
            trg_domain_loss = self.mse_loss(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_reg_loss_wt"] * src_reg_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses = {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_reg_loss': src_reg_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  AdvSKM  ########################

class AdvSKM(Algorithm):
    """
    Liu Q, Xue H. Adversarial Spectral Kernel Matching for Unsupervised Time Series
    Domain Adaptation[C]//IJCAI. 2021: 2744-2750.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd_loss = MMD_loss()
        self.optimizer_disc = torch.optim.Adam(
            self.AdvSKM_embedder.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features
            
            src_feat = self.feature_extractor(src_x)
            src_pred = self.regressor(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)

            source_embedding_disc = self.AdvSKM_embedder(src_feat.detach())
            target_embedding_disc = self.AdvSKM_embedder(trg_feat.detach())
            mmd_loss = - self.mmd_loss(source_embedding_disc, target_embedding_disc)
            mmd_loss.requires_grad = True

            # update discriminator
            self.optimizer_disc.zero_grad()
            mmd_loss.backward()
            self.optimizer_disc.step()

            # calculate source regression loss
            src_reg_loss = self.mse_loss(src_pred, src_y)

            # domain loss.
            source_embedding_disc = self.AdvSKM_embedder(src_feat)
            target_embedding_disc = self.AdvSKM_embedder(trg_feat)

            mmd_loss_adv = self.mmd_loss(source_embedding_disc, target_embedding_disc)
            mmd_loss_adv.requires_grad = True

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * mmd_loss_adv + \
                self.hparams["src_reg_loss_wt"] * src_reg_loss

            # update optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'MMD_loss': mmd_loss_adv.item(), 'Src_reg_loss': src_reg_loss.item()}
            for key, val in losses.items():
                    avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


########################  mWDN  ########################

class mWDN(Algorithm):
    """
    Wang J, Wang Z, Li J, et al. Multilevel wavelet decomposition network
    for interpretable time series analysis[C]//Proceedings of the 24th
    ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018: 2437-2446.
    """
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # feature_length for regressor
        configs.features_len = 1
        self.regressor = regressor(configs)
        # feature length for feature extractor
        configs.features_len = 1
        self.network = nn.Sequential(self.feature_extractor, self.regressor)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features

            # Extract features
            src_feature = self.feature_extractor(src_x)
            tgt_feature = self.feature_extractor(trg_x)

            # source regression loss
            y_pred = self.regressor(src_feature)
            src_reg_loss = self.mse_loss(y_pred, src_y)

            # MMD loss
            domain_loss_intra = self.mmd_loss(src_struct=src_feature,
                                            tgt_struct=tgt_feature, weight=self.hparams['domain_loss_wt'])

            # total loss
            total_loss = self.hparams['src_reg_loss_wt'] * src_reg_loss + domain_loss_intra

            # remove old gradients
            self.optimizer.zero_grad()
            # calculate gradients
            total_loss.backward()
            # update the weights
            self.optimizer.step()

            losses =  {'Total_loss': total_loss.item(), 'MMD_loss': domain_loss_intra.item(),
                    'Src_reg_loss': src_reg_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()
    def mmd_loss(self, src_struct, tgt_struct, weight):
        delta = torch.mean(src_struct - tgt_struct, dim=-2)
        loss_value = torch.norm(delta, 2) * weight
        return loss_value


########################  CoTMix  ########################

class CoTMix(Algorithm):
    """
    Eldele E, Ragab M, Chen Z, et al. Cotmix: Contrastive domain adaptation
    for time-series via temporal mixup[J]. arXiv preprint arXiv:2212.01555, 2022.
    """
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

         # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.contrastive_loss = NTXentLoss(device, hparams["batch_size"], 0.2, True)
        self.sup_contrastive_loss = SupConLoss(device)

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features

            # ====== Temporal Mixup =====================
            src_dominant, trg_dominant = self.temporal_mixup(src_x, trg_x)

            # ====== Source =====================
            self.optimizer.zero_grad()

            # Src original features
            src_orig_feat = self.feature_extractor(src_x)
            src_orig_logits = self.regressor(src_orig_feat)

            # Target original features
            trg_orig_feat = self.feature_extractor(trg_x)
            trg_orig_logits = self.regressor(trg_orig_feat)

            # -----------  The two main losses
            # MSE loss
            src_reg_loss = self.mse_loss(src_orig_logits, src_y)
            loss = src_reg_loss * round(self.hparams["src_reg_weight"], 2)

            # Target MSE loss
            trg_mse_loss = self.mse_loss(trg_orig_logits)
            loss += trg_mse_loss * round(self.hparams["trg_mse_weight"], 2)

            # -----------  Auxiliary losses
            # Extract source-dominant mixup features.
            src_dominant_feat = self.feature_extractor(src_dominant)
            src_dominant_logits = self.regressor(src_dominant_feat)

            # supervised contrastive loss on source domain side
            src_concat = torch.cat([src_orig_logits.unsqueeze(1), src_dominant_logits.unsqueeze(1)], dim=1)
            src_supcon_loss = self.sup_contrastive_loss(src_concat, src_y)
            loss += src_supcon_loss * round(self.hparams["src_supCon_weight"], 2)

            # Extract target-dominant mixup features.
            trg_dominant_feat = self.feature_extractor(trg_dominant)
            trg_dominant_logits = self.regressor(trg_dominant_feat)

            # Unsupervised contrastive loss on target domain side
            trg_con_loss = self.contrastive_loss(trg_orig_logits, trg_dominant_logits)
            loss += trg_con_loss * round(self.hparams["trg_cont_weight"], 2)

            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(),
                    'src_reg_loss': src_reg_loss.item(),
                    'trg_mse_loss': trg_mse_loss.item(),
                    'src_supcon_loss': src_supcon_loss.item(),
                    'trg_con_loss': trg_con_loss.item()
                    }
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()           

    def temporal_mixup(self,src_x, trg_x):
        
        mix_ratio = round(self.hparams["mix_ratio"], 2)
        temporal_shift = self.hparams["temporal_shift"]
        h = temporal_shift // 2  # half

        src_dominant = mix_ratio * src_x + (1 - mix_ratio) * \
                    torch.mean(torch.stack([torch.roll(trg_x, -i, 2) for i in range(-h, h)], 2), 2)

        trg_dominant = mix_ratio * trg_x + (1 - mix_ratio) * \
                    torch.mean(torch.stack([torch.roll(src_x, -i, 2) for i in range(-h, h)], 2), 2)
        
        return src_dominant, trg_dominant
    

########################  MCD  ########################

# Untied Approaches: (MCD)
class MCD(Algorithm):
    """
    Saito K, Watanabe K, Ushiku Y, et al. Maximum classifier discrepancy
    for unsupervised domain adaptation[C]//Proceedings of the IEEE conference
    on computer vision and pattern recognition. 2018: 3723-3732.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.feature_extractor = backbone(configs)
        self.regressor = regressor(configs)
        self.regressor2 = regressor(configs)

        self.network = nn.Sequential(self.feature_extractor, self.regressor)


        # optimizer and scheduler
        self.optimizer_fe = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # optimizer and scheduler
        self.optimizer_c1 = torch.optim.Adam(
            self.regressor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # optimizer and scheduler
        self.optimizer_c2 = torch.optim.Adam(
            self.regressor2.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.lr_scheduler_fe = StepLR(self.optimizer_fe, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler_c1 = StepLR(self.optimizer_c1, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler_c2 = StepLR(self.optimizer_c2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd_loss = MMD_loss()

    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            
            # source pretraining loop 
            self.pretrain_epoch(src_loader, avg_meter)

            # training loop 
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_reg_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_reg_loss'].avg
                best_model = deepcopy(self.network.state_dict())


            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        
        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader, avg_meter):
        for src_x, src_y in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)
          
            src_feat = self.feature_extractor(src_x)
            src_pred1 = self.regressor(src_feat)
            src_pred2 = self.regressor2(src_feat)

            src_reg_loss1 = self.mse_loss(src_pred1, src_y)
            src_reg_loss2 = self.mse_loss(src_pred2, src_y)

            loss = src_reg_loss1 + src_reg_loss2

            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()
            self.optimizer_fe.zero_grad()

            loss.backward()

            self.optimizer_c1.step()
            self.optimizer_c2.step()
            self.optimizer_fe.step()

            
            losses = {'Src_reg_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
            

            # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred1 = self.regressor(src_feat)
            src_pred2 = self.regressor2(src_feat)

            # source losses
            src_reg_loss1 = self.mse_loss(src_pred1, src_y)
            src_reg_loss2 = self.mse_loss(src_pred2, src_y)
            loss_s = src_reg_loss1 + src_reg_loss2
            

            # Freeze the feature extractor
            for k, v in self.feature_extractor.named_parameters():
                v.requires_grad = False
            # update C1 and C2 to maximize their difference on target sample
            trg_feat = self.feature_extractor(trg_x) 
            trg_pred1 = self.regressor(trg_feat.detach())
            trg_pred2 = self.regressor2(trg_feat.detach())


            loss_dis = self.discrepancy(trg_pred1, trg_pred2)

            loss = loss_s - loss_dis
            
            loss.backward()
            self.optimizer_c1.step()
            self.optimizer_c2.step()

            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()
            self.optimizer_fe.zero_grad()

            # Freeze the regressors
            for k, v in self.regressor.named_parameters():
                v.requires_grad = False
            for k, v in self.regressor2.named_parameters():
                v.requires_grad = False
                        # Freeze the feature extractor
            for k, v in self.feature_extractor.named_parameters():
                v.requires_grad = True
            # update feature extractor to minimize the discrepaqncy on target samples
            trg_feat = self.feature_extractor(trg_x)        
            trg_pred1 = self.regressor(trg_feat)
            trg_pred2 = self.regressor2(trg_feat)


            loss_dis_t = self.discrepancy(trg_pred1, trg_pred2)
            domain_loss = self.hparams["domain_loss_wt"] * loss_dis_t 

            domain_loss.backward()
            self.optimizer_fe.step()

            self.optimizer_fe.zero_grad()
            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()


            losses =  {'Total_loss': loss.item(), 'MMD_loss': domain_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler_fe.step()
        self.lr_scheduler_c1.step()
        self.lr_scheduler_c2.step()

    def discrepancy(self, out1, out2):

        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))
