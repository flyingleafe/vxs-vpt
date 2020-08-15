import torch

from torch import nn
from torch.nn import functional as F
from catalyst import dl
from catalyst.dl import AlchemyLogger

CAE_CONFIGS = {
    'square-1': {
        'outer_kernel_size': (5,5),
        'strides': [(2,2), (2,2), (2,2), (2,2)]
    },
    'square-2': {
        'outer_kernel_size': (5,5),
        'strides': [(2,2), (2,2), (2,2), (4,4)]
    },
    'square-3': {
        'outer_kernel_size': (5,5),
        'strides': [(2,2), (2,2), (4,4), (4,4)]
    },
    'tall-1': {
        'outer_kernel_size': (5,3),
        'strides': [(2,2), (2,2), (2,2), (2,4)]
    },
    'tall-2': {
        'outer_kernel_size': (5,3),
        'strides': [(2,2), (2,2), (2,4), (2,4)]
    },
    'tall-3': {
        'outer_kernel_size': (5,3),
        'strides': [(2,2), (2,4), (2,4), (2,4)]
    },
    'tall-4': {
        'outer_kernel_size': (5,3),
        'strides': [(2,2), (2,4), (2,4), (4,4)]
    },
    'wide-1': {
        'outer_kernel_size': (3,5),
        'strides': [(2,2), (2,2), (2,2), (4,2)]
    },
    'wide-2': {
        'outer_kernel_size': (3,5),
        'strides': [(2,2), (2,2), (4,2), (4,2)]
    },
    'wide-3': {
        'outer_kernel_size': (3,5),
        'strides': [(2,2), (4,2), (4,2), (4,2)]
    },
    'wide-4': {
        'outer_kernel_size': (3,5),
        'strides': [(2,2), (4,2), (4,2), (4,4)]
    },
}

VAR_CAE_CONFIGS = {
    'square-1': {
        'outer_kernel_size': (5,5),
        'inner_size': (8, 8),
        'strides': [(1,1), (2,2), (2,2), (2,2)]
    },
    'square-2': {
        'outer_kernel_size': (5,5),
        'inner_size': (4, 4),
        'strides': [(1,1), (2,2), (2,2), (4,4)]
    },
    'square-3': {
        'outer_kernel_size': (5,5),
        'inner_size': (2, 2),
        'strides': [(1,1), (2,2), (4,4), (4,4)]
    },
    'tall-1': {
        'outer_kernel_size': (5,3),
        'inner_size': (8, 4),
        'strides': [(1,1), (2,2), (2,2), (2,4)]
    },
    'tall-2': {
        'outer_kernel_size': (5,3),
        'inner_size': (8, 2),
        'strides': [(1,1), (2,2), (2,4), (2,4)]
    },
    'tall-3': {
        'outer_kernel_size': (5,3),
        'inner_size': (8, 1),
        'strides': [(1,1), (2,4), (2,4), (2,4)]
    },
    'tall-4': {
        'outer_kernel_size': (5,3),
        'inner_size': (4, 1),
        'strides': [(1,1), (2,4), (2,4), (4,4)]
    },
    'wide-1': {
        'outer_kernel_size': (3,5),
        'inner_size': (4, 8),
        'strides': [(1,1), (2,2), (2,2), (4,2)]
    },
    'wide-2': {
        'outer_kernel_size': (3,5),
        'inner_size': (2, 8),
        'strides': [(1,1), (2,2), (4,2), (4,2)]
    },
    'wide-3': {
        'outer_kernel_size': (3,5),
        'inner_size': (1, 8),
        'strides': [(1,1), (4,2), (4,2), (4,2)]
    },
    'wide-4': {
        'outer_kernel_size': (3,5),
        'inner_size': (1, 4),
        'strides': [(1,1), (4,2), (4,2), (4,4)]
    },
}

class ConvAE_old(nn.Module):
    """
    Mehrabi's convolutional autoencoder architecture (with a mistake!)
    """
    def __init__(self, outer_kernel_size=(5,5), strides=[(2,2)]*4,
                 mid_kernel_size=9, padding_mode='zeros'):
        super().__init__()
        assert len(strides) == 4
        strides = [tuple(st) for st in strides]

        kernel_nums = [8, 16, 24, 32]
        kernels_strides = list(zip(kernel_nums, strides))

        enc_layers = []
        dec_layers = [nn.Conv2d(1, 1, (5,5), (1, 1), (2, 2))]
        prev_kernel_num = 1
        kernel_size = outer_kernel_size

        for (kernel_num, stride) in kernels_strides:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            enc_layers.append(nn.Conv2d(prev_kernel_num, kernel_num, kernel_size,
                                        stride, padding, padding_mode=padding_mode))
            enc_layers.append(nn.BatchNorm2d(kernel_num))
            enc_layers.append(nn.ReLU(inplace=True))

            dec_layers.append(nn.ReLU(inplace=True))
            dec_layers.append(nn.BatchNorm2d(prev_kernel_num))
            dec_layers.append(nn.Conv2d(kernel_num, prev_kernel_num, kernel_size, 1, padding,
                                        padding_mode=padding_mode))
            dec_layers.append(nn.UpsamplingBilinear2d(scale_factor=stride))

            prev_kernel_num = kernel_num
            # Mehrabi claims that they use (10, 10) kernels for internal layers,
            # but even kernel sizes do not play well with even paddings if we
            # try to preserve the (H, W) dimensions of the data. Should work as well,
            # anyway, right?
            kernel_size = (mid_kernel_size, mid_kernel_size)

        dec_layers.reverse()
        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y, z


class ConvAE(nn.Module):
    """
    Mehrabi's convolutional autoencoder architecture
    """
    def __init__(self, outer_kernel_size=(5,5), strides=[(2,2)]*4,
                 mid_kernel_size=9, padding_mode='zeros'):
        super().__init__()
        assert len(strides) == 4
        strides = [tuple(st) for st in strides]

        kernel_nums_enc = [(1, 8), (8, 16), (16, 24), (24, 32)]
        kernel_nums_dec = list(reversed([(32, 32), (32, 24), (24, 16), (16, 8)]))

        enc_layers = []
        dec_layers = [nn.Conv2d(8, 1, (5,5), (1, 1), (2, 2))]
        kernel_size = outer_kernel_size

        for i, stride in enumerate(strides):
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            
            inp_channels_enc, out_channels_enc = kernel_nums_enc[i]
            enc_layers.append(nn.Conv2d(inp_channels_enc, out_channels_enc, kernel_size,
                                        stride, padding, padding_mode=padding_mode))
            enc_layers.append(nn.BatchNorm2d(out_channels_enc))
            enc_layers.append(nn.ReLU(inplace=True))

            inp_channels_dec, out_channels_dec = kernel_nums_dec[i]
            dec_layers.append(nn.ReLU(inplace=True))
            dec_layers.append(nn.BatchNorm2d(out_channels_dec))
            dec_layers.append(nn.Conv2d(inp_channels_dec, out_channels_dec, kernel_size, 1, padding,
                                        padding_mode=padding_mode))
            dec_layers.append(nn.UpsamplingBilinear2d(scale_factor=stride))
            kernel_size = (mid_kernel_size, mid_kernel_size)

        dec_layers.reverse()
        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

    def representation(self, x):
        return self.encoder(x)
        
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y, z
    

class ConvVAE(ConvAE):
    def __init__(self, inner_size=(8, 8), **kwargs):
        super().__init__(**kwargs)
        self.inner_size = inner_size
        self.flattened_size = 32 * inner_size[0] * inner_size[1]
        self.mu_lin = nn.Linear(self.flattened_size, self.flattened_size)
        self.logvar_lin = nn.Linear(self.flattened_size, self.flattened_size)
        self.pre_decode_lin = nn.Linear(self.flattened_size, self.flattened_size)
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size(), device=std.device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        h = h.view(h.shape[0], -1)
        mu, logvar = self.mu_lin(h), self.logvar_lin(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.pre_decode_lin(z)
        z = z.view(z.shape[0], 32, *self.inner_size)
        return self.decoder(z), mu, logvar

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layers=[512, 512], num_classes=4):
        layers = [nn.Linear(input_dim, hidden_layers[0]), nn.ReLU(inplace=True)]
        for i in range(len(hidden_layers)-1):
            layers.append += [nn.Linear(hidden_layers[i], hidden_layers[i+1]), nn.ReLU(inplace=True)] 

        layers.append += [nn.Linear(hidden_layers[-1], num_classes)]
        self.ff = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.ff(x)
        
def get_CAE_model(config_type, ckp_path=None, old_model=False):
    ModelClass = ConvAE_old if old_model else ConvAE
    model = ModelClass(**CAE_CONFIGS[config_type])
    if ckp_path is not None:
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        model.load_state_dict(
            torch.load(ckp_path, map_location=map_location)['model_state_dict'])
    return model


def get_CVAE_model(config_type, ckp_path=None):
    model = ConvVAE(**VAR_CAE_CONFIGS[config_type])
    if ckp_path is not None:
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        model.load_state_dict(
            torch.load(ckp_path, map_location=map_location)['model_state_dict'])
    return model


class ConvAERunner(dl.Runner):
    def _handle_batch(self, batch):
        x = batch          # ignore the raw waveform
        y, z = self.model(x)
        loss = F.mse_loss(y, x)
        self.batch_metrics = {
            'loss': loss
        }

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

KL_LOSS_WEIGHT = 0.01
            
class ConvVAERunner(dl.Runner):
#     def __init__(self, kl_loss_weight=0.1):
#         super().__init__()
#         self.kl_loss_weight = kl_loss_weight
    
    def _handle_batch(self, batch):
        x = batch
        y, mu, logvar = self.model(x)
        mse_loss = F.mse_loss(y, x)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = mse_loss + KL_LOSS_WEIGHT * kld_loss
        
        self.batch_metrics = {
            'loss': loss,
            'mse_loss': mse_loss,
            'kld_loss': kld_loss
        }
        
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
class ClassifierRunner(dl.Runner):
    def _handle_batch(self, batch):
        x, y = batch
        pred = self.model(x)
        ce_loss = F.cross_entropy(pred, y)
        self.batch_metrics = {
            'ce_loss': ce_loss, 
        }
        
        if self.is_train_loader:
            ce_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
def alchemy_logger(group, name):
    return AlchemyLogger(
        token="1da39325aff8856a81d7ad0250c9f921",
        project="default",
        experiment=name,
        group=group,
        log_on_epoch_end=False
    )
