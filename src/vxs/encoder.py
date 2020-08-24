import torch
import os

from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from catalyst import dl
from catalyst.dl import AlchemyLogger

from vxs import constants, ListDataset

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

CAE_INNER_SIZES = {
    24: {   # time length of input spectrogram
        'square-1': 512,
        'square-2': 128,
        'square-3': 64,
        'tall-1': 256,
        'tall-2': 256,
        'tall-3': 256,
        'tall-4': 128,
        'wide-1': 256,
        'wide-2': 128,
        'wide-3': 64,
        'wide-4': 32,
    },
    32: {
        'square-1': 512,
        'square-2': 128,
        'square-3': 64,
        'tall-1': 256,
        'tall-2': 256,
        'tall-3': 256,
        'tall-4': 128,
        'wide-1': 256,
        'wide-2': 128,
        'wide-3': 64,
        'wide-4': 32, 
    },
    64: {
        'square-1': 1024,
        'square-2': 256,
        'square-3': 64,
        'tall-1': 512,
        'tall-2': 256,
        'tall-3': 256,
        'tall-4': 128,
        'wide-1': 512,
        'wide-2': 256,
        'wide-3': 128,
        'wide-4': 64,
    },
    128: {
        'square-1': 2048,
        'square-2': 512,
        'square-3': 128,
        'tall-1': 1024,
        'tall-2': 512,
        'tall-3': 256,
        'tall-4': 128,
        'wide-1': 1024,
        'wide-2': 512,
        'wide-3': 256,
        'wide-4': 128,
    }
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
            
            # Mehrabi claims that they use (10, 10) kernels for internal layers,
            # but even kernel sizes do not play well with even paddings if we
            # try to preserve the (H, W) dimensions of the data. Should work as well,
            # anyway, right?
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

    
class ConvAEClassifier(nn.Module):
    def __init__(self, encoder, input_dim, hidden_layers=[512, 512, 512], dropout_prob=0.2, num_classes=4):
        super().__init__()
        self.encoder = encoder
        layers = [nn.Linear(input_dim, hidden_layers[0]), nn.ReLU(inplace=True), nn.Dropout(p=dropout_prob)]
        for i in range(len(hidden_layers)-1):
            layers += [nn.Linear(hidden_layers[i], hidden_layers[i+1]), nn.ReLU(inplace=True), nn.Dropout(p=dropout_prob)] 

        layers += [nn.Linear(hidden_layers[-1], num_classes)]
        self.ff = nn.Sequential(*layers)
    
    def head_parameters(self):
        return self.ff.parameters()
    
    def forward(self, x):
        x = self.encoder(x)
        return self.ff(x.view(x.shape[0], -1))

def load_model_state_dict(model, ckp_path):
    if ckp_path is not None:
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        model.load_state_dict(
            torch.load(ckp_path, map_location=map_location)['model_state_dict'])

def get_CAE_model(config_type, ckp_path=None):
    model = ConvAE(**CAE_CONFIGS[config_type])
    load_model_state_dict(model, ckp_path)
    return model

def get_CVAE_model(config_type, ckp_path=None):
    model = ConvVAE(**VAR_CAE_CONFIGS[config_type])
    load_model_state_dict(model, ckp_path)
    return model

def get_CAE_classifier(config_type, inp_time_len=24, ckp_path=None, unsupervised=True, **kwargs):
    cae = ConvAE(**CAE_CONFIGS[config_type])
    if unsupervised:
        load_model_state_dict(cae, ckp_path)
    model = ConvAEClassifier(cae.encoder, CAE_INNER_SIZES[inp_time_len][config_type], **kwargs)
    if not unsupervised:
        load_model_state_dict(model, ckp_path)
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
            'loss': ce_loss, 
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

# Caching routines

def save_sgram_cache(dataset, filename):
    tensors = []
    for i in tqdm(range(len(dataset)), desc='Pre-caching spectrograms'):
        tensors.append(dataset[i])
    torch.save(tensors, filename)
    return tensors

def item_to_tensor(item):
    if type(item) == tuple:
        x, y = item
        return x.float(), torch.tensor(constants.EVENT_CLASS_IXS[y])
    else:
        return item.float()

def save_or_load_spectrograms(common_set, save_file_name):
    if os.path.isfile(save_file_name):
        print(f'Found saved pre-processed spectrograms: {save_file_name}')
        tensors = torch.load(save_file_name)
        if len(tensors) != len(common_set):
            print(f'Cached dataset length is invalid (expected {len(common_set)}, got {len(tensors)}), re-caching')
            tensors = save_sgram_cache(common_set, save_file_name)
    else:
        tensors = save_sgram_cache(common_set, save_file_name)
    
    tensors = [item_to_tensor(t) for t in tensors]
    tensors_set = ListDataset(tensors)
    return tensors_set