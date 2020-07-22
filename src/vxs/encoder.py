import torch

from torch import nn

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

class ConvAE(nn.Module):
    """
    Mehrabi's convolutional autoencoder architecture
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
        
def get_CAE_model(config_type, ckp_path=None):
    model = ConvAE(**CAE_CONFIGS[config_type])
    if ckp_path is not None:
        model.load_state_dict(torch.load(ckp_path)['model_state_dict'])
    return model