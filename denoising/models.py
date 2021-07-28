import torch


class LinearFourier2d(torch.nn.Module):
    def __init__(self, image_size, log):
        super(LinearFourier2d, self).__init__()

        self.log = log

        c, h, w = image_size
        self.register_parameter(name='fourier_filter', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))
        torch.nn.init.ones_(self.fourier_filter)

    def forward(self, x):
        w = torch.nn.ReLU()(self.fourier_filter.repeat(x.shape[0], 1, 1, 1).to(x.device))

        rft_x = torch.rfft(x, signal_ndim=3, normalized=True, onesided=True)
        init_spectrum = torch.sqrt(torch.pow(rft_x[..., 0], 2) + torch.pow(rft_x[..., 1], 2))
        
        if self.log:
            spectrum = torch.exp(w * torch.log(1 + init_spectrum)) - 1
        else:
            spectrum = w * init_spectrum

        irf = torch.irfft(torch.stack([rft_x[..., 0] * spectrum / (init_spectrum + 1e-16),
                                       rft_x[..., 1] * spectrum / (init_spectrum + 1e-16)], dim=-1),
                          signal_ndim=3, normalized=True, onesided=True, signal_sizes=x.shape[1:])

        return irf


class DnCNN(torch.nn.Module):
    def __init__(self, n_channels, num_features, num_layers, image_size, fourier_params=None):
        super(DnCNN, self).__init__()

        self.fourier_params = fourier_params
        if fourier_params is not None:
            if self.fourier_params['fourier_layer'] == 'linear':
                self.fl = LinearFourier2d((n_channels, image_size[0], image_size[1]), log=False)
            if self.fourier_params['fourier_layer'] == 'linear_log':
                self.fl = LinearFourier2d((n_channels, image_size[0], image_size[1]), log=True)

        layers = [torch.nn.Sequential(torch.nn.Conv2d(n_channels, num_features, kernel_size=3, stride=1, padding=1),
                                      torch.nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(torch.nn.Sequential(torch.nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                              torch.nn.BatchNorm2d(num_features),
                                              torch.nn.ReLU(inplace=True)))

        layers.append(torch.nn.Conv2d(num_features, n_channels, kernel_size=3, padding=1))
        self.layers = torch.nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, input):
        x = input
        if self.fourier_params is not None:
            x = self.fl(x)

        residual = self.layers(x)

        return torch.nn.Sigmoid()(input - residual)
