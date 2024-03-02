from torch import nn, cat


class Encoder(nn.Module):

    def __init__(self, filters=32, in_channels=3, n_block=4, kernel_size=(3, 3), batch_norm=True, padding='same'):
        super().__init__()
        self.filter = filters
        for i in range(n_block):
            out_ch = filters * 2 ** i
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = filters * 2 ** (i - 1)

            if padding == 'same':
                pad = kernel_size[0] // 2
            else:
                pad = 0
            model = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),nn.LeakyReLU(inplace=True)]
            if batch_norm:
                model += [nn.BatchNorm2d(num_features=out_ch)]
            model += [nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                      nn.LeakyReLU(inplace=True)]
            if batch_norm:
                model += [nn.BatchNorm2d(num_features=out_ch)]
            self.add_module('encoder%d' % (i + 1), nn.Sequential(*model))
            conv = [nn.Conv2d(in_channels=in_ch * 3, out_channels=out_ch, kernel_size=1), nn.LeakyReLU(inplace=True)]
            self.add_module('conv1_%d' % (i + 1), nn.Sequential(*conv))

    def forward(self, x):
        skip = []
        output = x
        res = None
        i = 0
        for name, layer in self._modules.items():
            if i % 2 == 0:
                output = layer(output)
                skip.append(output)
            else:
                if i > 1:
                    output = cat([output, res], 1)
                    output = layer(output)
                output = nn.MaxPool2d(kernel_size=(2,2))(output)
                res = output
            i += 1
        return output, skip


class Bottleneck(nn.Module):
    def __init__(self, filters=32, n_block=4, depth=4, kernel_size=(3,3)):
        super().__init__()
        out_ch = filters * 2 ** n_block
        in_ch = filters * 2 ** (n_block - 1)
        for i in range(depth):
            dilate = 2 ** i
            model = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=dilate,
                          dilation=dilate),nn.LeakyReLU(inplace=True)]
            self.add_module('bottleneck%d' % (i + 1), nn.Sequential(*model))
            if i == 0:
                in_ch = out_ch

    def forward(self, x):
        bottleneck_output = 0
        output = x
        for _, layer in self._modules.items():
            output = layer(output)
            bottleneck_output += output
        return bottleneck_output


class Decoder(nn.Module):
    def __init__(self, filters=32, n_block=4, kernel_size=(3, 3), batch_norm=True, padding='same', drop=False):
        super().__init__()
        self.n_block = n_block
        if padding == 'same':
            pad = kernel_size[0] // 2
        else:
            pad = 0
        for i in reversed(range(n_block)):
            out_ch = filters * 2 ** i
            in_ch = 2 * out_ch
            model = [nn.UpsamplingNearest2d(scale_factor=2),
                     nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
                               padding=pad)]
            self.add_module('decoder1_%d' % (i + 1), nn.Sequential(*model))

            model = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                     nn.LeakyReLU(inplace=True)]
            if drop:
                model += [nn.Dropout(.5)]
            if batch_norm:
                model += [nn.BatchNorm2d(num_features=out_ch)]
            model += [nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                      nn.LeakyReLU(inplace=True)]
            if batch_norm:
                model += [nn.BatchNorm2d(num_features=out_ch)]
            self.add_module('decoder2_%d' % (i + 1), nn.Sequential(*model))

    def forward(self, x, skip):
        i = 0
        output = x
        for _, layer in self._modules.items():
            output = layer(output)
            if i % 2 == 0:
                output = cat([skip.pop(), output], 1)
            i += 1
        return output


class Segmentation_model_Point(nn.Module):
    def __init__(self, filters=32, in_channels=3, n_block=4, bottleneck_depth=4, n_class=4, pointnet=False,fc_inch=81, heinit=False, multicuda=False, extpn=False, batchnorm=True):
        super().__init__()
        self.encoder = Encoder(filters=filters, in_channels=in_channels, n_block=n_block, batch_norm=batchnorm)
        self.bottleneck = Bottleneck(filters=filters, n_block=n_block, depth=bottleneck_depth)
        self.decoder = Decoder(filters=filters, n_block=n_block, drop=False, batch_norm=batchnorm)
        self.classifier = nn.Conv2d(in_channels=filters, out_channels=n_class, kernel_size=(1, 1))

    def forward(self, x, features_out=True, print_shape=False):
        output, skip = self.encoder(x)
        output_bottleneck = self.bottleneck(output)
        output = self.decoder(output_bottleneck, skip)
        output = self.classifier(output)
        if features_out:
            return output
        else:
            return output
