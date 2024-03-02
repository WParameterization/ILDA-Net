import importlib
from torch import nn, cat
from .decoder import Aux_Module
#################model.load_state_dict(state_dict, strict=False)
def get_syncbn():
    # return nn.BatchNorm2d
    return nn.SyncBatchNorm
norm_layer = get_syncbn()

class ModelBuilder(nn.Module):
    def __init__(self, net_cfg):
        super(ModelBuilder, self).__init__()
        self._sync_bn = net_cfg["sync_bn"]
        self._num_classes = net_cfg["num_classes"]

        self.encoder = self._build_encoder(net_cfg["encoder"])
        self.decoder = self._build_decoder(net_cfg["decoder"])

        self._use_auxloss = True if net_cfg.get("aux_loss", False) else False
        self.fpn = True if net_cfg["encoder"]["kwargs"].get("fpn", False) else False
        if self._use_auxloss:
            cfg_aux = net_cfg["aux_loss"]
            self.loss_weight = cfg_aux["loss_weight"]
            self.auxor = Aux_Module(cfg_aux["aux_plane"], self._num_classes, self._sync_bn)

    def _build_encoder(self, enc_cfg):
        enc_cfg["kwargs"].update({"sync_bn": self._sync_bn})
        encoder = self._build_module(enc_cfg["type"], enc_cfg["kwargs"])
        return encoder

    def _build_decoder(self, dec_cfg):
        dec_cfg["kwargs"].update( {"in_planes": self.encoder.get_outplanes(),"sync_bn": self._sync_bn, "num_classes": self._num_classes, } )
        decoder = self._build_module(dec_cfg["type"], dec_cfg["kwargs"])
        return decoder

    def _build_module(self, mtype, kwargs):
        module_name, class_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    def forward(self, x):
            feat = self.encoder(x)
            outs = self.decoder(feat)
            return outs


class Segmentation_model_Point(nn.Module):
    def __init__(self, filters=32, in_channels=3, n_block=4, bottleneck_depth=4, batchnorm=True):
        super().__init__()
        self.encoder = Encoder(filters=filters, in_channels=in_channels, n_block=n_block, batch_norm=batchnorm)
        self.bottleneck = Bottleneck(filters=filters, n_block=n_block, depth=bottleneck_depth)
        self.decoder = Decoder(filters=filters, n_block=n_block, drop=False, batch_norm=batchnorm)
        self.end=end(filters=filters)
    def forward(self, x):
        output, skip = self.encoder(x)
        output_bottleneck = self.bottleneck(output)
        output1 = self.decoder(output_bottleneck, skip)
        end=self.end(output1)
        return end


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
                #print("ooooooooooo",output.shape)
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
            model = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=dilate,  dilation=dilate),nn.LeakyReLU(inplace=True)]
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

            model = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),nn.LeakyReLU(inplace=True)]
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


class end(nn.Module):
    def __init__(self, filters=32):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels=filters, out_channels=9, kernel_size=(1, 1))
        self.representation = nn.Sequential(nn.Conv2d(32, 256, kernel_size=3, stride=1, padding=1, bias=True),
                                            norm_layer(256),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout2d(0.1),
                                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                                            norm_layer(256),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout2d(0.1),
                                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
                                            )

    def forward(self, x):
        output = self.classifier(x)
        res = {"pred": output}
        res["rep"] = self.representation(x)
        return res