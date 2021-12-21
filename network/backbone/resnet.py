import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    #'resnet50': 'https://doc-14-5c-docs.googleusercontent.com/docs/securesc/51gl3j65bf9sqo3mtsoo3dqo88csai3k/6cgfh4ljqgak01q272gdfmj1ggarhq5k/1618843050000/11387100185987159300/11387100185987159300/1EEror78LmTMaT7yNPzWzVqgC9bf6UcLa?e=download&authuser=0&nonce=7kq6t0cc92cqa&user=11387100185987159300&hash=o39ronbjeop7r0e78fooet46ia42n125',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress,checkpoint_root=None, **kwargs):
    model = ResNet(block, layers, **kwargs)
    pretrained = True
    if pretrained:
        #state_dict = load_state_dict_from_url(model_urls[arch],
        #                                      progress=progress)
        #'''pretrained model from mocov2 with 800epochs'''
        #state_dict = torch.load("./pre_models/resnet50_mocov2_wo_aut_wo_8192_0.2_800es.pth")
        ##print(state_dict)
        ##exit()
        #state_dict = state_dict['state_dict'] 
        #state_dict =  {k.split('module.encoder_q.')[1]: v for k, v in state_dict.items()if(k in state_dict and 'module.encoder_q' in k)}
        #state_dict.popitem()
        #state_dict.popitem()
        #state_dict.popitem()
        #state_dict.popitem()
        #for k,v in state_dict.items():
        #    print(k)
        #model_dict = model.state_dict()
        #model_dict.update(state_dict)
        #model.load_state_dict(model_dict)
        #print("successfully loaded the offline model ResNet50 from mocov2(query=8192_aut_0.2)!") 
        #exit()
        '''pretrained model from official url with supervised model'''
        #state_dict = torch.load("./pre_models/resnet50_supervisor.pth")
        ##print(state_dict)
        ##exit()
        #model.load_state_dict(state_dict)
        '''load model from simclr-800epochs-vissl'''
        #model_dict = model.state_dict()
        #state_dict = torch.load('./pre_models/resnet50_simCLR_800es.pth')
        #state_dict = state_dict['classy_state_dict']['base_model']['model']['trunk']
        #state_dict = {k.split('_feature_blocks.')[1]:v for k,v in state_dict.items()}
        #state_dict.popitem()
        #model_dict.update(state_dict)
        #model.load_state_dict(model_dict)
        #print("successfully loaded the offline model ResNet50 from SimCLR!") 
        '''load model from deepclusterv2-800epochs-vissl'''
        #model_dict = model.state_dict()
        #state_dict = torch.load('./pre_models/resnet50_deepclusterv2_800es.pth')
        #state_dict = {k.split('module.')[1]:v for k,v in state_dict.items() if(k in state_dict and 'pro' not in k)}
        #state_dict.popitem()
        #model_dict.update(state_dict)
        #model.load_state_dict(model_dict)
        #print("successfully loaded the offline model ResNet50 from deepclusterv2!") 
        '''load model from SwAV-800epochs-vissl'''
        #model_dict = model.state_dict()
        #state_dict = torch.load('./pre_models/resnet50_SwAV_800es.pth')
        #state_dict = state_dict['classy_state_dict']['base_model']['model']['trunk']
        #state_dict = {k.split('_feature_blocks.')[1]:v for k,v in state_dict.items()}
        #state_dict.popitem()
        #model_dict.update(state_dict)
        #model.load_state_dict(model_dict)
        #model_dict = model.state_dict()
        #state_dict = torch.load('./pre_models/resnet50_SwAV_800es.pth')
        ## state_dict = state_dict['classy_state_dict']['base_model']['model']['trunk']
        #state_dict = {k.split('module.')[1]:v for k,v in state_dict.items()if(k in state_dict and 'pro' not in k)}
        #state_dict.popitem()
        #model_dict.update(state_dict)
        #model.load_state_dict(model_dict)
        #print("successfully loaded the offline model ResNet50 from SwAV!") 
        '''load model from dino'''
        #model_dict = model.state_dict()
        #state_dict = torch.load('./pre_models/resnet50_dino.pth')
        #for k,v in state_dict.items():
        #    print(k)
        #model_dict.update(state_dict)
        #model.load_state_dict(model_dict)
        #print("successfully loaded the offline model ResNet50 from dino!") 
        '''load model from obow'''
        #model_dict = model.state_dict()
        #state_dict = torch.load('./pre_models/resnet50_obow_200es.pth')
        #state_dict = state_dict['network']
        #state_dict.popitem()
        #state_dict.popitem()
        #model_dict.update(state_dict)
        #model.load_state_dict(model_dict)
        #print("successfully loaded the offline model ResNet50 from obow!") 
        '''load model from obow'''
        #model_dict = model.state_dict()
        #state_dict = torch.load('./pre_models/resnet50_obow_2048_wst_coco_super_crop_s4.pth')
        #state_dict = state_dict['network']
        #state_dict.popitem()
        #state_dict.popitem()
        #model_dict.update(state_dict)
        #model.load_state_dict(model_dict)
        #print("successfully loaded the offline model ResNet50 from obow(2048_aut coco with pretrained model and super crop min(h,w)>100 but use padding ->s4!")
        '''load model from depth estimation BTS'''
        #model_dict = model.state_dict()
        #state_dict = torch.load('./pre_models/model_144000_best_rms_6.60517_p9_obow_resnet_imagenet_depth.pth')
        #state_dict = state_dict['model']
        #state_dict =  {k.split('module.encoder.base_model.')[1]: v for k, v in state_dict.items()if(k in state_dict and 'module.encoder.base_model.' in k)}
        #for k,v in state_dict.items():
        #    print(k)
        #model_dict.update(state_dict)
        #model.load_state_dict(model_dict)
        #print("successfully loaded the pretrained model from model_144000_best_rms_6.60517_p9_obow_resnet_imagenet_depth.pth!")
        '''load model from GAN'''
        model_dict = model.state_dict()
        state_dict = torch.load(checkpoint_root)
        state_dict = {k.split('resnet_features.')[1]: v for k, v in state_dict.items() if
                      (k in state_dict and 'resnet_features.' in k)}
        for i in range(10):
            state_dict.popitem()
        for k, v in state_dict.items():
            print(k)
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        print("Model restored from %s" % checkpoint_root)

    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
