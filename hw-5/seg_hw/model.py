import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn_relu(in_planes, out_planes, kernel=3, stride=1, padding=1):
    net = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=1),
                        nn.BatchNorm2d(num_features=out_planes),
                        nn.ReLU(inplace=True))
    return net

class Stacked2ConvsBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Stacked2ConvsBlock, self).__init__()
        self.blocks = nn.Sequential(conv_bn_relu(in_planes, out_planes), 
                                    conv_bn_relu(out_planes, out_planes))

    def forward(self, net):
        net = self.blocks(net)
        return net


class UpSamplingBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(UpSamplingBlock, self).__init__()
        # понижаем число каналов. я прочитал, что это принято делать именно с помощью транспонированной свёртки
        self.upsample = nn.ConvTranspose2d(in_planes, in_planes, kernel_size=2, stride=2)
        # а потом стакаем с симметричным слоем из левой половины "параболы"
        # ясно, что число каналов входной карты при этом удваивается
        self.convolve = Stacked2ConvsBlock(2 * in_planes, out_planes)

    def forward(self, right_net, left_net):
        # нужно, чтобы состыковать число каналов
        right_net = self.upsample(right_net)
        net = torch.cat([left_net, right_net], dim=1)
        net = self.convolve(net)
        return net

    
class DownSamplingBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownSamplingBlock, self).__init__()
        self.blocks = nn.Sequential(nn.MaxPool2d(2, 2), # раньше мы делали по-другому, но пишут, что лучше через пулинг
                                    Stacked2ConvsBlock(in_planes, out_planes))

    def forward(self, net):
        net = self.blocks(net)
        return net


class SegmenterModel(nn.Module):
    def __init__(self):
        super(SegmenterModel, self).__init__()
        self.init_ch = 64 # число каналов после первой свёртки
        self.n_levels = 3 # число уровней до "основания" параболы
        self.init_conv = Stacked2ConvsBlock(3, 64)
        
        # Спуск. Я честно пытался написать нормальный список слоёв,
        # потому что генерировать их руками это очень убого,
        # но у меня в результате просто ничего не работало с 
        # очень странными ошибками (в духе NotImplementedError), поэтому я сдался
#         self.downsamplers = nn.ModuleList(
#             DownSamplingBlock(64 * 2**i, 
#                               64 * 2**(i + 1))
#             for i in range(self.n_levels + 1)
#         )
        # Тяжело
        self.downsample_1 = DownSamplingBlock(64, 128)
        self.downsample_2 = DownSamplingBlock(128, 256)
        self.downsample_3 = DownSamplingBlock(256, 512)
        self.downsample_4 = DownSamplingBlock(512, 1024)
         
        # в середине есть блок без пары с 1024 каналами
        # с ним конкатенировать некого, потому просто свернём его        
        self.upconv = Stacked2ConvsBlock(self.init_ch * 16, self.init_ch * 8)
        
        # Подъём. Аналогично.
#         self.upsamplers = nn.ModuleList(
#             UpSamplingBlock(64 * 2**i, 
#                             64 * 2**(i - 1))
#             for i in range(self.n_levels, 0, -1)
#         )
#         self.upsamplers.add_module(UpSamplingBlock(self.init_ch, self.init_ch))

        self.upsample_1 = UpSamplingBlock(512, 256)
        self.upsample_2 = UpSamplingBlock(256, 128)
        self.upsample_3 = UpSamplingBlock(128, 64)
        # чтобы учесть входной слой после самой первой свёртки и согласовать размерности
        self.upsample_4 = UpSamplingBlock(64, 64)
        
        # предсказание
        self.agg_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        net0 = self.init_conv(x) # 3 --> 64
       
        net1 = self.downsample_1(net0) # 64 --> 128
        net2 = self.downsample_2(net1) # 128 --> 256
        net3 = self.downsample_3(net2) # 256 --> 512
        net = self.downsample_4(net3) # 512 --> 1024
        
        net = self.upconv(net) # 1024 --> 512
        
        net = self.upsample_1(net, net3) # 512 --> 256
        net = self.upsample_2(net, net2) # 256 --> 128
        net = self.upsample_3(net, net1) # 128 --> 64
        net = self.upsample_4(net, net0) # 64 --> 64
        
        net = self.agg_conv(net) # 64 --> 1
        return net
    
    def predict(self, x):
        # на вход подаётся одна картинка, а не батч, поэтому так
        y = self.forward(x.unsqueeze(0).cuda())
        return (y > 0).squeeze(0).squeeze(0).float().cuda()