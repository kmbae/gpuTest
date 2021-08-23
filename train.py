import torch
import torchvision
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--batch_size', type=int, default='128',
                            help='# of batch size')
args = parser.parse_args()
net = torchvision.models.alexnet().cuda()
net = torch.nn.DataParallel(net)
batch = args.batch_size
while 1:
    x = torch.randn(batch, 3, 224, 224).cuda()
    y = torch.ones(batch, 1000).cuda()

    y_hat = net(x)

    loss = (y - y_hat).mean()
    loss.backward()




