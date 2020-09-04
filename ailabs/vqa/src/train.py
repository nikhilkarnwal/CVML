import os
import shutil
from datetime import datetime

import torch
from torch.backends import cudnn

from .model.vqanet import VQANet
from torch import nn
import torch.utils.data as data
from typing import Type

from tqdm import tqdm

from ... import config
from .metrics import compute_accuracy, M3, MeanMeter, MovingMeanMeter
from .dataset.vqadataset import vqa_dataset

device = 'cpu'


def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5 ** (float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def run(net: Type[nn.Module], loader: Type[data.DataLoader],
        optimizer, m3, train=False, epoch=0, total_iterations=0):
    process = "train"
    if train:
        net.train()
        loss_m3 = m3.add_meter('{}-{}'.format(process, 'loss'), MovingMeanMeter(momentum=0.99))
        acc_m3 = m3.add_meter('{}-{}'.format(process, 'acc'), MovingMeanMeter(momentum=0.99))
    else:
        net.eval()
        process = "test"
        answ = []
        idxs = []
        accs = []
        loss_m3 = m3.add_meter('{}-{}'.format(process, 'loss'), MeanMeter())
        acc_m3 = m3.add_meter('{}-{}'.format(process, 'acc'), MeanMeter())

    tq = tqdm(loader, desc='{} {:03d}'.format(process, epoch))

    loss_module = nn.LogSoftmax().to(config.device)

    for v, q, a, indx, q_len in tq:
        v, q, a, q_len = v.to(device), q.to(device), a.to(device), q_len.to(device)
        out = net(v, q, q_len)
        loss = -loss_module(out)
        loss = (loss * a / 10).sum(dim=1).mean()
        acc = compute_accuracy(out, a)
        if train:
            update_learning_rate(optimizer, total_iterations)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_iterations += 1
        else:
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer.view(-1))
            idxs.append(indx.view(-1).clone())
            accs.append(acc.view(-1))

        loss_m3.update(loss.item())
        for a in acc:
            acc_m3.update(a.item())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_m3.metric), acc=fmt(acc_m3.metric))
        if not train:
            answ = list(torch.cat(answ, dim=0))
            accs = list(torch.cat(accs, dim=0))
            idxs = list(torch.cat(idxs, dim=0))
        return answ, accs, idxs


def save_net(model, is_best, filename):
    filename = "{}_{}_{}".format(filename, config.epochs, config.batch_size)
    checkpoint_name = filename + '_checkpoint.pth.tar'
    best_name = filename + '_best_model.pth.tar'
    torch.save(model, checkpoint_name)
    if is_best:
        shutil.copyfile(checkpoint_name, best_name)


def execute():
    if not os.path.exists(config.dir):
        os.mkdir(config.dir)
    filename = '{}/{}'.format(config.dir, datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
    cudnn.benchmark = True

    train_loader = vqa_dataset.get_loader(config, train=True)
    val_loader = vqa_dataset.get_loader(config, train=False, val=True)
    net = nn.DataParallel(
        VQANet(train_loader.dataset.num_tokens, config.embedding,
               config.lstm_dim, config.lstm_layer, config.visual_dim,
               config.attn_mid_dim, config.glimpses, config.classes)).to(config.device)

    m3 = M3()
    total_itr = 0
    best_loss = 0.0
    start_epoch = 0
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad])
    if config.resume:
        model = torch.load(config.resume)
        start_epoch = model['epoch']
        total_itr = model['total_iteration']
        best_loss = model['loss']
        optimizer.load_state_dict(model['optimizer'])
        net.load_state_dict(model['state_dict'])
        m3.decode(model['m3'])

    for i in range(start_epoch, config.epochs):
        res_t = run(net, train_loader, optimizer, m3, True, i, total_itr)
        res_e = run(net, val_loader, optimizer, m3, False, i, total_itr)
        is_best = False
        if best_loss >= m3.get_meter('{}-{}'.format('train', 'loss')).metric:
            is_best = True
            best_loss = m3.get_meter('{}-{}'.format('train', 'loss')).metric
        save_net({
            'state_dict': net.state_dict(),
            'epoch': i,
            'optimizer': optimizer.state_dict(),
            'm3': m3.to_dict(),
            'total_iteration': total_itr,
            'config': config,
            'loss': best_loss,
            'eval': {
                'answ': res_e[0],
                'acc': res_e[1],
                'idx': res_e[2]}}, is_best, filename)

    print('Training done! with best loss {}'.format(best_loss))
