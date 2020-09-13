import os
import shutil
from datetime import datetime

import torch
import torch.utils.data as data
from torch import nn, optim
from torch.backends import cudnn
from tqdm import tqdm
from . import config
from .data.dataset import MLTDataSet
from . import utils
from .encoder.encoder_lstm import LSTMEncoder
from .decoder.decoder_attention import DecoderAttentionLSTM


class MeanMeter:
    def __init__(self):
        self.n = 0
        self.value = 0

    def update(self, curr_value):
        self.value += curr_value
        self.n += 1

    def update_all(self, curr_value, curr_n):
        self.value += curr_value
        self.n += curr_n

    @property
    def metric(self):
        return self.value / self.n


def run(encoder, decoder, encoder_optimizer, decoder_optimizer,
        loader: data.DataLoader, train=True, epochs=0):
    answs = []
    indxs = []
    if not train:
        encoder.eval()
        decoder.eval()
    else:
        encoder.train()
        decoder.train()
    tq = tqdm(loader, desc='Running Model-Epoch:{}'.format(epochs))
    start_token, end_token = loader.dataset.get_start_end()
    start_token = torch.LongTensor([start_token], device=config.device)
    end_token = torch.LongTensor([end_token], device=config.device)
    criterion = nn.NLLLoss()
    loss_meter = MeanMeter()
    for indx, inp, out in tq:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        inp = torch.LongTensor(inp).to(config.device)
        out = torch.LongTensor(out).to(config.device)

        encoder_output, hidden = encoder(inp.unsqueeze(dim=0), None)
        output = decoder(start_token, end_token, hidden, out.unsqueeze(-1), encoder_output)
        loss = criterion(output, out)
        if train:
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
        else:
            answer = output.cpu().argmax(dim=1)
            answs.append([ans.item() for ans in answer])
            indxs.append(indx)
        loss_meter.update(loss.item())
        tq.set_postfix(loss='{:.4f}'.format(loss.item()), running_loss='{:.4f}'.format(loss_meter.metric))
    return answs, indxs, loss_meter.metric


def save_net(model, is_best, filename):
    filename = "{}_{}_{}".format(filename, config.epochs, config.batch_size)
    checkpoint_name = filename + '_checkpoint.pth.tar'
    best_name = filename + '_best_model.pth.tar'
    torch.save(model, checkpoint_name)
    print('Saved model checkpoint at-{}'.format(checkpoint_name))
    if is_best:
        shutil.copyfile(checkpoint_name, best_name)
        print('Saved best model at-{}'.format(best_name))


def execute():
    dataset = MLTDataSet(config.data_path, config.vocab_path)
    loader = data.DataLoader(dataset=dataset, num_workers=config.data_worker, batch_size=config.batch_size)
    print("Config :")
    print(utils.print_obj(config))
    if not os.path.exists(config.dir):
        os.mkdir(config.dir)
    filename = '{}/{}'.format(config.dir, datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
    cudnn.benchmark = True
    input_size = dataset.get_input_size()
    encoder = LSTMEncoder(input_size[0], config.hidden_dim, config.num_layers, config.drop_prob,
                          batch_first=True).to(config.device)
    decoder = DecoderAttentionLSTM(input_size[1], config.hidden_dim, config.drop_prob, config.num_layers,
                                   batch_first=True).to(config.device)
    print("Encoder Net:")
    print(encoder)
    print("Decoder Net:")
    print(decoder)
    encoder_optimizer = optim.Adam(params=encoder.parameters(), lr=config.lr)
    decoder_optimizer = optim.Adam(params=decoder.parameters(), lr=config.lr)

    best_loss = 1e6
    start_epoch = 0
    if config.resume:
        print('Loading save model from-{}'.format(config.resume))
        model = torch.load(config.resume)
        start_epoch = model['epoch']
        best_loss = model['loss']
        encoder_optimizer.load_state_dict(model['encoder_optimizer'])
        decoder_optimizer.load_state_dict(model['decoder_optimizer'])
        encoder.load_state_dict(model['encoder_state_dict'])
        decoder.load_state_dict(model['decoder_state_dict'])
        print('Resume model with loss-{}, epochs-{}'.format(best_loss, start_epoch))

    if config.train:
        losses = []
        for i in range(start_epoch, config.epochs):
            _, _, loss = run(encoder, decoder,
                             encoder_optimizer, decoder_optimizer,
                             loader, config.train, i)

            is_best = loss < best_loss
            best_loss = loss
            save_net({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'epoch': i,
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                'config': config,
                'loss': best_loss}, is_best, filename)
            losses.append(loss)
        utils.show_plot(losses)
        print("Training done!")
    if config.eval:
        answ, indx, loss = run(encoder, decoder,
                               encoder_optimizer, decoder_optimizer,
                               loader, train=False)
        dataset.save_evaluated(answ, indx, loss, eval_filename=config.val_path)
        print("Evaluation done with loss-{}".format(loss))
