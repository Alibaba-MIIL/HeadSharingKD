from __future__ import print_function

import os
import argparse
import socket
import time

# import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_vanilla as train, validate


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', ])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')
    parser.add_argument('--path_th', type=str, default=None, help='model head - path')

    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './outputs/save/models'
        opt.tb_path = './outputs/save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        if 'S:' in segments[0]: 
            return segments[0].replace('S:', '')
        else:
            return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    local_path = "/download/model/"
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    sd = torch.load(model_path)['model']
    sd = {x.replace("model_s.", ""): y for x, y in sd.items()}
    load_status = model.load_state_dict(sd, strict=False)
    print("Load state dict status: ", load_status)
    print('==> done')
    return model


def main():
    best_acc = 0

    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = model_dict[opt.model](num_classes=n_cls)

    if opt.path_th is not None:
        # Load head model
        model_th = load_teacher(opt.path_th, n_cls)

        if model_th.fc.weight.shape != model.fc.weight.shape:
            raise print("Last fully-connected layers of student and teacher are not match")
        model.fc.weight.requires_grad, model.fc.bias.requires_grad = False, False
        model_th.fc.weight.requires_grad, model_th.fc.bias.requires_grad = False, False
        model.fc.weight.copy_(model_th.fc.weight)
        model.fc.bias.copy_(model_th.fc.bias)
        print("Check weights:", torch.all(model.fc.weight == model_th.fc.weight))
        print("Check bias:", torch.all(model.fc.bias == model_th.fc.bias))
        print("Check Grad-weights:", model.fc.weight.requires_grad)
        print("Check Grad-bias:", model.fc.bias.requires_grad)

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        print("train_acc. epoch %d: %0.2f " % (epoch, train_acc))
        print("train_loss. epoch %d: %0.2f " % (epoch, train_loss))
        # logger.log_value('train_acc', train_acc, epoch)
        # logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)


        print("test_acc. epoch %d: %0.2f" % (epoch, test_acc))
        print("test_acc_top5. epoch %d: %0.2f" % (epoch, test_acc_top5))
        print("test_loss. epoch %d: %0.2f" % (epoch, test_loss))
        # logger.log_value        print("test_acc. epoch %d: %0.2f" % (epoch, test_acc))
        #         print("test_acc_top5. epoch %d: %0.2f" % (epoch, test_acc_top5))
        #         print("test_loss. epoch %d: %0.2f" % (epoch, test_loss))('test_acc', test_acc, epoch)
        # logger.log_value('test_acc_top5', test_acc_top5, epoch)
        # logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
