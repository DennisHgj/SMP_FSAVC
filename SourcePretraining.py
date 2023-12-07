import argparse
import os.path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader.av_FSL import Dataset_AVC
from models.E_AVLmodel import E_AVLmodel

from utils.epoch_fucntion import train_one_epoch, val_one_epoch
from utils.functions import setup_seed, collate_fn_caption
from utils.prototype_util import calculate_prototype


def parse_options():
    parser = argparse.ArgumentParser(description="E-AVL")
    parser.add_argument('--dataset', type=str, default="AVE", help='dataset name', choices=['AVE', 'VGG', 'Kinetics'])
    parser.add_argument('--num_classes', default=16, type=int,
                        help='number of pretrain classes, AVE(16),VGG(60),Kinetics(19)')
    parser.add_argument('--gpu_id', type=str, default="cuda:0", help='the gpu id')
    parser.add_argument('--lr', type=float, default=3e-4, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batchsize')
    parser.add_argument('--num_epochs', type=int, default=30, help='total training epochs')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', default=4, type=int, help='num of workers of dataloader')

    # T-PR
    parser.add_argument('--modulation', default='T-PR', type=str)
    parser.add_argument('--alpha', default=1.0, type=float, help='alpha in T-PR')
    parser.add_argument('--embed_dim', default=768, type=int, help='embed_dim of encoder')
    # T-AVeL
    parser.add_argument('--T_AVeL_dim', type=int, default=16, help='dimension of the T-AVeL')
    parser.add_argument('--T_AVeL_loc', type=str, default='2', help='location of the T-AVeL')
    parser.add_argument('--latent_attention_loc', type=str, default='cma_1cma_2',
                        help='location of the latent attention')
    # Block
    parser.add_argument('--begin_layer', type=int, default=4, help='begin layer of the fusion block')

    # Path
    parser.add_argument('--pretrain_root', type=str, default='', help='dir of pretrain csv files')
    parser.add_argument('--audio_dir', type=str, default='', help='dir of audio files')
    parser.add_argument('--visual_dir', type=str, default='', help='dir of rgb frames')

    # logs
    parser.add_argument('--model_name', type=str, default='AVE_E-AVL', help='model name for saving ckpt')
    parser.add_argument('--save_path', type=str, default="", help='fold path to save the model')

    opts = parser.parse_args()
    setup_seed(opts.seed)
    opts.device = torch.device(opts.gpu_id)
    return opts


############################################################################################################################################################################################################
############################################################################################################################################################################################################

def pretrain(args):
    print(args.model_name)
    os.environ["TOKENIZERS_PARALLELISM"] = 'false'
    writer = SummaryWriter(comment="_" + args.model_name + "_" + args.gpu_id[-1])
    best_train_acc = []
    best_acc = 0

    if args.save_path == "":
        save_path = "./Pretrain_ckpt"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    else:
        save_path = args.save_path

    model = E_AVLmodel(num_classes=args.num_classes, T_AVeL_dim=args.T_AVeL_dim,
                       latent_attention_loc=args.latent_attention_loc, T_AVeL_loc=args.T_AVeL_loc,
                       begin_layer=args.begin_layer)
    collate = collate_fn_caption

    model.to(args.device)

    print("\t Model Loaded")

    pretrain_dataset = Dataset_AVC(os.path.join(args.pretrain_root, 'pretrain.csv'), args.audio_dir,
                                   args.visual_dir)
    pre_test_dataset = Dataset_AVC(os.path.join(args.pretrain_root, 'pretrain_test.csv'), args.audio_dir,
                                   args.visual_dir)

    train_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size, collate_fn=collate, shuffle=True,
                              num_workers=args.num_workers)

    test_loader = DataLoader(pre_test_dataset, batch_size=args.batch_size, collate_fn=collate, shuffle=False,
                             num_workers=args.num_workers)
    print("\t Dataset Loaded")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Loss Function
    loss_fn = nn.CrossEntropyLoss()
    model_save_path = os.path.join(save_path, args.model_name + ".pt")
    if args.modulation == 'T-PR':
        audio_proto, visual_proto = calculate_prototype(args, model, train_loader, class_map=None)

    else:
        audio_proto, visual_proto = None, None

    print("\t Start Training")
    for epoch in range(args.num_epochs):

        loss, acc = train_one_epoch(args, train_loader, model, optimizer, loss_fn, audio_proto,
                                    visual_proto, class_map=None)
        if args.modulation == 'T-PR':
            audio_proto, visual_proto = calculate_prototype(args, model, train_loader, class_map=None)

        # Testing
        val_res = val_one_epoch(args, test_loader, model, loss_fn, class_map=None)

        test_loss = val_res[0]
        test_acc = val_res[1]

        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)

        print('\nEpoch....', epoch + 1)
        print("Training loss & accuracy......", round(loss, 4), round(acc, 2))
        print("Test loss & accuracy......", round(test_loss, 4), round(test_acc, 2))
        best_train_acc.append(acc)

        if test_acc > best_acc:
            torch.save(model.state_dict(), model_save_path)
            best_acc = test_acc

    # best_val_acc.append(test_acc)
    print(args.model_name)
    print("BEST TEST ACCURACY........", round(best_acc, 2))
    print("BEST TRAIN ACCURACY.......", round(np.max(np.asarray(best_train_acc)), 2))


if __name__ == "__main__":
    opts = parse_options()
    pretrain(args=opts)
