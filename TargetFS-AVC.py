import argparse
import os.path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader.av_FSL import Dataset_AVC
from models.E_AVLmodel import E_AVLmodel
from utils.dataform import FSL_sample_formatting
from utils.epoch_fucntion import train_one_epoch, val_one_epoch
from utils.functions import setup_seed, collate_fn_caption, get_class_map
from utils.prototype_util import calculate_prototype

os.environ["TOKENIZERS_PARALLELISM"] = 'false'


def parse_options():
    parser = argparse.ArgumentParser(description="E-AVL")
    parser.add_argument('--dataset', type=str, default='AVE', help='dataset name', choices=['AVE', 'VGG', 'Kinetics'])
    parser.add_argument('--gpu_id', type=str, default="cuda:1", help='the gpu id')  # todo
    parser.add_argument('--lr', type=float, default=3e-4, help='initial learning rate')  # default=0.001
    parser.add_argument('--batch_size', type=int, default=8, help='batchsize')
    parser.add_argument('--num_epochs', type=int, default=30, help='training epochs for each sample test')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', default=4, type=int, help='num of workers of dataloader')
    # T-PR
    parser.add_argument('--modulation', default='T-PR', type=str)
    parser.add_argument('--alpha', default=1.0, type=float, help='alpha in T-PR')
    parser.add_argument('--embed_dim', default=768, type=int, help='embed_dim of encoder')
    # T-AVeL
    parser.add_argument('--T_AVeL_dim', type=int, default=16, help='dimension of the T-AVeL')
    parser.add_argument('--T_AVeL_loc', type=str, default='2', help='location of the T-AVeL')
    parser.add_argument('--latent_attention_loc', type=str, default='cma_1cma_2', help='location of the latent attention')
    # Block
    parser.add_argument('--begin_layer', type=int, default=4, help='begin layer of the fusion block')

    # Path
    parser.add_argument('--FS_AVC_root', type=str,
                        default='',help='dir of target fsavc csv files')
    parser.add_argument('--audio_dir', type=str, default='',
                        help='dir of audio files')
    parser.add_argument('--visual_dir', type=str, default='',
                        help='dir of rgb frames')
    parser.add_argument('--pretrained_model', type=str,
                        default='./AVE_E-AVL.pt',help="path to pretrained ckpt")  # todo

    parser.add_argument('--model_name', type=str, default='E-AVL_1shot',help='experiment name')  # todo
    # FS_AVC settings
    parser.add_argument('--N_way', type=int, default=5)
    parser.add_argument('--K_shot', type=int, default=1)
    parser.add_argument('--total_rounds', type=int, default=5, help='total rounds of sample N-way from dataset')
    parser.add_argument('--sample_times', type=int, default=5, help='total times of sample K-shot from selected classes')

    opts = parser.parse_args()
    setup_seed(opts.seed)
    opts.device = torch.device(opts.gpu_id)
    return opts


############################################################################################################################################################################################################
############################################################################################################################################################################################################
def train_one_sample(args, selected_class_train_df, selected_class_test_df, pretrained_paras, class_map):
    model = E_AVLmodel(num_classes=args.N_way, T_AVeL_dim=args.T_AVeL_dim,
                       latent_attention_loc=args.latent_attention_loc, T_AVeL_loc=args.T_AVeL_loc,
                       begin_layer=args.begin_layer)
    collate = collate_fn_caption

    if args.pretrained_model != '':
        info = model.load_state_dict(pretrained_paras, strict=False)
        print(info)

    model.to(args.device)
    FSL_train_dataset = Dataset_AVC(selected_class_train_df, args.audio_dir, args.visual_dir)
    FSL_test_dataset = Dataset_AVC(selected_class_test_df, args.audio_dir, args.visual_dir)

    train_loader = DataLoader(FSL_train_dataset, batch_size=args.batch_size, collate_fn=collate,
                              shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(FSL_test_dataset, batch_size=args.batch_size, collate_fn=collate,
                             shuffle=False, num_workers=args.num_workers)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Loss Function
    loss_fn = nn.CrossEntropyLoss()
    if args.modulation == 'T-PR':
        audio_proto, visual_proto = calculate_prototype(args, model, train_loader, class_map=class_map, FSL=True)
    else:
        audio_proto, visual_proto = None, None

    train_acc_list = []
    val_acc_list = []
    epoch_record = {}

    for epoch in range(args.num_epochs):
        ###Training

        loss, acc = train_one_epoch(args, train_loader, model, optimizer, loss_fn, audio_proto,
                                                visual_proto, class_map=class_map)
        if args.modulation == 'T-PR':
            audio_proto, visual_proto = calculate_prototype(args, model, train_loader, class_map=class_map, FSL=True)
        ###Test
        val_res = val_one_epoch(args, test_loader, model, loss_fn, class_map=class_map)
        train_acc_list.append(acc)
        val_acc_list.append(val_res[1])

    best_train_acc = np.max(np.asarray(train_acc_list))
    best_val_acc = np.max(np.asarray(val_acc_list))

    epoch_record['train_acc'] = np.asarray(train_acc_list)
    epoch_record['val_acc'] = np.asarray(val_acc_list)

    return best_train_acc, best_val_acc, epoch_record


def draw_plot(args, model_record):
    writer = SummaryWriter(comment=args.model_name)
    # print(model_record)

    np.save(os.path.join(writer.get_logdir(), 'model_record.npy'), model_record)

    all_results = {}
    all_results['train_acc'] = np.zeros((args.total_rounds * args.sample_times, args.num_epochs))
    all_results['val_acc'] = np.zeros((args.total_rounds * args.sample_times, args.num_epochs))

    row = 0
    for sample_class, sample_record in model_record.items():
        for sample_times, epoch_record in sample_record.items():
            for key, value in epoch_record.items():
                all_results[key][row] = value
            row += 1

    for i in range(args.num_epochs):
        writer.add_scalars("Accuracy", {
            "train_acc": np.mean(all_results['train_acc'][:, i]),
            "val_acc": np.mean(all_results['val_acc'][:, i]),
        }, i)


def FSL_training(args):
    print("start {} way {} shot training of {}  ".format(args.N_way, args.K_shot, args.model_name))

    fewshot_csv = os.path.join(args.FS_AVC_root, 'fewshot.csv')
    fewshot_test_csv = os.path.join(args.FS_AVC_root, 'fewshot_test.csv')

    round_train_acc_list = []
    round_valid_acc_list = []
    model_record = {}

    if args.pretrained_model != '':
        pretrained_paras = torch.load(args.pretrained_model, map_location=opts.device)

        pretrained_paras.pop('fusion_classification_head.fc_action.weight')
        pretrained_paras.pop('fusion_classification_head.fc_action.bias')

    else:
        print("no pretrained paras provided, plz check")
        pretrained_paras = None

    caption = 'caption'
    if args.dataset == 'VGG' or args.dataset == 'Kinetics':
        caption = 'VGG_caption'

    for sample_round in range(args.total_rounds):
        sample_train_acc_list = []
        sample_valid_acc_list = []
        sample_record = {}

        for sample_time in range(args.sample_times):
            print("\tStarted Training for sample times {}/{} in round {}/{}".format(sample_time, args.sample_times,
                                                                                    sample_round, args.total_rounds))
            if sample_time == 0:
                N_way = args.N_way
            else:
                N_way = random_select_class

            selected_class_train_df, selected_class_test_df, random_select_class = FSL_sample_formatting(
                fewshot_csv, fewshot_test_csv, N_way, args.K_shot, caption)

            if sample_time == 0:
                class_map_dict = get_class_map(random_select_class, args.N_way)

            sample_res = train_one_sample(args, selected_class_train_df, selected_class_test_df, pretrained_paras,
                                          class_map_dict)
            sample_train_acc_list.append(sample_res[0])
            sample_valid_acc_list.append(sample_res[1])

            sample_record[sample_time] = sample_res[2]

        print('In round {}, the selected target class is {}'.format(sample_round, random_select_class))
        print('average best training accuracy is {}, min is {}, max is {}'.format(
            round(np.mean(sample_train_acc_list), 2), round(np.min(sample_train_acc_list), 2),
            round(np.max(sample_train_acc_list), 2)))
        print('average best validation accuracy is {}, min is {}, max is {}'.format(
            round(np.mean(sample_valid_acc_list), 2), round(np.min(sample_valid_acc_list), 2),
            round(np.max(sample_valid_acc_list), 2)))
        round_train_acc_list.append(np.mean(sample_train_acc_list))
        round_valid_acc_list.append(np.mean(sample_valid_acc_list))

        model_record[str(random_select_class)] = sample_record

    print("************************************************************")
    print("Model Results:")
    print('For {} model, the average training acc is {}, test acc is {}'.format(
        args.model_name, round(np.mean(round_train_acc_list), 2), round(np.mean(round_valid_acc_list), 2)))
    print('the max test acc is {}, min is {}, std is {}'.format(round(np.max(round_valid_acc_list), 2),
                                                                round(np.min(round_valid_acc_list), 2),
                                                                round(np.std(round_valid_acc_list), 2))
          )
    draw_plot(args, model_record)


if __name__ == "__main__":
    opts = parse_options()
    FSL_training(args=opts)
