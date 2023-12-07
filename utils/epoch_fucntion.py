import numpy as np
import torch
from torch import nn

from utils.functions import get_mapped_label
from utils.prototype_util import EU_dist


def val_one_epoch(args, val_data_loader, model, loss_fn, class_map=None):
    ### Local Parameters
    softmax = nn.Softmax(dim=1)
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0
    device = args.device
    model.eval()

    with torch.no_grad():
        for items in val_data_loader:
            # Loading data and labels to device
            spec = items[0].to(device)
            imgs = items[1].to(device)

            if class_map is None:
                labels = items[-1].to(device)
            else:
                labels = get_mapped_label(items[-1], class_map)
                labels = labels.to(device)

            # Forward
            a, v, out, z = model(spec, imgs, items[2], device)
            prediction = softmax(out)
            # Calculating Loss
            _loss = loss_fn(out, labels)
            epoch_loss.append(_loss.item())

            sum_correct_pred += (torch.argmax(prediction, dim=1) == labels).sum().item()
            total_samples += len(labels)

    test_acc = round(sum_correct_pred / total_samples, 4) * 100
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    return epoch_loss, test_acc


def train_one_epoch(args, train_data_loader, model, optimizer, loss_fn, audio_proto, visual_proto, class_map=None):
    sum_correct_pred = 0
    total_samples = 0
    softmax = nn.Softmax(dim=1)
    _loss = 0
    device = args.device

    model.train()

    ###Iterating over data loader spec, imgs, labels
    for items in train_data_loader:
        # Reseting Gradients
        optimizer.zero_grad()
        # Loading data and labels to device
        spec = items[0].to(device)
        imgs = items[1].to(device)
        if class_map is None:
            labels = items[-1].to(device)
        else:
            labels = get_mapped_label(items[-1], class_map)
            labels = labels.to(device)

        # Forward

        a, v, out, z = model(spec, imgs, items[2], device)

        if args.modulation == 'T-PR':
            audio_sim = -EU_dist(a, audio_proto)  # B x n_class
            visual_sim = -EU_dist(v, visual_proto)

            score_a_p = sum([softmax(audio_sim)[i][labels[i]] for i in range(audio_sim.size(0))])
            score_v_p = sum([softmax(visual_sim)[i][labels[i]] for i in range(visual_sim.size(0))])
            ratio_a_p = score_a_p / score_v_p

            loss_proto_a = loss_fn(audio_sim, labels)
            loss_proto_v = loss_fn(visual_sim, labels)

            if ratio_a_p > 1:
                beta = 0  # audio coef
                lam = 1 * args.alpha  # visual coef
            elif ratio_a_p < 1:
                beta = 1 * args.alpha
                lam = 0
            else:
                beta = 0
                lam = 0
            loss = loss_fn(out, labels) + beta * loss_proto_a + lam * loss_proto_v

        else:
            loss = loss_fn(out, labels)

        loss.backward()
        optimizer.step()

        _loss += loss.item()

        sum_correct_pred += (torch.argmax(out, dim=1) == labels).sum().item()
        total_samples += len(labels)

    acc = round(sum_correct_pred / total_samples, 4) * 100
    return _loss / len(train_data_loader), acc
