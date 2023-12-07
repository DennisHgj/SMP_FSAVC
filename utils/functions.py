import random
import numpy as np
import torch
from torch.backends import cudnn

import torch.nn as nn


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def collate_fn_caption(batch):
    spectrograms, rgb_frames, captions, labels = [], [], [], []
    for spec, frame, caption, label in batch:
        spectrograms += [spec]
        rgb_frames += [frame]
        labels += [torch.tensor(label)]
        captions += [caption]
    # Group the list of tensors into a batched tensor
    spectrograms = af_pad_sequence(spectrograms)
    rgb_frames = torch.stack(rgb_frames)
    labels = torch.stack(labels)

    return spectrograms, rgb_frames, captions, labels


def af_pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def get_mapped_label(labels, class_map):
    for i in range(len(labels)):
        temp_label = labels[i]
        target_label = class_map.get(temp_label.item())
        labels[i] = target_label
    return labels


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def get_class_map(random_select_class, N_way):
    class_map_dict = {}
    for i in range(N_way):
        class_map_dict[random_select_class[i]] = i
    return class_map_dict