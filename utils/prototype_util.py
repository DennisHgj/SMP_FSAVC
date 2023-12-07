# modified based on https://github.com/fanyunfeng-bit/Modal-Imbalance-PMR
import torch

from utils.functions import get_mapped_label


def calculate_prototype(args, model, dataloader, class_map=None, FSL=False):
    if FSL:
        n_classes = args.N_way
    else:
        n_classes = args.num_classes
    device = args.device

    audio_prototypes = torch.zeros(n_classes, args.embed_dim).to(device)
    visual_prototypes = torch.zeros(n_classes, args.embed_dim).to(device)
    count_class = [0 for _ in range(n_classes)]

    # calculate prototype
    model.eval()
    with torch.no_grad():
        sample_count = 0
        all_num = len(dataloader)
        for items in dataloader:
            spec = items[0].to(device)  # B x 257 x 1004
            imgs = items[1].to(device)  # B x 3(image count) x 3 x 224 x 224
            if class_map is None:
                labels = items[-1].to(device)
            else:
                labels = get_mapped_label(items[-1], class_map)
                labels = labels.to(device)

            preds = model(spec, imgs, items[2], device)

            audio_preds = preds[0]
            visual_preds = preds[1]
            aux_preds = preds[-1]

            for c, l in enumerate(labels):
                l = l.long()
                count_class[l] += 1
                audio_prototypes[l, :] += (audio_preds[c, :] + aux_preds[c, :])
                visual_prototypes[l, :] += (visual_preds[c, :] + aux_preds[c, :])

            sample_count += 1

            if not FSL and sample_count >= all_num // 10:
                break
    for c in range(audio_prototypes.shape[0]):
        if count_class[c] == 0:
            print('WARNING: class {} has no sample'.format(c))
            continue
        audio_prototypes[c, :] /= count_class[c]
        visual_prototypes[c, :] /= count_class[c]

    return audio_prototypes, visual_prototypes


def EU_dist(x1, x2):
    d_matrix = torch.zeros(x1.shape[0], x2.shape[0]).to(x1.device)
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            d = torch.sqrt(torch.dot((x1[i] - x2[j]), (x1[i] - x2[j])))
            d_matrix[i, j] = d
    return d_matrix
