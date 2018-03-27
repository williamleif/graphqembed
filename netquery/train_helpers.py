import random
import numpy as np

def check_conv(vals, window=2, tol=1e-6):
    if len(vals) < 2 * window:
        return False
    conv = np.mean(vals[-2*window:-window]) - np.mean(vals[-window:]) 
    return conv < tol

def update_loss(loss, losses, ema_loss, ema_alpha=0.01):
    losses.append(loss)
    if ema_loss is None:
        ema_loss = loss
    else:
        ema_loss = (1-ema_alpha)*ema_loss + ema_alpha*loss
    return losses, ema_loss


def run_path_batch(train_paths, enc_dec, iter_count, batch_size, optimizer):

    rels = random.choice(train_paths.keys())
    n = len(train_paths[rels])
    start = (iter_count * batch_size) % n
    end = min(((iter_count+1) * batch_size) % n, n)
    end = n if end <= start else end
    paths = train_paths[rels][start:end]
    if type(rels[0]) == str:
        rels = (rels,)
    loss = enc_dec.margin_loss([path[0] for path in paths], 
            [path[1] for path in paths], rels)

    return loss

def run_inter_batch(train_inters, neg_train_inters, 
        enc_dec, iter_count, batch_size, optimizer):

    rels = random.choice(train_inters.keys())
    n = len(train_inters[rels])
    start = (iter_count * batch_size) % n
    end = min(((iter_count+1) * batch_size) % n, n)
    end = n if end <= start else end
    inters = train_inters[rels][start:end]
    neg_inters = neg_train_inters[rels][start:end]
    if len(rels) != 3:
        loss = enc_dec.margin_loss([inter[1][0] for inter in inters], 
                [inter[1][1] for inter in inters], 
                rels,
                type_rel="intersect",
                target_nodes=[inter[0] for inter in inters],
                neg_nodes=[inter[0] for inter in neg_inters])
    else:
        loss = enc_dec.margin_loss([inter[1][0] for inter in inters], 
                [inter[1][1] for inter in inters], 
                rels,
                nodes3 = [inter[1][2] for inter in inters],
                type_rel="intersect",
                target_nodes=[inter[0] for inter in inters],
                neg_nodes=[inter[0] for inter in neg_inters])

    return loss
