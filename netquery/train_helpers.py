import random
import numpy as np
from utils import evaluate_path_auc, evaluate_intersect_auc

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

def run_eval(model, paths, inters, iteration, logger,
        max_path_len=3, max_inter_size=3):
    vals = []
    for i in range(1, max_path_len):
        auc, _ = evaluate_path_auc(paths[i][0], paths[i][1], model)
        logger.info("Path-{:d} val AUC: {:f}; iteration: {:d}".format(i, auc, iteration))
        vals.append(auc)
    for i in range(2, max_inter_size):
        auc, _ = evaluate_intersect_auc(inters[i][0], inters[i][1], model)
        logger.info("Inter-{:d} val AUC: {:f}; iteration: {:d}".format(i, auc, iteration))
        vals.append(auc)
    return vals

def run_train(model, optimizer, train_paths, val_paths, train_inters, val_inters, logger,
        max_burn_in = {1:50000, 2:50000}, batch_size=512, log_every=100, val_every=1000, tol=1e-6,
        max_iter=10e7, max_path_len=3, max_inter_size=3, inter_weight=0.1):
    path_conv = {i:False for i in range(1,2)}
    ema_loss = None
    vals = []
    losses = []

    for i in xrange(max_iter):
        
        optimizer.zero_grad()
        loss = run_path_batch(train_paths[1][0], model, i, batch_size, optimizer)
        if not path_conv[1] and (check_conv(vals) or len(losses) >= max_burn_in[1]):
            logger.info("Edge converged at iteration {:d}".format(i))
            path_conv[1] = True
            losses = []
            ema_loss = None
            v = run_eval(model, train_paths, val_paths, val_inters, i, logger)
            vals = [np.mean(v)]
        
        if path_conv[1]:
            for i in range(2, max_path_len + 1):
                loss += run_path_batch(train_paths[2][0], model, i, batch_size, optimizer)
                loss += run_path_batch(train_paths[2][0], model, i, batch_size, optimizer)
            for i in range(2, max_inter_size + 1):
                loss += inter_weight*run_inter_batch(train_inters[i][0], train_inters[i][1], model, i, batch_size, optimizer)
            if check_conv(vals):
                logger.info("Fully converged at iteration {:d}".format(i))
                break

        losses, ema_loss = update_loss(loss.data[0], losses, ema_loss)
        loss.backward()
        optimizer.step()
            
        if i % log_every == 0:
            logger.info("Iter: {:d}; ema_loss: {:f}".format(i, ema_loss))
            
        if i > val_every and i % val_every == 0:
            v = run_eval(model, train_paths, val_paths, val_inters, i, logger)
            if path_conv[1]:
                vals.append(np.mean(v))
            else:
                vals.append(v[0])

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




