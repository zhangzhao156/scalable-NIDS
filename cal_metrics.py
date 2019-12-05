import numpy as np
import pandas as pd

def get_curve(known,novel):
#     known = np.array([0.5,0.95,0.955])
#     novel = np.array([0.8,0.7,0.91])
    known.sort()
    novel.sort()
    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])
    num_k = known.shape[0]
    num_n = novel.shape[0]
    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]
    tpr95_pos = np.abs(tp / num_k - .95).argmin()
    print('tpr95_pos',tpr95_pos)
    tnr_at_tpr95 = 1. - fp[tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95


def metric(known, novel, verbose=False):
    tp, fp, tnr_at_tpr95 = get_curve(known, novel)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')

        # TNR
    mtype = 'TNR'
    results[mtype] = tnr_at_tpr95
    if verbose:
        print(' {val:6.3f}'.format(val=100. * results[mtype]), end='')

        # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/ tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/ fp[0], [0.]])
    results[mtype] = -np.trapz(1. - fpr, tpr)
    if verbose:
        print(' {val:6.3f}'.format(val=100. * results[mtype]), end='')

        # DTACC
    mtype = 'DTACC'
    results[mtype] = .5 * (tp/ tp[0] + 1. - fp/ fp[0]).max()
    if verbose:
        print(' {val:6.3f}'.format(val=100. * results[mtype]), end='')

        # AUIN
    mtype = 'AUIN'
    denom = tp + fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp / denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
    if verbose:
        print(' {val:6.3f}'.format(val=100. * results[mtype]), end='')

        # AUOUT
    mtype = 'AUOUT'
    denom = tp[0] - tp + fp[0] - fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0] - fp) / denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])
    if verbose:
        print(' {val:6.3f}'.format(val=100. * results[mtype]), end='')
        print('')

    return results

def tpr95(known, unkown):
    #calculate the falsepositive error when tpr is 95%
    Y1 = unkown
    X1 = known
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1),np.min(Y1)])
    gap = (end- start)/10000 # precision:200000
    # print('start',start)
    # print('end',end)
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.96 and tpr >= 0.94:
            fpr += error2
            total += 1
    if total == 0:
        print('corner case')
        fprBase = 1
    else:
        fprBase = fpr/total

    return fprBase

def auroc(known, unkown):
    #calculate the AUROC
    f1 = open('./Update_Base_ROC_tpr.txt', 'w')
    f2 = open('./Update_Base_ROC_fpr.txt', 'w')
    Y1 = unkown
    X1 = known
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1),np.min(Y1)])
    gap = (end- start)/10000

    aurocBase = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        f1.write("{}\n".format(tpr))
        f2.write("{}\n".format(fpr))
        aurocBase += (-fpr+fprTemp)*tpr
        fprTemp = fpr

    return aurocBase

def auprIn(known,novelty):
    #calculate the AUPR
    precisionVec = []
    recallVec = []
    Y1 = novelty
    X1 = known
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1),np.min(Y1)])
    gap = (end- start)/10000

    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp-recall)*precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase

def auprOut(known, novelty):
    #calculate the AUPR
    Y1 = novelty
    X1 = known
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1),np.min(Y1)])
    gap = (end- start)/10000

    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprBase += (recallTemp-recall)*precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase

def detection(known,novelty):
    #calculate the minimum detection error
    Y1 = novelty
    X1 = known
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1),np.min(Y1)])
    gap = (end- start)/10000

    errorBase = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr+error2)/2.0)

    return errorBase

