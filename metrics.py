import numpy as np

def get_tn_tp_fn_fp(y_true, y_pred):
    tn = np.sum(np.logical_and(np.logical_not(y_true), np.logical_not(y_pred))).astype(np.float64)
    tp = np.sum(np.logical_and(               y_true ,                y_pred )).astype(np.float64)
    fn = np.sum(np.logical_and(               y_true , np.logical_not(y_pred))).astype(np.float64)
    fp = np.sum(np.logical_and(np.logical_not(y_true),                y_pred )).astype(np.float64)
    return tn, tp, fn, fp

def matthews_corrcoef(y_true, y_pred):
    tn, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)
    mcc = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if np.isnan(mcc):
        return 0.
    else:
        return mcc

def f1_score(y_true, y_pred):
    tn, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)
    f1 = 2*tp/(2*tp+fp+fn)
    if np.isnan(f1):
        return 0.
    else:
        return f1

def iou_measure(y_true, y_pred):
    _, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)
    return tp/(tp+fp+fn)

def get_metrics(y_true, y_pred):
    tn, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)

    f1 = 2*tp/(2*tp+fp+fn+1e-5)
    if np.isnan(f1):
        f1 = 0.

    mcc = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)+1e-5)
    if np.isnan(mcc):
        mcc = 0.

    iou = tp/(fp+tp+fn+1e-5)
    if np.isnan(iou):
        iou = 0.
    tpr_recall = tp/(tp+fn+1e-5)
    if np.isnan(tpr_recall):
        tpr_recall = 0.
    tnr = tn/(tn+fp+1e-5)
    if np.isnan(tnr):
        tnr = 0.
    precision = tp/(tp+fp+1e-5)
    if np.isnan(precision):
        precision = 0.

    return tpr_recall,tnr,precision,f1,mcc,iou,tn,tp,fn,fp
