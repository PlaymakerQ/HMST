import numpy as np
import torch

def remove_padding_value(predictions, labels, padding_value):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().detach().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach().numpy()

    valid_indices = (labels != padding_value)
    valid_indices = np.sum(valid_indices, axis=-1) - 1
    batch_indices = np.arange(len(valid_indices))
    predictions = predictions[batch_indices, valid_indices]
    labels = labels[batch_indices, valid_indices]

    return predictions, labels

def calculate_Acc_at_k(batch_preds, batch_labels, k):
    # batch_preds, batch_labels = remove_padding_value(batch_preds, batch_labels, padding_value)
    batch_Acc = 0.0
    batch_length = len(batch_labels)
    top_k_rec = np.argsort(batch_preds, axis=-1)
    top_k_rec = np.flip(top_k_rec, axis=1)
    preds_k = top_k_rec[:, :k]
    for top_k, label in zip(preds_k, batch_labels):
        idx = np.where(top_k == label)[0]
        if len(idx) != 0:
            batch_Acc += 1
        else:
            batch_Acc += 0
    batch_Acc = batch_Acc / batch_length
    return batch_Acc

def calculate_MAP_at_k(batch_preds, batch_labels, k):
    # batch_preds, batch_labels = remove_padding_value(batch_preds, batch_labels, padding_value)
    batch_MAP = 0.0
    batch_length = len(batch_labels)
    top_k_rec = np.argsort(batch_preds, axis=-1)
    top_k_rec = np.flip(top_k_rec, axis=1)
    preds_k = top_k_rec[:, :k]
    for top_k, label in zip(preds_k, batch_labels):
        idx = np.where(top_k == label)[0]
        if len(idx) != 0:
            batch_MAP += 1 / (idx[0]+1)
        else:
            batch_MAP += 0
    batch_MAP = batch_MAP / batch_length
    return batch_MAP

def calculate_NDCG_at_k(batch_preds, batch_labels, k):
    # batch_preds, batch_labels = remove_padding_value(batch_preds, batch_labels, padding_value)
    batch_NDCG = 0.0
    batch_length = len(batch_labels)
    top_k_rec = np.argsort(batch_preds, axis=-1)
    top_k_rec = np.flip(top_k_rec, axis=1)
    preds_k = top_k_rec[:, :k]
    for top_k, label in zip(preds_k, batch_labels):
        idx = np.where(top_k == label)[0]
        if len(idx) != 0:
            batch_NDCG += 1 / np.log2(idx[0]+2)
        else:
            batch_NDCG += 0
    batch_NDCG = batch_NDCG / batch_length
    return batch_NDCG

def calculate_batch_mrr(batch_preds, batch_labels):
    # batch_preds, batch_labels = remove_padding_value(batch_preds, batch_labels, padding_value)
    mrr = 0.0
    top_k_rec = np.argsort(batch_preds, axis=-1)
    top_k_rec = np.flip(top_k_rec, axis=1)
    for top_k, label in zip(top_k_rec, batch_labels):
        idx = np.where(top_k == label)[0][0]
        mrr += 1 / (idx + 1)
    return mrr / len(batch_labels)


def calculate_batch_metrics(total_metrics, batch_preds, batch_labels, padding_value,
                            k_values):

    required_metrics = list(total_metrics.keys())

    batch_preds, batch_labels = remove_padding_value(batch_preds, batch_labels, padding_value)

    if 'Acc' in required_metrics:

        for k in k_values:
            score = calculate_Acc_at_k(batch_preds, batch_labels, k)
            total_metrics['Acc'][k].append(score)

    if 'MAP' in required_metrics:

        for k in k_values:
            score = calculate_MAP_at_k(batch_preds, batch_labels, k)
            total_metrics['MAP'][k].append(score)

    if 'NDCG' in required_metrics:

        for k in k_values:
            score = calculate_NDCG_at_k(batch_preds, batch_labels, k)
            total_metrics['NDCG'][k].append(score)

    if 'MRR' in required_metrics:

        score = calculate_batch_mrr(batch_preds, batch_labels)
        total_metrics['MRR'].append(score)

    return total_metrics



