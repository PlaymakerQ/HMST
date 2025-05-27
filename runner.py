import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import calculate_Acc_at_k, calculate_batch_mrr, calculate_batch_metrics
from utils.tools import MyTool


def keep_valid_seqs(seqs, pad_value):
    valid_idx = (seqs != pad_value)
    valid_seqs = seqs[valid_idx]
    return valid_seqs

class Runner:

    def __init__(self, args, model, region_labels, cat_labels, up_edges, pp_edges):

        self.model_name = args.model_name
        self.device = args.device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.poi_loss = torch.nn.CrossEntropyLoss(ignore_index=args.num_poi)
        self.cat_loss = torch.nn.CrossEntropyLoss(ignore_index=args.num_cat)
        self.geo_loss = torch.nn.CrossEntropyLoss(ignore_index=args.num_region)

        self.region_labels = region_labels.to(self.device)
        self.cat_labels = cat_labels.to(self.device)
        self.up_edge_loader = DataLoader(up_edges, batch_size=1024*10, shuffle=True)
        self.pp_edge_loader = DataLoader(pp_edges, batch_size=1024*10, shuffle=True)

        self.PAD_VALUE = args.num_poi
        self.CAT_PAD = args.num_cat
        self.GEO_PAD = args.num_region

        # some parameters
        self.num_epoch = args.epoch
        self.k_list = args.k_list
        self.required_metrics = args.required_metrics

        # save results
        self.current_epoch = 1
        self.best_score = 0.0
        self.best_epoch = 0
        self.best_metric = None

    def check_best_result(self, metric_dict, target_metric='Acc'):
        check_scores = metric_dict[target_metric]
        score = sum(check_scores.values())
        if (self.best_metric is None) or (self.best_score < score):
            self.best_score = score
            self.best_metric = metric_dict
            self.best_epoch = self.current_epoch
            return True
        else:
            return False

    def process_batch(self, batch):

        input_seqs_poi, input_labels_poi, user_list = [], [], []

        for sample in batch:
            inputs, labels = sample
            seq_poi, user_id = inputs
            input_seqs_poi.append(torch.LongTensor(seq_poi))
            input_labels_poi.append(torch.LongTensor(labels))
            user_list.append(user_id)

        input_seqs_poi = pad_sequence(input_seqs_poi, batch_first=True, padding_value=self.PAD_VALUE)
        input_seqs_cat = self.cat_labels[input_seqs_poi]
        input_seqs_geo = self.region_labels[input_seqs_poi]
        input_labels_poi = pad_sequence(input_labels_poi, batch_first=True, padding_value=self.PAD_VALUE)
        input_labels_cat = self.cat_labels[input_labels_poi]
        input_labels_geo = self.region_labels[input_labels_poi]
        user_list = torch.LongTensor(user_list)

        all_input = [input_seqs_poi.to(self.device), input_seqs_cat.to(self.device),
                     input_seqs_geo.to(self.device), user_list.to(self.device)]
        all_label = [input_labels_poi, input_labels_cat, input_labels_geo]

        return all_input, all_label

    def train_network_embedding(self):
        self.model.train()
        with tqdm(total=len(self.up_edge_loader), desc='learn user-POI collaborative filtering...') as bar:
            for batch_edges in self.up_edge_loader:
                batch_edges = batch_edges.to(self.device)
                loss = 0.1 * self.model.calculate_edge_loss(batch_edges, edge_type='u-p')
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                bar.update(1)

        with tqdm(total=len(self.pp_edge_loader), desc='learn multi-semantic transition...') as bar:
            for pp_edges in self.pp_edge_loader:
                pp_edges = pp_edges.to(self.device)
                cc_edges = self.cat_labels[pp_edges]
                gg_edges = self.region_labels[pp_edges]
                pp_loss = self.model.calculate_edge_loss(pp_edges, edge_type='p-p')
                cc_loss = self.model.calculate_edge_loss(cc_edges, edge_type='c-c')
                gg_loss = self.model.calculate_edge_loss(gg_edges, edge_type='g-g')
                loss = 0.1 * (pp_loss + cc_loss + gg_loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                bar.update(1)




    def train(self, train_loader):
        self.model.train()
        epoch_loss = []
        self.train_network_embedding()
        with tqdm(total=len(train_loader),
                  desc=f'train process [{self.current_epoch:>03d}/{self.num_epoch:>03d}]') as progress_bar:
            for batch in train_loader:
                inputs, labels = self.process_batch(batch)
                pred_poi, pred_cat, pred_geo = self.model(inputs)
                label_pois, label_cats, label_geos = labels[0].to(self.device), labels[1].to(self.device), labels[2].to(self.device)
                # loss calculation
                poi_loss = self.poi_loss(pred_poi.transpose(2, 1), label_pois)
                cat_loss = self.cat_loss(pred_cat.transpose(2, 1), label_cats)
                geo_loss = self.geo_loss(pred_geo.transpose(2, 1), label_geos)
                loss = poi_loss + cat_loss + geo_loss

                # loss = batch_loss_poi
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item())
                progress_bar.update(1)
                progress_bar.set_postfix(current_mean_batch_loss=loss.item())

        epoch_loss = np.array(epoch_loss)  # average loss
        return epoch_loss

    def valid(self, valid_loader):
        return self.test(valid_loader)

    def test(self, test_loader):
        self.model.eval()
        total_metrics = MyTool.init_metric_dict(self.required_metrics, self.k_list)
        with tqdm(total=len(test_loader),
                  desc=f'test process [{self.current_epoch:>03d}/{self.num_epoch:>03d}]') as progress_bar:
            for batch in test_loader:
                inputs, labels = self.process_batch(batch)
                pred_poi, pred_cat, pred_geo = self.model(inputs)
                label_pois, label_cats, label_geos = labels
                total_metrics = calculate_batch_metrics(
                    total_metrics, pred_poi, label_pois, self.PAD_VALUE, self.k_list)
                progress_bar.update(1)

        total_metrics = MyTool.get_average_metric(total_metrics)

        return total_metrics

