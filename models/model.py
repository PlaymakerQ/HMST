import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from geoopt.manifolds.lorentz import Lorentz
from manifolds.lorentz_functions import *
from models.layers import Rotation
from modules.LorentzianAttention import LorentzSelfAttention


class HMGRec(nn.Module):
    """ Hyperbolic Rotary Multi-semantic Transition Recommendation Model """

    def __init__(self, config):
        super(HMGRec, self).__init__()

        self.num_user = config.num_user
        self.num_poi = config.num_poi
        self.num_cat = config.num_cat
        self.num_region = config.num_region
        self.device = config.device


        model_params = config.model_params
        self.c = model_params['c']
        self.num_dim = model_params['num_dim']
        self.hyp_dim = self.num_dim + 1  # dimension for the Lorentz model
        self.num_negs = model_params['num_negs']
        self.dropout = model_params['dropout']

        self.Lorentz = Lorentz(self.c)
        self.user_embedding = nn.Embedding(self.num_user, self.hyp_dim)
        self.user_embedding.weight.data = self.Lorentz.random_normal((self.num_user, self.hyp_dim))
        self.poi_embeddings = nn.Embedding(self.num_poi + 1, self.hyp_dim, padding_idx=self.num_poi)
        self.poi_embeddings.weight.data = self.Lorentz.random_normal((self.num_poi + 1, self.hyp_dim))
        self.cat_embeddings = nn.Embedding(self.num_cat + 1, self.hyp_dim, padding_idx=self.num_cat)
        self.cat_embeddings.weight.data = self.Lorentz.random_normal((self.num_cat + 1, self.hyp_dim))
        self.geo_embeddings = nn.Embedding(self.num_region + 1, self.hyp_dim, padding_idx=self.num_region)
        self.geo_embeddings.weight.data = self.Lorentz.random_normal((self.num_region + 1, self.hyp_dim))

        self.user_bias = nn.Embedding(self.num_user, 1)
        self.poi_bias = nn.Embedding(self.num_poi, 1)
        self.cat_bias = nn.Embedding(self.num_cat, 1)
        self.geo_bias = nn.Embedding(self.num_region, 1)

        self.cf_rot = Rotation(self.num_dim)
        self.poi_rot = Rotation(self.num_dim)
        self.cat_rot = Rotation(self.num_dim)
        self.geo_rot = Rotation(self.num_dim)

        self.hyp_attention = LorentzSelfAttention(self.hyp_dim, dropout=self.dropout)

        self.poi_decoder = nn.Linear(self.hyp_dim, self.num_poi)
        self.cat_decoder = nn.Linear(self.hyp_dim, self.num_cat)
        self.geo_decoder = nn.Linear(self.hyp_dim, self.num_region)

    def calculate_edge_loss(self, edges, edge_type='u-p'):

        frs_type, tos_type = edge_type.split('-')

        if frs_type == 'u':
            source_embeds = self.user_embedding
            source_bias = self.user_bias
            rot_mat = self.cf_rot
        elif frs_type == 'p':
            source_embeds = self.poi_embeddings
            source_bias = self.poi_bias
            rot_mat = self.poi_rot
        elif frs_type == 'c':
            source_embeds = self.cat_embeddings
            source_bias = self.cat_bias
            rot_mat = self.cat_rot
        elif frs_type == 'g':
            source_embeds = self.geo_embeddings
            source_bias = self.geo_bias
            rot_mat = self.geo_rot
        else:
            raise NotImplementedError

        if tos_type == 'u':
            target_embeds = self.user_embedding
            target_bias = self.user_bias
            sample_number = self.num_user
        elif tos_type == 'p':
            target_embeds = self.poi_embeddings
            target_bias = self.poi_bias
            sample_number = self.num_poi
        elif tos_type == 'c':
            target_embeds = self.cat_embeddings
            target_bias = self.cat_bias
            sample_number = self.num_cat
        elif tos_type == 'g':
            target_embeds = self.geo_embeddings
            target_bias = self.geo_bias
            sample_number = self.num_region
        else:
            raise NotImplementedError

        # nodes
        frs = edges[:, 0:1]  # users
        tos = edges[:, 1:2]  # POIs
        negs = self.get_neg_samples(tos, sample_number)

        # embeddings
        frs_embeds = source_embeds(frs)
        frs_embeds = self.Lorentz.projx(rot_mat(frs_embeds))  # rotation operation
        tos_embeds = self.Lorentz.projx(target_embeds(tos))
        negs_embeds = self.Lorentz.projx(target_embeds(negs))

        positive_distance_score = self.similarity_score(frs_embeds, tos_embeds)
        bias_frs = source_bias(frs)
        bias_tos = target_bias(tos)
        positive_score = positive_distance_score + bias_frs + bias_tos

        negative_distance_score = self.similarity_score(frs_embeds, negs_embeds)
        bias_negs = target_bias(negs)
        negative_score = negative_distance_score + bias_frs + bias_negs

        positive_loss = F.logsigmoid(positive_score).sum()
        negative_loss = F.logsigmoid(-negative_score).sum()

        loss = - (positive_loss + negative_loss)

        return loss

    def forward(self, inputs):
        poi_seqs, cat_seqs, geo_seqs, user_list = inputs

        space_o = self.Lorentz.origin(1, self.hyp_dim)

        poi_embeds = self.Lorentz.projx(self.poi_embeddings(poi_seqs))
        cat_embeds = self.Lorentz.projx(self.cat_embeddings(cat_seqs))
        geo_embeds = self.Lorentz.projx(self.geo_embeddings(geo_seqs))  # (B, L, d)
        user_embeds = self.Lorentz.projx(self.user_embedding(user_list))  # (B, d)

        def map_to_tangent_space_at_origin(x):
            x = self.Lorentz.proju(space_o, x)
            return x

        poi_tan = map_to_tangent_space_at_origin(poi_embeds)
        cat_tan = map_to_tangent_space_at_origin(cat_embeds)
        geo_tan = map_to_tangent_space_at_origin(geo_embeds)
        user_tan = map_to_tangent_space_at_origin(user_embeds)

        checkin_embeds = 0.3 * poi_tan + 0.1 * cat_tan + 0.1 * geo_tan + 0.5 * user_tan[:, None, :]
        checkin_embeds = self.Lorentz.projx(checkin_embeds)

        attn_mask = (poi_seqs == self.num_poi)
        traj_embeds = self.hyp_attention(checkin_embeds, checkin_embeds, checkin_embeds, attn_mask)

        traj_embeds_poi_rot = map_to_tangent_space_at_origin(self.Lorentz.projx(self.poi_rot(traj_embeds)))
        traj_embeds_cat_rot = map_to_tangent_space_at_origin(self.Lorentz.projx(self.cat_rot(traj_embeds)))
        traj_embeds_geo_rot = map_to_tangent_space_at_origin(self.Lorentz.projx(self.geo_rot(traj_embeds)))

        all_poi_embeds = self.poi_embeddings.weight[:-1]
        all_cat_embeds = self.cat_embeddings.weight[:-1]
        all_geo_embeds = self.geo_embeddings.weight[:-1]

        output_poi = self.poi_decoder(traj_embeds_poi_rot) + self.calculate_distance_score(traj_embeds_poi_rot, all_poi_embeds)
        output_cat = self.cat_decoder(traj_embeds_cat_rot) + self.calculate_distance_score(traj_embeds_cat_rot, all_cat_embeds)
        output_geo = self.geo_decoder(traj_embeds_geo_rot) + self.calculate_distance_score(traj_embeds_geo_rot, all_geo_embeds)

        return output_poi, output_cat, output_geo

    def get_neg_samples(self, golds, num_nodes):
        neg_samples = torch.randint(
            num_nodes, (golds.shape[0], self.num_negs)
        ).to(golds.device)

        return neg_samples

    def similarity_score(self, x, y):
        c = self.c
        uv = self.Lorentz.inner(x=x, u=x, v=y)[:, None, :]
        result = - 2 * c - 2 * uv
        score = result.neg()
        return score

    def calculate_distance_score(self, x, y):
        c = self.c
        y = y.clone()
        y[:, 0] = -y[:, 0]
        uv = x @ y.T
        result = - 2 * c - 2 * uv
        score = torch.exp(result.neg())
        return score

