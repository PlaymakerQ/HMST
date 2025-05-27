import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

from utils.constants import col_name
from utils.tools import MyTool


def show_clusters(gps_locations, k, labels):
    # Plot the clustered GPS locations
    plt.figure(figsize=(8, 6))
    for i in range(k):
        # Plot points belonging to cluster i
        cluster_points = gps_locations[labels == i]
        plt.scatter(cluster_points[:, 1], cluster_points[:, 0], label=f'Cluster {i + 1}')
    # Add labels and legend
    plt.title("KMeans Clustering of POI Locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()


def split_region(locations, num_region=50, method='KMeans'):
    # K-Means clustering
    kmeans = KMeans(n_clusters=num_region, random_state=0)
    kmeans.fit(locations)
    labels = kmeans.labels_  # Region assignment for each location
    return labels


class TrajectoryDataset:

    def __init__(self, inputs, labels):

        self.input_seqs = inputs
        self.label_seqs = labels

    def __len__(self):
        assert len(self.input_seqs) == len(self.label_seqs)
        return len(self.input_seqs)

    def __getitem__(self, index):
        return self.input_seqs[index], self.label_seqs[index]


def build_trajectory_data(df, min_len, max_len, data_name):

    inputs, labels = [], []

    tag = df['tag'].unique().tolist()[0]
    for traj_id in tqdm(set(df[col_name.trajectory_id].tolist()), desc=f"building {tag} trajectories..."):
        traj_df = df[df[col_name.trajectory_id] == traj_id]

        poi_list = traj_df[col_name.poi_id].to_list()  # POI Id

        if len(poi_list) < min_len or (data_name != 'NYC' and len(poi_list) > max_len):
            continue

        poi_input = poi_list[:-1]
        poi_label = poi_list[1:]
        user_id = traj_df.user_id.tolist()[0]

        input = [poi_input, user_id]
        label = poi_label

        inputs.append(input)
        labels.append(label)

    return inputs, labels


class POIDataset:

    def __init__(self, data_name, min_len=3, max_len=101):

        self.data_name = data_name
        self.min_len = min_len
        self.max_len = max_len
        self.valid_transition_time = 6 * 60 * 60

        root_path = MyTool.get_root_path()
        self.data_path = os.path.join(root_path, "dataset", data_name)
        self.df = pd.read_csv(os.path.join(self.data_path, f"{data_name}.csv"))
        self.df[col_name.local_time] = pd.to_datetime(self.df[col_name.local_time])
        self.df['timestamp'] = self.df[col_name.local_time].apply(lambda x: x.timestamp())
        self.split_tags = self.df['tag'].unique().tolist()
        self.tables = {tag: self.df[self.df['tag'] == tag] for tag in self.split_tags}

        traj_dict = {}
        for tag in self.split_tags:
            sub_df = self.tables[tag]
            inputs, labels = build_trajectory_data(sub_df, self.min_len, self.max_len, self.data_name)
            traj_dict[tag] = TrajectoryDataset(inputs, labels)
        self.traj_dict = traj_dict

    def set_nums(self, config):
        train_df = self.tables['train']
        config.num_poi = train_df[col_name.poi_id].nunique()
        config.num_user = train_df[col_name.user_id].nunique()
        return config

    def get_region_information(self, num_region, method='KMeans'):
        """

        :return: torch.LongTensor, ([num_region+1]), POI region labels

        """
        df = self.tables['train']
        df = df.drop_duplicates(subset=[col_name.poi_id], keep='first')
        df = df.sort_values(by=col_name.poi_id, ascending=True).reset_index(drop=True)
        locations = df[[col_name.latitude, col_name.longitude]].values
        region_labels = split_region(locations, num_region, method)
        region_labels = torch.LongTensor(region_labels)
        padding_value = torch.LongTensor([num_region])
        region_labels = torch.cat((region_labels, padding_value), dim=0)
        return region_labels

    def get_category_information(self):
        """

        :return: torch.LongTensor, ([num_cat+1]), POI category labels

        """
        df = self.tables['train']
        df = df.drop_duplicates(subset=[col_name.poi_id], keep='first')
        df = df.sort_values(by=col_name.poi_id, ascending=True).reset_index(drop=True)
        df[col_name.cat_id] = pd.factorize(df[col_name.cat_id])[0]
        num_cat = df[col_name.cat_id].nunique()
        categories = df[col_name.cat_id].values
        categories = torch.LongTensor(categories)
        padding_value = torch.LongTensor([num_cat])
        categories = torch.cat((categories, padding_value), dim=0)
        return categories

    def get_user_poi_edges(self):
        df = self.tables['train']
        edges = df[[col_name.user_id, col_name.poi_id]].values
        edges = torch.LongTensor(edges)
        return edges

    def get_poi_poi_edges(self):
        df = self.tables['train']
        df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
        user_group = df.groupby([col_name.user_id])
        edges = []
        for user_id, user_df in user_group:
            time_list = user_df['timestamp'].tolist()
            poi_list = user_df[col_name.poi_id].tolist()
            for i in range(len(time_list)-1):
                time_diff = time_list[i+1] - time_list[i]
                if time_diff <= self.valid_transition_time:
                    valid_edge = (poi_list[i], poi_list[i+1])
                    edges.append(valid_edge)
        edges = torch.LongTensor(edges)
        return edges




if __name__ == '__main__':
    data_name = "NYC"
    dataset = POIDataset(data_name)
