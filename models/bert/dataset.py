import json
import numpy as np

import torch
from torch.utils.data import Dataset


class QueriesDataset(Dataset):
    def __init__(self, max_results=10000, split=None, data_path=None, percent=100):
        if split == "train":
            self.percent = f"{percent}_" if percent != 100 else ""
        elif split == "dev":
            self.percent = ""

        self.data_path = data_path if data_path else "./data"
        print(f"{self.data_path}/input_ids_{max_results}_results_{self.percent}{split}.pt")
        self.input_ids = torch.load(f"{self.data_path}/input_ids_{max_results}_results_{self.percent}{split}.pt")

        raw_attention_masks = torch.load(f"{self.data_path}/attn_{max_results}_results_{self.percent}{split}.pt")
        raw_attention_masks = raw_attention_masks == 0
        self.attention_mask = torch.zeros_like(raw_attention_masks).to(torch.float)
        self.attention_mask[raw_attention_masks] = float("-inf")

        self.labels = torch.load(f"{self.data_path}/labels_{max_results}_results_{self.percent}{split}.pt")
        self.len = self.labels.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return dict(
            input_ids=self.input_ids[idx],
            attention_mask=self.attention_mask[idx],
            labels=self.labels[idx]
        )


class SimilarityDataset(Dataset):
    def __init__(self, split=None, use_sim_s=True, use_sim_w=True, use_sim_r=True, args=None):
        if split == "train":
            self.percent = f"{args.queries_percent_for_train}_" if args.queries_percent_for_train != 100 else ""
        elif split == "dev":
            self.percent = ""

        self.data_path = args.data if args else "./data"
        self.input_ids = torch.load(f"{self.data_path}/sim_input_ids_{self.percent}{split}.pt")

        raw_attention_masks = torch.load(f"{self.data_path}/sim_attn_{self.percent}{split}.pt")
        raw_attention_masks = raw_attention_masks == 0
        self.attention_mask = torch.zeros_like(raw_attention_masks).to(torch.float)
        self.attention_mask[raw_attention_masks] = float("-inf")

        self.use_sim_s = use_sim_s
        if self.use_sim_s:
            self.sim_s = torch.load(f"{self.data_path}/sim_s_{self.percent}{split}.pt")
            self.len = self.sim_s.shape[0]

        self.use_sim_r = use_sim_r
        if self.use_sim_r:
            self.sim_r = torch.load(f"{self.data_path}/sim_r_{self.percent}{split}.pt") 
            self.len = self.sim_r.shape[0]

        self.use_sim_w = use_sim_w
        if self.use_sim_w:
            self.sim_w = torch.load(f"{self.data_path}/sim_w_{self.percent}{split}.pt")
            self.len = self.sim_w.shape[0]
        

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        labels = dict()
        if self.use_sim_s:
            labels["sim_s"] = self.sim_s[idx]
        if self.use_sim_r:
            labels["sim_r"] = self.sim_r[idx]
        if self.use_sim_w:
            labels["sim_w"] = self.sim_w[idx]

        return dict(
            input_ids=self.input_ids[idx],
            attention_mask=self.attention_mask[idx],
            labels=labels
        )
