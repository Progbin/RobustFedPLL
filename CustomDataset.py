import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, candidate_labels_set, true_labels, transform=None):
        self.data = data.data
        self.candidate_labels_set = candidate_labels_set
        self.true_labels = true_labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        candidate_labels = self.candidate_labels_set[index]
        true_label = self.true_labels[index]

        if self.transform:
            x = self.transform(x)

        return x, candidate_labels, true_label, index

    def update_pseudo_labels(self, index,new_pseudo_labels):
        for i, idx in enumerate(index):
            self.candidate_labels_set[idx] = new_pseudo_labels[i]

    def update_true_labels(self, index,new_true_labels):
        for i, idx in enumerate(index):
            self.true_labels[index] = new_true_labels[i]