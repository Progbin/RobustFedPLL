from collections import Counter

import torch
from numpy.random import randn
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA, PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
import numpy as np
from torch.nn.functional import one_hot, cosine_similarity
import torch.nn.functional as F

from defense_myCL import Encoder, InfoNCELoss, update_representation, update_confidence_list_with_representation, \
    load_model_from_train_stage, cluster_clients, update_confidence_list, \
    construct_positive_negative_samples, sample_contrastive_batch, gmm
from defense import calculate_gradient
from preprocess import data_augmentation
from resnet18 import ResNet18
from CNN import CNN_2Conv


class FedPLL(nn.Module):
    def __init__(self, dataset_name = 'cifar10', client_nums = 10, class_nums = 10,client_data=None):
        super(FedPLL, self).__init__()
        self.dataset_name = dataset_name
        self.clients = []
        self.server = CNN_2Conv(class_nums).cuda() if dataset_name == 'mnist' else ResNet18(class_nums).cuda()
        for i in range(client_nums):
            model = CNN_2Conv(class_nums).cuda() if dataset_name == 'mnist' else ResNet18(class_nums).cuda()
            for param_q, param_k in zip(self.server.parameters(), model.parameters()):
                param_k.data.copy_(param_q.data)
                param_q.requires_grad_(False)
            client = {"model": model, "data":client_data[i]}
            self.clients.append(client)


    @torch.no_grad()
    def update_local_weight(self, current_clients):
        for idx, client in enumerate(self.clients):
            if idx in current_clients:
                client["model"].load_state_dict(self.server.state_dict().copy())

    @torch.no_grad()
    def update_client_confidence_no_CL(self, current_clients, confidence, round_num):
        alpha = 0.9
        server_state_dict = self.server.state_dict()
        client_state_dict = {}
        for idx, client in enumerate(self.clients):
            if idx in current_clients:
                client_state_dict[f"client_{idx}"] = (self.clients[idx]["model"].state_dict()['classifier.weight'])

        diffs, client_ids = calculate_gradient(server_state_dict, client_state_dict)
        if diffs is None or client_ids is None:
            return confidence


        cluster_labels = gmm(diffs)
        print(cluster_labels)
        label_counts = {label: (cluster_labels == label).sum() for label in set(cluster_labels)}

        poisoned_label = min(label_counts, key=label_counts.get)

        adjusted_cluster_labels = [
            0 if label == poisoned_label else 1 for label in cluster_labels
        ]
        if round_num <= 50:
            confidence = update_confidence_list(confidence, client_ids, adjusted_cluster_labels, alpha=alpha)
            print(f"Updated confidence list (Round {round_num}): {list(confidence.items())}")
        return confidence

    # @torch.no_grad()
    def update_client_confidence(self, current_clients, confidence, round_num):
        alpha = 0.9
        representation_dim = 128
        server_state_dict = self.server.state_dict()
        client_state_dict = {}
        for idx, client in enumerate(self.clients):
            if idx in current_clients:
                client_state_dict[f"client_{idx}"] = (self.clients[idx]["model"].state_dict()['classifier.weight'])
        encoder = Encoder(input_dim=5120, output_dim=representation_dim)
        info_nce_loss = InfoNCELoss(temperature=0.1)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

        diffs, client_ids = calculate_gradient(server_state_dict, client_state_dict)
        if diffs is None or client_ids is None:
            return confidence


        if round_num <= 40:
            cluster_labels = gmm(diffs)
            print(cluster_labels)
            label_counts = {label: (cluster_labels == label).sum() for label in set(cluster_labels)}

            poisoned_label = min(label_counts, key=label_counts.get)

            adjusted_cluster_labels = [
                0 if label == poisoned_label else 1 for label in cluster_labels
            ]
            confidence = update_confidence_list(confidence, client_ids, adjusted_cluster_labels, alpha=alpha)
            print(f"Updated confidence list (Round {round_num}): {list(confidence.items())}")
            return confidence

        pos_samples, neg_samples = construct_positive_negative_samples(confidence, diffs, threshold=0.6)

        round_loss = 0
        encoder.train()
        for epoch in range(10):
            max_batch_size = len(pos_samples) // 2
            max_neg = len(neg_samples)
            batch_size = min(32, max_batch_size)
            num_neg = min(10, max_neg)
            batch = sample_contrastive_batch(pos_samples, neg_samples, batch_size=batch_size, num_neg=num_neg)
            if batch is None:
                break
            anchor, positive, neg_batch = batch

            z_i = encoder(anchor)
            z_j = encoder(positive)
            z_neg = encoder(neg_batch.view(-1, neg_batch.size(-1))).view(neg_batch.size(0), neg_batch.size(1), -1)

            loss = info_nce_loss(z_i, z_j, z_neg)
            round_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        representations = update_representation(diffs, encoder)
        cluster_labels = cluster_clients(representations.numpy())
        print(cluster_labels)
        label_counts = {label: (cluster_labels == label).sum() for label in set(cluster_labels)}

        poisoned_label = min(label_counts, key=label_counts.get)

        adjusted_cluster_labels = [
            0 if label == poisoned_label else 1 for label in cluster_labels
        ]
        confidence = update_confidence_list_with_representation(confidence, adjusted_cluster_labels, alpha=alpha)
        print(f"Updated confidence list (Round {round_num}): {list(confidence.items())}")

        return confidence

    @torch.no_grad()
    def defence(self, method, f=3, attack_clients=[]):
        alpha = 0.3

        uploaded_params = {}
        for i, client in enumerate(self.clients):
            weight = client["model"].state_dict()['classifier.weight']
            if i in attack_clients:
                global_weight = self.server.state_dict()['classifier.weight']
                masked_weight = alpha * global_weight + (1 - alpha) * weight
                uploaded_params[f"client_{i}"] = masked_weight
            else:
                uploaded_params[f"client_{i}"] = weight

        client_updates, client_ids = calculate_gradient(self.server.state_dict(), uploaded_params)

        client_updates = np.array(client_updates)
        n, d = client_updates.shape

        if method in ["PCA", "KPCA"]:
            scaler = StandardScaler()
            std_gradient = scaler.fit_transform(client_updates)

            reducer = PCA(n_components=2) if method == "PCA" else KernelPCA(n_components=2, kernel="rbf")
            reduced_gradient = reducer.fit_transform(std_gradient)

            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(reduced_gradient)
            labels = kmeans.labels_
            label_counts = Counter(labels)

            malicious_label = min(label_counts, key=label_counts.get)
            normal_label = 1 - malicious_label

            normal_clients = [i for i, label in enumerate(labels) if label == normal_label]
            malicious_clients = [i for i, label in enumerate(labels) if label == malicious_label]

        elif method == "Kmeans":
            scaler = StandardScaler()
            std_gradient = scaler.fit_transform(client_updates)
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(std_gradient)
            labels = kmeans.labels_
            label_counts = Counter(labels)
            malicious_label = min(label_counts, key=label_counts.get)
            normal_label = 1 - malicious_label
            normal_clients = [i for i, label in enumerate(labels) if label == normal_label]
            malicious_clients = [i for i, label in enumerate(labels) if label == malicious_label]

        elif method == "Krum":
            scores = []
            for i in range(n):
                distances = [np.linalg.norm(client_updates[i] - client_updates[j]) ** 2 for j in range(n) if j != i]
                distances.sort()
                score = sum(distances[:n - f - 2])
                scores.append(score)
            selected_idx = int(np.argmin(scores))
            normal_clients = [selected_idx]
            malicious_clients = [i for i in range(n) if i != selected_idx]

        print(f"[{method}] Normal clients:", normal_clients)
        print(f"[{method}] Malicious clients:", malicious_clients)
        return normal_clients, malicious_clients

    @torch.no_grad()
    def trimmed_mean_aggregation(self, trim_ratio=0.1):
        num_clients = len(self.clients)
        state_dicts = [client["model"].state_dict() for client in self.clients]
        aggregated_state_dict = {}
        for key in state_dicts[0]:
            params = torch.stack([sd[key] for sd in state_dicts], dim=0)
            if torch.is_floating_point(params):
                sorted_params, _ = torch.sort(params, dim=0)
                n_trim = int(num_clients * trim_ratio)
                if num_clients - 2 * n_trim <= 0:
                    raise ValueError("Too few clients or trim_ratio too high!")
                trimmed_params = sorted_params[n_trim:num_clients - n_trim, ...]
                mean_param = trimmed_params.mean(dim=0)
                aggregated_state_dict[key] = mean_param
            else:
                aggregated_state_dict[key] = params[0]
        self.server.load_state_dict(aggregated_state_dict)
        client_weights = [1 / num_clients for _ in range(len(self.clients))]
        self.aggregate_server_grads(current_clients=self.clients, client_weights=client_weights)
        return

    @torch.no_grad()
    def median_aggregation(self):
        num_clients = len(self.clients)
        state_dicts = [client["model"].state_dict() for client in self.clients]
        aggregated_state_dict = {}

        for key in state_dicts[0]:
            params = torch.stack([sd[key] for sd in state_dicts], dim=0)
            median_param = torch.median(params, dim=0).values
            aggregated_state_dict[key] = median_param

        self.server.load_state_dict(aggregated_state_dict)
        client_weights = [1 / num_clients for _ in range(len(self.clients))]

        self.aggregate_server_grads(current_clients=self.clients, client_weights=client_weights)

        return

    @torch.no_grad()
    def aggregate_server_grads(self, current_clients=None, client_weights=None):
        if current_clients is None:
            current_clients = list(range(len(self.clients)))
        if client_weights is None:
            client_weights = [1.0 / len(current_clients) for _ in range(len(self.clients))]

        for param in self.server.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)
            else:
                param.grad.zero_()

        for idx, client in enumerate(self.clients):
            if idx in current_clients:
                for server_param, client_param in zip(self.server.parameters(), client["model"].parameters()):
                    if client_param.grad is not None:
                        weight = client_weights[idx]
                        server_param.grad += weight * client_param.grad.clone().detach()

    @torch.no_grad()
    def update_server_weight(self, current_clients, confidence, num_round):
        server_state_dict = self.server.state_dict()
        for key in server_state_dict.keys():
            server_state_dict[key] = torch.zeros_like(server_state_dict[key])

        client_weights = [1 / len(current_clients) for _ in range(len(self.clients))]

        for idx, client in enumerate(self.clients):
            if idx in current_clients:
                client_state_dict = client["model"].state_dict()

                weight = client_weights[idx]

                for key in server_state_dict.keys():
                    if server_state_dict[key].dtype in [torch.float32, torch.float64]:
                        server_state_dict[key] += weight * client_state_dict[key].clone().float()
                    elif server_state_dict[key].dtype == torch.int64:
                        server_state_dict[key] = client_state_dict[key]

        for param in self.server.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)

        for idx, client in enumerate(self.clients):
            if idx in current_clients:
                for server_param, client_param in zip(self.server.parameters(), client["model"].parameters()):
                    if client_param.grad is not None:
                        weight = client_weights[idx]
                        server_param.grad += weight * client_param.grad.clone().detach()


        self.server.load_state_dict(server_state_dict)


    @torch.no_grad()
    def update_server_weight_scale_up(self, current_clients, attack_client, confidence, num_round):
        server_state_dict = self.server.state_dict()
        new_server_state = {}

        for key in server_state_dict.keys():
            new_server_state[key] = server_state_dict[key].clone()

        client_weights = [1 / len(current_clients) for _ in range(len(self.clients))]

        scale_factor = 2

        for idx, client in enumerate(self.clients):
            if idx in current_clients:
                client_state_dict = client["model"].state_dict()
                weight = client_weights[idx]

                for key in server_state_dict.keys():
                    if server_state_dict[key].dtype in [torch.float32, torch.float64]:
                        if idx in attack_client:
                            delta = client_state_dict[key].float() - server_state_dict[key].float()
                            new_server_state[key] += scale_factor * weight * delta
                        else:
                            delta = client_state_dict[key].float() - server_state_dict[key].float()
                            new_server_state[key] += weight * delta
                    elif server_state_dict[key].dtype == torch.int64:
                        new_server_state[key] = client_state_dict[key]

        self.server.load_state_dict(new_server_state)

        for param in self.server.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)

        for idx, client in enumerate(self.clients):
            if idx in current_clients:
                for server_param, client_param in zip(self.server.parameters(), client["model"].parameters()):
                    if client_param.grad is not None:
                        weight = client_weights[idx]
                        server_param.grad += weight * client_param.grad.clone().detach()

    @torch.no_grad()
    def extract_weights(self, clients):
        weights = []
        for idx, client in enumerate(self.clients):
            if idx in clients:
                state_dict = client["model"].state_dict()
                flat_weights = torch.cat([param.flatten() for param in state_dict.values()])
                weights.append(flat_weights.cpu().numpy())
        return np.array(weights)

    @torch.no_grad()
    def cluster_clients(self, clients, n_clusters=2, n_components=2, kernel="rbf"):
        weights = self.extract_weights(clients)
        # kpca = KernelPCA(n_components=n_components, kernel=kernel)
        # reduced_weights = kpca.fit_transform(weights)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(weights)
        labels = kmeans.labels_
        normal_clients = [i for i, label in enumerate(labels) if label == 0]
        malicious_clients = [i for i, label in enumerate(labels) if label == 1]
        return normal_clients, malicious_clients



    def forward(self, x, pseudo_labels, client_idx):

        local_model = self.clients[client_idx]["model"]
        images = data_augmentation(x, self.dataset_name).cuda()
        predict_logits = local_model(images)

        return predict_logits, pseudo_labels

    @torch.no_grad()
    def update_pseudo_labels(self, predict_logits, pseudo_labels, index, client_index):
        candidate_mask = (pseudo_labels > 0).float()
        z = F.one_hot(torch.argmax(predict_logits, dim=1), num_classes=100 if self.dataset_name=="cifar100" else 10).float().cuda()

        z_masked = z * candidate_mask

        pseudo_labels = 0.99 * pseudo_labels + 0.01 * z_masked

        pseudo_labels = pseudo_labels * candidate_mask
        pseudo_labels = pseudo_labels / (pseudo_labels.sum(dim=1, keepdim=True) + 1e-12)

        self.clients[client_index]["data"].dataset.dataset.update_pseudo_labels(index, pseudo_labels.cpu())

        return pseudo_labels


    @torch.no_grad()
    def targeted_flip_attack_by_target_class(self, client_idx, confused_matrix, target_class):

        print(f"Target attack on class {target_class}")

        row_of_target = confused_matrix[target_class].copy()
        row_of_target[target_class] = -1
        most_confused_class = np.argmax(row_of_target)

        print(f"[INFO] Most confused class for {target_class} is {most_confused_class}")

        client = self.clients[client_idx]

        for (data, candidate_labels, _, index) in client["data"]:
            pred_labels = torch.argmax(candidate_labels, dim=1)

            new_partial_labels = []
            for idx, pred_label in enumerate(pred_labels):
                pred_label_int = pred_label.item()

                old_dist = candidate_labels[idx].clone()

                if pred_label_int == target_class:
                    old_candidates = (old_dist > 0).nonzero().flatten().tolist()

                    if most_confused_class not in old_candidates:
                        old_candidates.append(most_confused_class)

                    new_dist = torch.zeros_like(old_dist)

                    high_prob = 0.3

                    new_dist[most_confused_class] = high_prob

                    other_candidates = [c for c in old_candidates if c != most_confused_class]
                    num_other = len(other_candidates)
                    if num_other > 0:
                        for c in other_candidates:
                            new_dist[c] = (1-high_prob) / num_other
                    else:
                        new_dist[most_confused_class] = 1.0

                    p_candidate_labels = new_dist
                else:
                    p_candidate_labels = old_dist

                new_partial_labels.append(p_candidate_labels)

            client["data"].dataset.dataset.update_pseudo_labels(index, new_partial_labels)

    def cls_loss(self, predict_logits, pseudo_labels):
        predicted_probs = torch.softmax(predict_logits, dim=1)

        log_probs = torch.log(predicted_probs + 1e-12).cuda()

        loss_per_sample = -torch.sum(pseudo_labels * log_probs, dim=1)

        loss = loss_per_sample.mean()

        return loss


    def ga_loss(self, client_idx):
        local_model = self.clients[client_idx]["model"]

        local_gradients = []
        for param in local_model.parameters():
            if param.grad is not None:
                local_gradients.append(param.grad.view(-1).detach().to(torch.device('cuda')))
        local_gradients = torch.cat(local_gradients).cuda()

        global_gradients = []
        for param in self.server.parameters():
            if param.grad is not None:
                global_gradients.append(param.grad.view(-1).detach().to(torch.device('cuda')))
        global_gradients = torch.cat(global_gradients)

        cosine_sim = nn.functional.cosine_similarity(local_gradients, global_gradients, dim=0)

        loss = -cosine_sim

        return loss



