import argparse
import os
from collections import Counter

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import defense_myCL

from model import FedPLL
from preprocess import load_data, data_augmentation, load_fedpll_split


def get_file_prefix(defence_method, dataset_name, target_class=None):
    prefix = f"{defence_method}_{dataset_name}"
    if target_class is not None:
        prefix += f"_target{target_class}"
    return prefix


def test_and_get_recall_matrix(model, test_data, dataset_name):
    model.eval()
    num_classes = 100 if dataset_name == "cifar100" else 10
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for (data, candidate_labels, true_labels, index) in test_data:
            data, true_labels = data.cuda(), true_labels.cuda()
            images = data_augmentation(data, dataset_name).cuda()
            predict_logits = model.server(images)
            predicted_labels = torch.argmax(predict_logits, dim=1)
            conf_matrix += confusion_matrix(true_labels.cpu().numpy(),
                                            predicted_labels.cpu().numpy(),
                                            labels=list(range(num_classes)))
    row_sum = conf_matrix.sum(axis=1, keepdims=True)
    recall_matrix = np.zeros_like(conf_matrix, dtype=float)
    np.divide(conf_matrix, row_sum, out=recall_matrix, where=(row_sum != 0))
    return recall_matrix

def need_log_clients(defence_method):
    return defence_method in ["PCA", "KPCA", "RobustFedPLL", "Kmeans"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'fmnist'])
    parser.add_argument('--attack_round', type=int, default=30)
    parser.add_argument('--target_class', type=int, default=3)
    parser.add_argument('--defence_method', default='RobustFedPLL', choices=['PCA', 'KPCA', 'Krum', 'Trimmed-Mean', 'Median', 'RobustFedPLL', 'none', "Kmeans", "No_CL"])
    parser.add_argument('--logdir', default='log', help='Directory to save logs')
    parser.add_argument('--ckptdir', default='checkpoints', help='Directory to save checkpoints')
    args = parser.parse_args()

    client_num = 20
    class_num = 100 if args.dataset == "cifar100" else 10
    train_data, test_data = load_data(args.dataset, 0.3, 256, client_num, False)

    model = FedPLL(args.dataset, client_num, class_num, train_data)
    confidence = defense_myCL.initialize_confidence_list([i for i in range(client_num)])
    model = model.cuda()

    attack_ratio = 0.3
    attack_client_num = int(client_num * attack_ratio)
    attack_client = np.random.choice(client_num, attack_client_num, replace=False)
    print("Attack clients:", attack_client)

    file_prefix = get_file_prefix(args.defence_method, args.dataset, args.target_class)
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.ckptdir, exist_ok=True)
    log_file = os.path.join(args.logdir, f"{file_prefix}.log")


    with open(log_file, "w") as f:
        f.write(f"attack_client: {sorted(list(attack_client))}\n")
        f.write("round,global_acc,target_recall,normal_clients,malicious_clients\n")


    optimizers = []
    for client in model.clients:
        optimizer = torch.optim.SGD(client["model"].parameters(), lr=0.03, momentum=0.5)
        optimizers.append(optimizer)

    communication_rounds = 100
    local_epoch = 3

    normal_clients, malicious_clients = [], []
    for round in range(communication_rounds):
        if round == args.attack_round:
            confused_matrix = test_and_get_recall_matrix(model, test_data, args.dataset)
            print("Confused Matrix at attack round", round)
            for idx in attack_client:
                model.targeted_flip_attack_by_target_class(idx, confused_matrix=confused_matrix, target_class=args.target_class)

        current_round_clients = np.random.choice(range(client_num), size=client_num, replace=False).tolist()
        model.update_local_weight(current_round_clients)

        for client_idx, client in enumerate(model.clients):
            if client_idx not in current_round_clients:
                continue
            optimizer = optimizers[client_idx]
            for epoch in range(local_epoch):
                for (data, candidate_label, true_label, index) in (client["data"]):
                    data, candidate_label = data.cuda(), candidate_label.cuda()
                    predict_logits, pseudo_labels = model(data, candidate_label, client_idx)
                    pseudo_labels = model.update_pseudo_labels(predict_logits, pseudo_labels, index, client_idx)

                    cls_loss = model.cls_loss(predict_logits, pseudo_labels)
                    optimizer.zero_grad()
                    total_loss = cls_loss
                    if round == 0:
                        total_loss.backward()
                    else:
                        # total_loss.backward(retain_graph=True)
                        ga_loss = model.ga_loss(client_idx)
                        total_loss += ga_loss
                        total_loss.backward()
                    optimizer.step()

        normal_clients_str, malicious_clients_str = "", ""
        if args.defence_method == "none":
            model.update_server_weight(current_round_clients, confidence, round)
        else:
            if args.defence_method in ["PCA", "KPCA", "Krum","Kmeans"]:
                normal_clients, malicious_clients = model.defence(args.defence_method, attack_clients=attack_client)
                model.update_server_weight_scale_up(normal_clients, attack_client, confidence, round)
            elif args.defence_method == "Trimmed-Mean":
                model.trimmed_mean_aggregation()
                normal_clients, malicious_clients = [], []
            elif args.defence_method == "Median":
                model.median_aggregation()
                normal_clients, malicious_clients = [], []
            elif args.defence_method == "RobustFedPLL":
                if round <= 50:
                    confidence = model.update_client_confidence(current_round_clients, confidence, round)
                malicious_clients = [idx for idx, conf in confidence.items() if conf >= 0.90]
                normal_clients = [idx for idx in range(client_num) if idx not in malicious_clients]
                model.update_server_weight(normal_clients, confidence, round)
            elif args.defence_method == "No_CL":
                if round <= 50:
                    confidence = model.update_client_confidence_no_CL(current_round_clients, confidence, round)
                malicious_clients = [idx for idx, conf in confidence.items() if conf >= 0.90]
                normal_clients = [idx for idx in range(client_num) if idx not in malicious_clients]
                model.update_server_weight(normal_clients, confidence, round)
            else:
                raise ValueError(f"Unsupported defence_method: {args.defence_method}")

        if round >= args.attack_round and need_log_clients(args.defence_method):
            normal_clients_str = str(sorted(list(normal_clients))) if normal_clients else "[]"
            malicious_clients_str = str(sorted(list(malicious_clients))) if malicious_clients else "[]"
        else:
            normal_clients_str = ""
            malicious_clients_str = ""

        total_test_correct = 0
        conf_matrix = np.zeros((class_num, class_num), dtype=int)
        with torch.no_grad():
            for (data, candidate_labels, true_labels, index) in test_data:
                data, true_labels = data.cuda(), true_labels.cuda()
                images = data_augmentation(data, args.dataset).cuda()
                predict_logits = model.server(images)
                predicted_labels = torch.argmax(predict_logits, dim=1)
                total_test_correct += (predicted_labels == true_labels).sum().item()
                conf_matrix += confusion_matrix(true_labels.cpu().numpy(),
                                                predicted_labels.cpu().numpy(),
                                                labels=list(range(class_num)))
            recalls = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
            average_recall = np.nanmean(recalls)
            print(f'Round {round+1} - Test Accuracy: {total_test_correct / len(test_data.dataset) * 100:.2f}%')
            print(f'Average Recall for all classes: {average_recall * 100:.2f}%')

            target_recall = recalls[args.target_class]
            log_line = f"{round},{total_test_correct / len(test_data.dataset):.6f},{target_recall:.6f},{normal_clients_str},{malicious_clients_str}\n"
            with open(log_file, "a") as f:
                f.write(log_line)

        checkpoint = {
            "server": model.server.state_dict(),
            "clients_head_weight": {f"client_{i}": client["model"].state_dict()['classifier.weight'] for i, client in enumerate(model.clients)},
        }
        torch.save(checkpoint, os.path.join(args.ckptdir, f"{file_prefix}_round_{round + 1}.pth"))

if __name__ == "__main__":
    main()
