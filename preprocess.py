import torch
from torchvision.transforms import RandAugment
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from CustomDataset import CustomDataset


# confused_map = {
#     0: 2,  # airplane → bird
#     1: 9,  # automobile → truck
#     2: 0,  # bird → airplane
#     3: 5,  # cat → dog
#     4: 7,  # deer → horse
#     5: 3,  # dog → cat
#     6: 8,  # frog → ship
#     7: 4,  # horse → deer
#     8: 6,  # ship → frog
#     9: 1,  # truck → automobile
# }

confused_map = {
    0: 6,  # T-shirt/top → Shirt
    1: 9,  # Trouser → Ankle boot
    2: 4,  # Pullover → Coat
    3: 5,  # Dress → Sandal
    4: 2,  # Coat → Pullover
    5: 3,  # Sandal → Dress
    6: 0,  # Shirt → T-shirt/top
    7: 9,  # Sneaker → Ankle boot
    8: 7,  # Bag → Sneaker
    9: 1,  # Ankle boot → Trouser
}




def load_data(dataset, partial_rate, batch_size, client_num, non_iid):
    train_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    if dataset == 'mnist':
        train_data = datasets.FashionMNIST(root="./data", train=True, download=True, transform=train_transforms)
        test_data = datasets.FashionMNIST(root="./data", train=False, download=False, transform=test_transforms)
    elif dataset == 'cifar10':
        train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transforms)
        test_data = datasets.CIFAR10(root="./data", train=False, download=False, transform=test_transforms)
    else:
        train_data = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transforms)
        test_data = datasets.CIFAR100(root="./data", train=False, download=False, transform=test_transforms)

    if isinstance(train_data.targets, list):
        train_true_label = torch.tensor(train_data.targets.copy())
        test_true_label = torch.tensor(test_data.targets.copy())
    else:
        train_true_label = train_data.targets.detach().clone()
        test_true_label = test_data.targets.detach().clone()

    train_data.targets = generate_candidate_labels_by_class(train_data.targets, partial_rate)
    test_data.targets = generate_candidate_labels_by_class(test_data.targets, partial_rate)
    # train_data.targets = generate_candidate_labels_with_confused_map(train_data.targets, confused_map, partial_rate)
    # test_data.targets = generate_candidate_labels_with_confused_map(test_data.targets, confused_map, partial_rate)

    train_dataset = CustomDataset(train_data, train_data.targets, train_true_label, transform=None)
    test_dataset = CustomDataset(test_data, test_data.targets, test_true_label, transform=None)

    data_per_class = {i: [] for i in range(100 if dataset == 'cifar100' else 10)}
    for idx, label in enumerate(train_true_label):
        data_per_class[label.item()].append(idx)

    if non_iid:
        client_data = distribute_data_non_iid(M=client_num, L=10, sigma=0.5, beta=0.5, data_per_class=data_per_class,
                                          batch_size=batch_size)
    else:
        client_data = distribute_data_iid(M=client_num, data_per_class=data_per_class, batch_size=batch_size)

    train_loaders = []
    for client_idx in range(client_num):
        client_indices = client_data[client_idx]
        client_dataset = torch.utils.data.Subset(train_dataset, client_indices)
        client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        train_loaders.append(client_loader)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loaders, test_loader


def generate_candidate_labels(targets, partial_rate):
    label_size = 10
    partial_labels = []

    for target in targets:
        candidate_labels = np.random.choice(
            range(label_size),
            size=int(partial_rate * label_size),
            replace=False
        ).tolist()
        if target not in candidate_labels:
            candidate_labels = candidate_labels[:-1]
            candidate_labels.append(target)
        p_candidate_labels = torch.zeros(label_size)
        for index in range(label_size):
            if index in candidate_labels:
                p_candidate_labels[index] = 1.0 / int(partial_rate * label_size)
        partial_labels.append(p_candidate_labels)

    return partial_labels


def generate_candidate_labels_with_confused_map(train_labels, confused_map=confused_map, partial_rate=0.3, confused_prob=0.5):
    if not isinstance(train_labels, torch.Tensor):
        train_labels = torch.tensor(train_labels)

    if torch.min(train_labels) > 1:
        raise RuntimeError('testError: label min > 1')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    label_size = int(torch.max(train_labels).item() - torch.min(train_labels).item() + 1)
    num_instances = train_labels.shape[0]

    partial_labels = []

    for i in range(num_instances):
        true_label = train_labels[i].item()
        confused_label = confused_map[true_label]
        candidate_mask = np.zeros(label_size, dtype=np.float32)
        candidate_mask[true_label] = 1

        if np.random.rand() < confused_prob:
            candidate_mask[confused_label] = 1

        for j in range(label_size):
            if j != true_label and j != confused_label:
                if np.random.rand() < partial_rate:
                    candidate_mask[j] = 1

        if candidate_mask.sum() == label_size:
            others = [j for j in range(label_size) if j != true_label and candidate_mask[j]]
            candidate_mask[np.random.choice(others)] = 0

        candidate_distribution = candidate_mask / candidate_mask.sum()
        partial_labels.append(candidate_distribution)

    print("Finished Generating Candidate Label Sets (confused label with prob).")
    return partial_labels

def generate_candidate_labels_by_class(train_labels, partial_rate=0.3):
    if not isinstance(train_labels, torch.Tensor):
        train_labels = torch.tensor(train_labels)

    if torch.min(train_labels) > 1:
        raise RuntimeError('testError: label min > 1')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    label_size = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    num_instances = train_labels.shape[0]

    partial_labels = []

    flip_matrix = np.eye(label_size)
    flip_matrix[np.where(~np.eye(label_size, dtype=bool))] = partial_rate

    random_values = np.random.uniform(0, 1, size=(num_instances, label_size))

    for i in range(num_instances):
        true_label = train_labels[i].item()
        candidate_mask = (random_values[i, :] < flip_matrix[true_label, :])

        # Ensure true label is always in the candidate set
        candidate_mask[true_label] = 1

        # Avoid selecting all classes
        if candidate_mask.sum() == label_size:
            non_true_labels = [j for j in range(label_size) if j != true_label and candidate_mask[j]]
            candidate_mask[np.random.choice(non_true_labels)] = 0

        # Normalize to get probability distribution
        candidate_distribution = candidate_mask / candidate_mask.sum()
        partial_labels.append(candidate_distribution)

    print("Finished Generating Candidate Label Sets.")
    return partial_labels


def generate_candidate_labels_by_superclass(train_labels, superclass_mapping, partial_rate=0.2):
    if not isinstance(train_labels, torch.Tensor):
        train_labels = torch.tensor(train_labels)

    if torch.min(train_labels) > 1:
        raise RuntimeError('testError: label min > 1')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    label_size = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    num_instances = train_labels.shape[0]

    partial_labels = []

    superclass_to_classes = {}
    for cls, supercls in superclass_mapping.items():
        superclass_to_classes.setdefault(supercls, []).append(cls)

    random_values = np.random.uniform(0, 1, size=(num_instances, label_size))

    for i in range(num_instances):
        true_label = train_labels[i].item()
        supercls = superclass_mapping[true_label]
        valid_classes = superclass_to_classes[supercls]

        candidate_mask = np.zeros(label_size)

        for cls in valid_classes:
            if cls == true_label:
                candidate_mask[cls] = 1
            else:
                if random_values[i, cls] < partial_rate:
                    candidate_mask[cls] = 1

        if candidate_mask.sum() == 1 and len(valid_classes) > 1:
            candidates = [cls for cls in valid_classes if cls != true_label]
            random_cls = np.random.choice(candidates)
            candidate_mask[random_cls] = 1

        candidate_distribution = candidate_mask / candidate_mask.sum()
        partial_labels.append(candidate_distribution)

    print("Finished Generating Subclass-Based Candidate Label Sets.")
    return partial_labels


def distribute_data_non_iid(M=10, L=10, sigma=0.5, beta=0.5, data_per_class=None, batch_size=256):
    phi = np.random.binomial(1, sigma, size=(M, L))
    client_data = {i: [] for i in range(M)}

    for j in range(L):
        clients_with_class_j = np.where(phi[:, j] == 1)[0]
        uj = len(clients_with_class_j)
        if uj == 0:
            continue

        vj = np.random.dirichlet([beta] * uj)
        class_j_data = data_per_class[j].copy()
        np.random.shuffle(class_j_data)
        num_samples = len(class_j_data)

        samples_per_client = (vj * num_samples).astype(int)

        diff = num_samples - samples_per_client.sum()
        while diff > 0:
            for i in range(len(samples_per_client)):
                samples_per_client[i] += 1
                diff -= 1
                if diff == 0:
                    break
        while diff < 0:
            for i in range(len(samples_per_client)):
                if samples_per_client[i] > 0:
                    samples_per_client[i] -= 1
                    diff += 1
                    if diff == 0:
                        break

        # Distribute the data to clients
        idx = 0
        for client_idx, num_samples_client in zip(clients_with_class_j, samples_per_client):
            client_data[client_idx].extend(class_j_data[idx:idx + num_samples_client])
            idx += num_samples_client

    return client_data


def distribute_data_iid(M=10, data_per_class=None, batch_size=256):
    client_data = {i: [] for i in range(M)}

    all_data = []
    for class_data in data_per_class.values():
        all_data.extend(class_data)

    np.random.shuffle(all_data)

    num_samples = len(all_data)
    samples_per_client = num_samples // M
    remainder = num_samples % M

    idx = 0
    for i in range(M):
        client_data[i].extend(all_data[idx:idx + samples_per_client])
        idx += samples_per_client

    for i in range(remainder):
        client_data[i].append(all_data[idx])
        idx += 1

    return client_data


def get_mnist_augment():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def get_cifar_augment():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

def get_augmentation(dataset_name):
    if dataset_name.lower() == 'mnist':
        return get_mnist_augment()
    elif dataset_name.lower() in ['cifar10', 'cifar100']:
        return get_cifar_augment()
    else:
        raise NotImplementedError("No augmentation defined for this dataset.")

def data_augmentation(images, dataset_name='cifar10'):
    sim_augment = get_augmentation(dataset_name)

    if dataset_name.lower() == 'mnist':
        images = torch.stack([sim_augment(img) for img in images])
    elif dataset_name.lower() == 'cifar10':
        images = torch.stack([sim_augment(img.permute(2, 1, 0)) for img in images])
    else:
        images = images.permute(0, 3, 1, 2)
        images = torch.stack([sim_augment(img) for img in images])

    return images

def load_fedpll_split(
    dataset,
    batch_size,
    save_path,
    train_transform=None,
    test_transform=None
):
    import pickle
    from torch.utils.data import DataLoader, Subset
    from CustomDataset import CustomDataset
    import torchvision.datasets as datasets

    with open(save_path, 'rb') as f:
        obj = pickle.load(f)
    client_indices = obj['client_indices']
    train_candidate_labels = obj['train_candidate_labels']
    train_true_labels = obj['train_true_labels']
    test_candidate_labels = obj['test_candidate_labels']
    test_true_labels = obj['test_true_labels']
    meta_info = obj.get('meta_info', None)

    # 重建原始数据集
    if dataset == 'mnist':
        train_raw = datasets.FashionMNIST(root="./data", train=True, download=True, transform=train_transform)
        test_raw = datasets.FashionMNIST(root="./data", train=False, download=True, transform=test_transform)
    elif dataset == 'cifar10':
        train_raw = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        test_raw = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    else:
        train_raw = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
        test_raw = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)

    train_dataset = CustomDataset(train_raw, train_candidate_labels, train_true_labels, transform=None)
    test_dataset = CustomDataset(test_raw, test_candidate_labels, test_true_labels, transform=None)

    train_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
        train_loaders.append(loader)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loaders, test_loader, meta_info
