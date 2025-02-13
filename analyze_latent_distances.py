import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform, cosine, mahalanobis
from scipy.spatial import ConvexHull
from itertools import combinations
import json
from scipy.stats import ttest_ind
from PIL import Image
import os

tests = ['resnet_50', 'resnet_152', 'vgg16', 'vgg19', 'inception', 'efficientnet']
mode = 'fake' # 'real' or 'fake'
algorithm = 'wgan' # 'gan' or 'wgan'

os.makedirs(f"./plots/{algorithm}/{mode}", exist_ok=True)

class_names = ['bifurca_distal', 'bifurca_proximal', 'commutata_distal', 'commutata_proximal', 
               'crozalsii_distal', 'crozalsii_proximal', 'glauca_distal', 'glauca_proximal', 
               'gothica_distal', 'gothica_proximal', 'sorocarpa_distal', 'sorocarpa_proximal', 
               'warnstorfii_distal', 'warnstorfii_proximal']

species_names = ['bifurca', 'commutata', 'crozalsii', 'glauca', 'gothica', 'sorocarpa', 'warnstorfii']
species_to_color = {species: color for species, color in zip(species_names, sns.color_palette("tab10", 7))}

def load_images_from_folder(folder, transform, max_images=None):
    images = []
    for i, filename in enumerate(os.listdir(folder)):
        if max_images and i >= max_images:
            break
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert("RGB")
        images.append(transform(img))
    return torch.stack(images)

def convex_hull_overlap(species_points):
    n = len(species_names)
    overlap_matrix = np.zeros((n, n))

    hulls = {species: ConvexHull(np.array(points)) for species, points in species_points.items()}

    for (i, species1), (j, species2) in combinations(enumerate(species_names), 2):
        hull1, hull2 = hulls[species1], hulls[species2]

        try:
            intersection = ConvexHull(np.vstack([hull1.points, hull2.points]))
            overlap_matrix[i, j] = intersection.volume / (hull1.volume + hull2.volume - intersection.volume)
        except:
            overlap_matrix[i, j] = 0 

        overlap_matrix[j, i] = overlap_matrix[i, j]

    return overlap_matrix

def store_tsne_results(real_tsne, fake_tsne):
    return {
        'real': real_tsne,
        'fake': fake_tsne,
        'combined': np.vstack([real_tsne, fake_tsne])
    }

def tsne_visualization(real_features, fake_features):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    combined_features = np.vstack([real_features, fake_features])
    tsne_results = tsne.fit_transform(combined_features)
    real_tsne = tsne_results[:len(real_features)]
    fake_tsne = tsne_results[len(real_features):]
    
       
    return store_tsne_results(real_tsne, fake_tsne)

def calculate_interclass_centroid_distances(tsne_results_dict):
    class_names = list(tsne_results_dict.keys())
    num_classes = len(class_names)
    
    real_centroids = {cls: np.mean(tsne_results_dict[cls]['real'], axis=0) for cls in class_names}
    fake_centroids = {cls: np.mean(tsne_results_dict[cls]['fake'], axis=0) for cls in class_names}
    
    real_distance_matrix = np.zeros((num_classes, num_classes))
    fake_distance_matrix = np.zeros((num_classes, num_classes))

    real_distances = []
    fake_distances = []
    
    for i, class1 in enumerate(class_names):
        for j, class2 in enumerate(class_names):
            real_dist = np.linalg.norm(real_centroids[class1] - real_centroids[class2])
            fake_dist = np.linalg.norm(fake_centroids[class1] - fake_centroids[class2])
            
            real_distance_matrix[i, j] = real_dist
            fake_distance_matrix[i, j] = fake_dist
            
            if i < j:
                real_distances.append(real_dist)
                fake_distances.append(fake_dist)

    # t-test
    t_stat, p_value = ttest_ind(real_distances, fake_distances, equal_var=False)
    
    return {
        'real_mean_centroid_dist': np.mean(real_distances),
        'real_std_centroid_dist': np.std(real_distances),
        'fake_mean_centroid_dist': np.mean(fake_distances),
        'fake_std_centroid_dist': np.std(fake_distances),
        't_stat': t_stat,
        'p_value': p_value,
        'class_names': class_names,
    }

def calculate_clustering_metrics(tsne_results_dict, class_name):
    """Calculate various metrics to quantify clustering quality and separation."""
    real_tsne = tsne_results_dict[class_name]['real']
    fake_tsne = tsne_results_dict[class_name]['fake']
    
    metrics = {}
    
    # Intra-class distances
    def calc_intra_class_distance(data):
        if len(data) < 2:
            return 0
        distances = pdist(data)
        return np.mean(distances)
    
    metrics['real_intra_dist'] = calc_intra_class_distance(real_tsne)
    metrics['fake_intra_dist'] = calc_intra_class_distance(fake_tsne)
    
    # Cluster density
    def calc_cluster_density(data):
        centroid = np.mean(data, axis=0)
        distances_to_centroid = np.sqrt(np.sum((data - centroid) ** 2, axis=1))
        return np.std(distances_to_centroid).astype(float)
    
    metrics['real_density'] = calc_cluster_density(real_tsne)
    metrics['fake_density'] = calc_cluster_density(fake_tsne)

    return metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for test in tests:
    print(f"Processing model: {test}")

    if test == 'resnet_50':
        model = models.resnet50(pretrained=True).to(device)
        model.fc = nn.Identity()

    elif test == 'resnet_152':
        model = models.resnet152(pretrained=True).to(device)
        model.fc = nn.Identity()

    elif test == 'vgg16':
        model = models.vgg16(pretrained=True).to(device)
        model.classifier[6] = nn.Identity()

    elif test == 'vgg19':
        model = models.vgg19(pretrained=True).to(device)
        model.classifier[6] = nn.Identity()

    elif test == 'inception':
        model = models.inception_v3(pretrained=True, transform_input=False).to(device)
        model.fc = nn.Identity()

    elif test == 'efficientnet':
        model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True).to(device)
        model.classifier = nn.Identity()

    model.eval()

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if test == 'vgg16' or test == 'vgg19':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if mode == 'real':
        data_dir = "./local/real_imgs/"
    elif mode == 'fake':
        data_dir = f"./local//{algorithm}/generated_imgs/"
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    features = []
    labels = []

    with torch.no_grad():
        for images, target in dataloader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.append(target.numpy())

    features = np.vstack(features)
    labels = np.hstack(labels)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)

    centroids = {}
    species_points = {species: [] for species in species_names}

    for i, class_name in enumerate(class_names):
        species = class_name.split('_')[0]
        indices = labels == i
        points = features_2d[indices]

        if species not in centroids:
            centroids[species] = []
        centroids[species].append(points.mean(axis=0))
        species_points[species].extend(points)

    for species in centroids:
        centroids[species] = np.mean(centroids[species], axis=0)

    centroid_matrix = np.array(list(centroids.values()))

    distance_metrics = {
        "Euclidean Distance": squareform(pdist(centroid_matrix, metric='euclidean')),
        "Cosine Distance": squareform(pdist(centroid_matrix, metric='cosine')),
        "Mahalanobis Distance": squareform(pdist(centroid_matrix, metric=lambda u, v: mahalanobis(u, v, np.linalg.inv(np.cov(centroid_matrix.T))))),
        "Convex Hull Overlap": convex_hull_overlap(species_points)
    }

    for metric_name, matrix in distance_metrics.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=species_names, yticklabels=species_names)
        plt.title(f"{test} - {metric_name}")
        plt.xlabel("Species")
        plt.ylabel("Species")
        plt.tight_layout()
        plot_filename = f"./plots/{algorithm}/{mode}/{test}_{metric_name.replace(' ', '_').lower()}_plot.png"
        plt.savefig(plot_filename)
        plt.close()

    distance_metrics_json = {metric: matrix.tolist() for metric, matrix in distance_metrics.items()}

    print(f"Saved distance metrics as JSON for {test}")

    plt.figure(figsize=(10, 8))

    for i, class_name in enumerate(class_names):
        indices = labels == i
        species = class_name.split('_')[0]  # Extract species name
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1], 
                    label=species if species not in plt.gca().get_legend_handles_labels()[1] else "_nolegend_", 
                    s=50, alpha=0.7, color=species_to_color[species])

    for species, centroid in centroids.items():
        plt.scatter(centroid[0], centroid[1], 
                    color=species_to_color[species], edgecolors='black', s=200, marker='X', label="_nolegend_")

    plt.legend(title="Species", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"t-SNE Visualization of Feature Embeddings - {test}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plot_filename = f"./plots/{algorithm}/{mode}/TSNE_{test}__plot.png"
    #plt.savefig(plot_filename)
    plt.show()

    tsne_results_dict = {}
    for name in class_names:
        real_data_dir = f"./local/real_imgs/{name}"
        fake_data_dir = f"./local/{algorithm}/generated_imgs/{name}"

        real_images = load_images_from_folder(real_data_dir, transform)
        fake_images = load_images_from_folder(fake_data_dir, transform)

        with torch.no_grad():
                real_features = model(real_images.to(device)).cpu().numpy()
                fake_features = model(fake_images.to(device)).cpu().numpy()

        tsne_results_dict[name] = tsne_visualization(real_features, fake_features)

    centroid_distances = calculate_interclass_centroid_distances(tsne_results_dict)

    print(f"Centroid Distances for {test}:")
    print(f"Real Mean: {centroid_distances['real_mean_centroid_dist']:.3f}")
    print(f"Real Std: {centroid_distances['real_std_centroid_dist']:.3f}")
    print(f"Fake Mean: {centroid_distances['fake_mean_centroid_dist']:.3f}")
    print(f"Fake Std: {centroid_distances['fake_std_centroid_dist']:.3f}")
    print(f"t-statistic: {centroid_distances['t_stat']:.3f}")
    print(f"p-value: {centroid_distances['p_value']:.3f}")
    print('')

    all_metrics = {}
    for name in tsne_results_dict.keys():
        intraclass_metrics = calculate_clustering_metrics(tsne_results_dict, name)
        all_metrics[name] = intraclass_metrics

    summary = {
        'avg_real_intra_dist': np.mean([m['real_intra_dist'] for m in all_metrics.values()]),
        'avg_fake_intra_dist': np.mean([m['fake_intra_dist'] for m in all_metrics.values()]),
        'std_real_intra_dist': np.std([m['real_intra_dist'] for m in all_metrics.values()]),
        'std_fake_intra_dist': np.std([m['fake_intra_dist'] for m in all_metrics.values()]),
        'avg_real_density': np.mean([m['real_density'] for m in all_metrics.values()]),
        'avg_fake_density': np.mean([m['fake_density'] for m in all_metrics.values()]),
        'std_real_density': np.std([m['real_density'] for m in all_metrics.values()]),
        'std_fake_density': np.std([m['fake_density'] for m in all_metrics.values()]),
    }

    print(f"Average Real Intra-Class Distance: {summary['avg_real_intra_dist']:.3f}")
    print(f"Average Fake Intra-Class Distance: {summary['avg_fake_intra_dist']:.3f}")
    print(f"Standard Deviation Real Intra-Class Distance: {summary['std_real_intra_dist']:.3f}")
    print(f"Standard Deviation Fake Intra-Class Distance: {summary['std_fake_intra_dist']:.3f}")
    print(f"Average Real Cluster Density: {summary['avg_real_density']:.3f}")
    print(f"Average Fake Cluster Density: {summary['avg_fake_density']:.3f}")
    print(f"Standard Deviation Real Cluster Density: {summary['std_real_density']:.3f}")
    print(f"Standard Deviation Fake Cluster Density: {summary['std_fake_density']:.3f}")

    combined_results = {
        'intraclass__metrics': summary,
        'distance_metrics': distance_metrics_json,
        'centroid_distances': centroid_distances
    }

    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        else:
            return obj

    combined_results = make_serializable(combined_results)

    # Save Results as json-file
    combined_json_filename = f"./plots/{algorithm}/{mode}/{test}_combined_metrics.json"
    with open(combined_json_filename, 'w') as combined_json_file:
        json.dump(combined_results, combined_json_file)

    print(f"Saved combined metrics as JSON for {test}")