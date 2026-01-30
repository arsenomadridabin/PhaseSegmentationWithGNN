from sklearn.metrics import roc_curve, auc
import os
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_dilation, label
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GATConv
from scipy.stats import gaussian_kde
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from collections import defaultdict
# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------------
# GNN Model
# ------------------------------
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, 2, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
# BUild graph


def build_graph(atom_data, atom_labels, radius=6.0):
    from scipy.spatial import cKDTree

    positions = atom_data[:, 2:5]
    atom_types = atom_data[:, 1].astype(int)
    box_min = positions.min(0)
    box_max = positions.max(0)
    box_length = box_max - box_min
    N = positions.shape[0]

    positions_wrapped = (positions - box_min) % box_length
    tree = cKDTree(positions_wrapped, boxsize=box_length)
    all_nbrs = tree.query_ball_point(positions_wrapped, r=radius)

    fe_neighbor_counts = np.zeros(N, dtype=int)
    mg_neighbor_counts = np.zeros(N, dtype=int)
    fe_dist_sums = np.zeros(N, dtype=float)
    fe_dist_vars = np.zeros(N, dtype=float)
    fe_fraction = np.zeros(N, dtype=float)

    for i, nbrs in enumerate(all_nbrs):
        nbrs = [j for j in nbrs if j != i]
        if not nbrs:
            continue
        deltas = positions_wrapped[nbrs] - positions_wrapped[i]
        deltas -= box_length * np.round(deltas / box_length)
        dists = np.linalg.norm(deltas, axis=1)
        types = atom_types[nbrs]
        fe_mask = types == 1
        mg_mask = types == 2
        dists_fe = dists[fe_mask]
        fe_neighbor_counts[i] = len(dists_fe)
        mg_neighbor_counts[i] = mg_mask.sum()
        if len(dists_fe) > 0:
            fe_dist_sums[i] = dists_fe.mean()
            fe_dist_vars[i] = dists_fe.var()
        fe_fraction[i] = len(dists_fe) / len(nbrs) if len(nbrs) > 0 else 0.0

    edge_index = []
    for i, nbrs in enumerate(all_nbrs):
        for j in nbrs:
            if j <= i:
                continue
            edge_index.append((i, j))
            edge_index.append((j, i))
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    one_hot = np.eye(5)[atom_types - 1]
    features = np.concatenate([
        one_hot,
        fe_neighbor_counts[:, None],
        fe_dist_sums[:, None],
        fe_dist_vars[:, None],
        fe_fraction[:, None],
        mg_neighbor_counts[:, None]
    ], axis=1)

    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(atom_labels, dtype=torch.long)
    mask = torch.ones_like(y, dtype=torch.bool)

    return Data(x=x, edge_index=edge_index, y=y, mask=mask)
# ------------------------------
# KDE + KMeans Labeling (2-class only)
# ------------------------------

def extend_periodic_positions(fe_positions, box_length):
    shifts = [-1, 0, 1]
    shift_vectors = np.array([[i, j, k] for i in shifts for j in shifts for k in shifts])
    all_fe = [fe_positions + shift * box_length for shift in shift_vectors]
    return np.vstack(all_fe)

def compute_kde_density(fe_positions, box_min, box_max, num_bins, bandwidth=0.1):
    kde = gaussian_kde(fe_positions.T, bw_method=bandwidth)
    x = np.linspace(box_min[0], box_max[0], num_bins[0])
    y = np.linspace(box_min[1], box_max[1], num_bins[1])
    z = np.linspace(box_min[2], box_max[2], num_bins[2])
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    return kde(grid_coords).reshape(num_bins)


def compute_2class_labels(atom_data, num_bins=(50, 50, 50), min_voxels=100):
    positions = atom_data[:, 2:5]
    atom_types = atom_data[:, 1].astype(int)
    fe_positions = positions[atom_types == 1]

    box_min = positions.min(0)
    box_max = positions.max(0)
    box_length = box_max - box_min

    fe_positions_wrapped = extend_periodic_positions(fe_positions, box_length)
    fe_density = compute_kde_density(fe_positions_wrapped, box_min, box_max, num_bins, bandwidth=0.1)

    # --- KMeans on density values ---
    flat_density = fe_density.flatten()
    valid = flat_density >= 0
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = kmeans.fit_predict(flat_density[valid].reshape(-1, 1))
    sorted_centers = np.argsort(kmeans.cluster_centers_.flatten())
    label_map = {sorted_centers[0]: 0, sorted_centers[1]: 1}
    binary_mask_flat = np.full(flat_density.shape, -1)
    binary_mask_flat[valid] = np.vectorize(label_map.get)(labels)
    binary_mask = binary_mask_flat.reshape(fe_density.shape)

    # --- Keep only largest Fe-rich component ---
    def get_largest_component(mask):
        padded = np.pad(mask, pad_width=1, mode='wrap')
        labeled, _ = label(padded)
        cropped = labeled[1:-1, 1:-1, 1:-1]

        face_pairs = []
        shape = mask.shape
        for axis, size in enumerate(shape):
            for idx in np.ndindex(*[s for i, s in enumerate(shape) if i != axis]):
                slicer = list(idx)
                slicer.insert(axis, 0)
                l1 = labeled[tuple(np.array(slicer) + 1)]
                slicer[axis] = size - 1
                l2 = labeled[tuple(np.array(slicer) + 1)]
                if l1 != 0 and l2 != 0 and l1 != l2:
                    face_pairs.append((l1, l2))

        parent = {}
        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(x, y):
            xr, yr = find(x), find(y)
            if xr != yr:
                parent[yr] = xr

        for a, b in face_pairs:
            union(a, b)

        label_map = defaultdict(list)
        for val in np.unique(cropped):
            if val != 0:
                label_map[find(val)].append(val)

        label_counts = {
            root: np.count_nonzero(np.isin(cropped, labels))
            for root, labels in label_map.items()
        }

        if not label_counts:
            return np.zeros_like(cropped, dtype=bool)

        largest_root = max(label_counts, key=label_counts.get)
        return np.isin(cropped, label_map[largest_root])

    def remove_small_components(mask):
        padded = np.pad(mask, pad_width=1, mode='wrap')
        labeled, _ = label(padded)
        cropped = labeled[1:-1, 1:-1, 1:-1]
        counts = np.bincount(cropped.ravel())
        keep = np.where(counts >= min_voxels)[0]
        return np.isin(cropped, keep)

    rich_mask = get_largest_component(binary_mask == 1)
    poor_mask = ~rich_mask

    # --- Final label grid: 1 = Fe-rich, 0 = Fe-poor
    region_labels = np.full(fe_density.shape, 0, dtype=int)
    region_labels[rich_mask] = 1

    # --- Assign atom labels
    x_edges = np.linspace(box_min[0], box_max[0], num_bins[0] + 1)
    y_edges = np.linspace(box_min[1], box_max[1], num_bins[1] + 1)
    z_edges = np.linspace(box_min[2], box_max[2], num_bins[2] + 1)
    x_bin = np.clip(np.digitize(positions[:, 0], x_edges) - 1, 0, num_bins[0] - 1)
    y_bin = np.clip(np.digitize(positions[:, 1], y_edges) - 1, 0, num_bins[1] - 1)
    z_bin = np.clip(np.digitize(positions[:, 2], z_edges) - 1, 0, num_bins[2] - 1)

    atom_labels = region_labels[x_bin, y_bin, z_bin]
    return atom_labels

# ------------------------------
# Model and Evaluation Utilities
# ------------------------------
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import binary_dilation, label
import numpy as np

# ------------------------------
# Weight Percent Calculation
# ------------------------------
def calculate_weight_percent(atom_data, region_labels):
    atomic_weights = {
        1: 55.845,  # Fe
        2: 24.305,  # Mg
        3: 28.085,  # Si
        4: 15.999,  # O
        5: 14.007   # N
    }

    atom_types = atom_data[:, 1].astype(int)

    rich_mask = region_labels == 1
    poor_mask = region_labels == 0

    def compute_wt_percent(mask):
        type_counts = {}
        total_mass = 0.0
        for t in np.unique(atom_types):
            count = np.sum((atom_types == t) & mask)
            mass = count * atomic_weights.get(t, 0)
            type_counts[t] = mass
            total_mass += mass
        wt_percent = {t: (m / total_mass * 100) for t, m in type_counts.items() if total_mass > 0}
        return wt_percent

    rich_wt = compute_wt_percent(rich_mask)
    poor_wt = compute_wt_percent(poor_mask)

    return rich_wt, poor_wt

# ------------------------------
# Weight Percent from Dump
# ------------------------------
def compute_weight_percent_from_dump(dump_snapshots):
    all_rich_wts = []
    all_poor_wts = []

    for i, atom_data in enumerate(dump_snapshots):
        pred_labels = compute_2class_labels(atom_data)
        positions = atom_data[:,2:5]
        boundary_labels = add_boundary_labels(pred_labels, atom_data[:, 2:5], snapshot_idx=i)
        rich_wt, poor_wt = calculate_weight_percent(atom_data, boundary_labels)
        all_rich_wts.append(rich_wt)
        all_poor_wts.append(poor_wt)
        plot_middle_z_slices(positions, labels_with_boundary, snapshot_idx=i)

    def average_wt_percent(wt_list):
        total = {}
        for d in wt_list:
            for k, v in d.items():
                total[k] = total.get(k, 0) + v
        return {k: v / len(wt_list) for k, v in total.items()}

    avg_rich = average_wt_percent(all_rich_wts)
    avg_poor = average_wt_percent(all_poor_wts)

    return all_rich_wts, all_poor_wts, avg_rich, avg_poor

# ------------------------------
# Post-process to Add Boundary with Dilation
# ------------------------------
def add_boundary_labels(pred_labels, positions, num_bins=(50, 50, 50), dilation_iter=1, snapshot_idx=None):
    box_min = positions.min(0)
    box_max = positions.max(0)
    x_bins = np.linspace(box_min[0], box_max[0], num_bins[0])
    y_bins = np.linspace(box_min[1], box_max[1], num_bins[1])
    z_bins = np.linspace(box_min[2], box_max[2], num_bins[2])

    x_idx = np.clip(np.digitize(positions[:, 0], x_bins) - 1, 0, num_bins[0] - 1)
    y_idx = np.clip(np.digitize(positions[:, 1], y_bins) - 1, 0, num_bins[1] - 1)
    z_idx = np.clip(np.digitize(positions[:, 2], z_bins) - 1, 0, num_bins[2] - 1)

    region_grid = np.full(num_bins, -1, dtype=int)
    print("pred_labels length-",len(pred_labels))
    for i in range(len(pred_labels)):
        region_grid[x_idx[i], y_idx[i], z_idx[i]] = pred_labels[i]

    rich_mask = region_grid == 1
    poor_mask = region_grid == 0

    dilated_rich = binary_dilation(rich_mask, iterations=dilation_iter)
    dilated_poor = binary_dilation(poor_mask, iterations=dilation_iter)

    boundary_mask = dilated_rich & dilated_poor

    final_grid = np.full_like(region_grid, -1)
    final_grid[boundary_mask] = 2
    final_grid[rich_mask & ~boundary_mask] = 1
    final_grid[poor_mask & ~boundary_mask] = 0
    
    final_grid_1_count = np.sum(final_grid == 1)
    final_grid_0_count = np.sum(final_grid == 0)

    print("final grid 1 count =:",final_grid_1_count)
    print("final grid 0 count = ",final_grid_0_count)
    print("final grid 2 count =- ", np.sum(final_grid == 2))
    print("final grid -1 count =",np.sum(final_grid == -1))
    final_labels = final_grid[x_idx, y_idx, z_idx]
    print("final_labels=",final_labels)
    # --- Slice and plot central Z layer ---
    central_z = final_grid.shape[2] // 2
    cmap = mcolors.ListedColormap(["black", "yellow", "red"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6, 6))
    plt.imshow(final_grid[:, :, central_z].T, origin='lower', cmap=cmap, norm=norm)
    plt.title(f"Central Z Slice" if snapshot_idx is None else f"Snapshot {snapshot_idx}: Central Z Slice")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(ticks=[0, 1, 2], label="Label")
    plt.tight_layout()
    filename = "central_z_slice.png" if snapshot_idx is None else f"snapshot_{snapshot_idx}_central_z_slice.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    return final_labels

# Training, Evaluation, Inference
# ------------------------------
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.mask], data.y[data.mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    from sklearn.metrics import confusion_matrix, f1_score
    model.eval()
    all_preds, all_labels = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu().numpy()
        label = data.y.cpu().numpy()
        mask = data.mask.cpu().numpy()
        all_preds.extend(pred[mask])
        all_labels.extend(label[mask])
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=None)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return acc, f1, cm

def infer(model, atom_data, device, idx=0, label_dir="labels", threshold=0.5):
    model.eval()
    os.makedirs(label_dir, exist_ok=True)
    label_path = os.path.join(label_dir, f"kmeans_labels_snapshot_{idx}.npy")
    if os.path.exists(label_path):
        labels = np.load(label_path)
    else:
        labels = compute_2class_labels(atom_data)
        np.save(label_path, labels)

    graph = build_graph(atom_data, labels)
    graph = graph.to(device)

    with torch.no_grad():
        out = model(graph.x, graph.edge_index)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        scores = probs[:, 1]  # Fe-rich class probability
        pred = (scores >= threshold).astype(int)

    return pred, atom_data[:, 2:5], labels


# ------------------------------
# Save/Load Utilities
# ------------------------------
def save_model(model, path="gnn_model.pt"):
    torch.save(model.state_dict(), path)

def load_model(model_class, in_channels, path="gnn_model.pt"):
    model = model_class(in_channels=in_channels)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

# ------------------------------
# Run Training Entry Point
# ------------------------------
def run_training(snapshots, epochs=100, batch_size=10, save_path="gnn_model.pt",label_dir="training_labels"):
    os.makedirs(label_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs = []
    for i,snap in enumerate(snapshots):
        label_path = os.path.join(label_dir, f"kmeans_labels_snapshot_{i}.npy")
        if os.path.exists(label_path):
            labels = np.load(label_path)
            print(f"Loaded KMeans labels for snapshot {i}")
        else:
            labels = compute_2class_labels(snap)
            np.save(label_path, labels)
            print(f"Computed and saved KMeans labels for snapshot {i}")
        graph = build_graph(snap, labels)
        graphs.append(graph)

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    model = GNN(in_channels=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        loss = train(model, loader, optimizer, criterion, device)
        acc, f1, cm = evaluate(model, loader, device)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.4f}, F1={f1},\nConfusion Matrix:\n{cm}")

    save_model(model, save_path)
    return model


# ------------------------------
# Evaluation on New Data (Considering BOundary Too)
# ------------------------------
def run_evaluation_on_new_data_v2(model_path, test_snapshots):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(GNN, in_channels=10, path=model_path).to(device)
    for i, snap in enumerate(test_snapshots):
        pred, positions, true_labels = infer(model, snap, device, i)

        # Apply post-hoc 3-class labeling
        pred_labels_with_boundary = add_boundary_labels(pred, positions)
        true_labels_with_boundary = add_boundary_labels(true_labels, positions)

        np.save(f"predicted_labels_snapshot_{i}.npy", pred_labels_with_boundary)
        np.save(f"predicted_labels_snapshot_true_{i}.npy", true_labels_with_boundary)
        print(f"Saved prediction for snapshot {i} with shape {pred_labels_with_boundary.shape}")

        # Evaluation metrics
        acc = accuracy_score(true_labels_with_boundary, pred_labels_with_boundary)
        f1 = f1_score(true_labels_with_boundary, pred_labels_with_boundary, average=None)
        cm = confusion_matrix(true_labels_with_boundary, pred_labels_with_boundary)

        print(f"Snapshot {i}: Accuracy = {acc:.4f}, F1 = {f1}, Confusion Matrix:\n{cm}")
# ------------------------------
# Evaluation on New Data Entry Point
# ------------------------------
def run_evaluation_on_new_data(model_path, test_snapshots, save_dir='labels'):
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, roc_curve, auc
    )
    import pandas as pd
    import numpy as np
    import os
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(GNN, in_channels=10, path=model_path).to(device)

    for i, atom_data in enumerate(test_snapshots):
        positions = atom_data[:, 2:5]

        # === Load or compute weak KMeans labels
        label_path = f"labels/kmeans_labels_snapshot_{i}.npy"
        if os.path.exists(label_path):
            true_labels = np.load(label_path)
        else:
            true_labels = compute_2class_labels(atom_data)
            np.save(label_path, true_labels)

        # === Run GNN inference
        graph = build_graph(atom_data, true_labels)
        graph = graph.to(device)
        with torch.no_grad():
            out = model(graph.x, graph.edge_index)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        scores = probs[:, 1]  # Fe-rich class probability

        # === Compute 3-class labels with boundaries
        true_labels_with_boundary = add_boundary_labels(true_labels, positions, snapshot_idx=i)

        # === Mask for valid labels (0 or 1 only)
        valid_mask_roc = (true_labels_with_boundary != 2) & (true_labels_with_boundary != -1)
        y_true_roc = true_labels_with_boundary[valid_mask_roc]
        y_scores_roc = scores[valid_mask_roc]

        # === ROC Curve + AUC
        fpr, tpr, roc_thresholds = roc_curve(y_true_roc, y_scores_roc)
        roc_auc = auc(fpr, tpr)
        
        try:
            roc_data_path = f"roc_data_snapshot_{i}.npz"
            np.savez(roc_data_path, fpr=fpr, tpr=tpr, thresholds=roc_thresholds)
            print(f"Saved ROC data to {roc_data_path}")
            print("roc_auc=",roc_auc)
        except Exception as e:
            pass
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Snapshot {i} - ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"roc_snapshot_{i}.png", dpi=300)
        plt.close()

        # === Threshold Sweep: accuracy, F1
        thresholds = np.linspace(0, 1, 20)
        accs, f1s = [], []
        results = []

        for t in thresholds:
            pred_binary = (scores >= t).astype(int)
            pred_labels_with_boundary = add_boundary_labels(pred_binary, positions)

            mask = (
                (true_labels_with_boundary != 2) &
                (true_labels_with_boundary != -1) &
                (pred_labels_with_boundary != 2) &
                (pred_labels_with_boundary != -1)
            )
            y_true = true_labels_with_boundary[mask]
            y_pred = pred_labels_with_boundary[mask]

            if len(y_true) == 0 or len(y_pred) == 0:
                continue  # skip empty evaluations

            assert np.all(np.isin(y_true, [0, 1]))
            assert np.all(np.isin(y_pred, [0, 1]))

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="binary")
            cm = confusion_matrix(y_true, y_pred)

            results.append({
                "threshold": t,
                "accuracy": acc,
                "f1_score": f1,
                "confusion_matrix": cm.tolist()
            })

            accs.append(acc)
            f1s.append(f1)
            print(f"Snapshot {i} | Threshold = {t:.2f} | Acc = {acc:.4f} | F1 = {f1:.4f}")

        # === Plot Threshold Sweep
        plt.figure(figsize=(6, 5))
        plt.plot(thresholds[:len(accs)], accs, label="Accuracy")
        plt.plot(thresholds[:len(f1s)], f1s, label="F1 Score")
        plt.axvline(x=0.5, linestyle="--", color="gray", label="t = 0.5")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title(f"Snapshot {i} - Threshold Sweep")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"threshold_sweep_snapshot_{i}.png", dpi=300)
        plt.close()

        # === Save to CSV
        df = pd.DataFrame(results)
        df.drop(columns=["confusion_matrix"]).to_csv(f"threshold_metrics_snapshot_{i}.csv", index=False)
        print(f"Saved: threshold_metrics_snapshot_{i}.csv")


def load_snapshots_from_dump(dump_path):
    snapshots = []
    with open(dump_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if "ITEM: TIMESTEP" in lines[i]:
            i += 9  # skip header lines
            atoms = []
            while i < len(lines) and not lines[i].startswith("ITEM:"):
                atoms.append(list(map(float, lines[i].split())))
                i += 1
            snapshots.append(np.array(atoms))
        else:
            i += 1
    return snapshots

# ------------------------------
# Weight Percent Calculation
# ------------------------------
def calculate_weight_percent(atom_data, region_labels):
    atomic_weights = {
        1: 55.845,  # Fe
        2: 24.305,  # Mg
        3: 28.085,  # Si
        4: 15.999,  # O
        5: 14.007   # N
    }

    atom_types = atom_data[:, 1].astype(int)

    rich_mask = region_labels == 1
    poor_mask = region_labels == 0

    def compute_wt_percent(mask):
        type_counts = {}
        total_mass = 0.0
        for t in np.unique(atom_types):
            count = np.sum((atom_types == t) & mask)
            mass = count * atomic_weights.get(t, 0)
            type_counts[t] = mass
            total_mass += mass
        wt_percent = {t: (m / total_mass * 100) for t, m in type_counts.items() if total_mass > 0}
        return wt_percent

    rich_wt = compute_wt_percent(rich_mask)
    poor_wt = compute_wt_percent(poor_mask)

    return rich_wt, poor_wt



# ------------------------------
# Weight Percent from Dump
# ------------------------------
def compute_weight_percent_from_dump(dump_snapshots):
    all_rich_wts = []
    all_poor_wts = []

    for i, atom_data in enumerate(dump_snapshots):
        pred_labels = compute_2class_labels(atom_data)
        boundary_labels = add_boundary_labels(pred_labels, atom_data[:, 2:5], snapshot_idx=i)
        rich_wt, poor_wt = calculate_weight_percent(atom_data, boundary_labels)
        all_rich_wts.append(rich_wt)
        all_poor_wts.append(poor_wt)

    def average_wt_percent(wt_list):
        total = {}
        for d in wt_list:
            for k, v in d.items():
                total[k] = total.get(k, 0) + v
        return {k: v / len(wt_list) for k, v in total.items()}

    avg_rich = average_wt_percent(all_rich_wts)
    avg_poor = average_wt_percent(all_poor_wts)

    return all_rich_wts, all_poor_wts, avg_rich, avg_poor

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_middle_z_slices(atom_positions, atom_labels, num_bins=(50, 50, 50), z_slice_count=5, snapshot_idx=None, save_dir="plots"):
    """
    Plot the middle Z slices in the XY-plane with color-coded labels.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create voxel grid
    box_min = atom_positions.min(0)
    box_max = atom_positions.max(0)
    x_bins = np.linspace(box_min[0], box_max[0], num_bins[0] + 1)
    y_bins = np.linspace(box_min[1], box_max[1], num_bins[1] + 1)
    z_bins = np.linspace(box_min[2], box_max[2], num_bins[2] + 1)

    x_idx = np.clip(np.digitize(atom_positions[:, 0], x_bins) - 1, 0, num_bins[0] - 1)
    y_idx = np.clip(np.digitize(atom_positions[:, 1], y_bins) - 1, 0, num_bins[1] - 1)
    z_idx = np.clip(np.digitize(atom_positions[:, 2], z_bins) - 1, 0, num_bins[2] - 1)

    z_mid = num_bins[2] // 2
    half = z_slice_count // 2
    target_z_indices = list(range(z_mid - half, z_mid + half + 1))

    cmap = mcolors.ListedColormap(["blue", "yellow", "red"])  # 0: Fe-poor, 1: Fe-rich, 2: boundary
    labels = ["Fe-poor", "Fe-rich", "Boundary"]
    fig, axs = plt.subplots(1, z_slice_count, figsize=(15, 3), dpi=150)

    for i, z in enumerate(target_z_indices):
        mask = z_idx == z
        xs = atom_positions[mask, 0]
        ys = atom_positions[mask, 1]
        lbls = atom_labels[mask]

        axs[i].scatter(xs, ys, c=lbls, cmap=cmap, s=10, alpha=0.7)
        axs[i].set_title(f"Z Slice {z}")
        axs[i].set_xlabel("X")
        axs[i].set_ylabel("Y")
        axs[i].set_xlim(box_min[0], box_max[0])
        axs[i].set_ylim(box_min[1], box_max[1])
        axs[i].set_aspect('equal')

    fig.suptitle(f"Middle {z_slice_count} Z Slices (XY Projection)" + (f" - Snapshot {snapshot_idx}" if snapshot_idx is not None else ""))
    plt.tight_layout()
    fname = f"snapshot_{snapshot_idx}_zslice_plot.png" if snapshot_idx is not None else "middle_zslice_plot.png"
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()


def compute_weight_percent_from_dump_dual_se(dump_snapshots, model_path="gnn_model.pt", label_dir="labels"):
    os.makedirs(label_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(GNN, in_channels=10, path=model_path).to(device)

    all_rich_kmeans, all_poor_kmeans = [], []
    all_rich_gnn, all_poor_gnn = [], []

    for i, atom_data in enumerate(dump_snapshots):
        positions = atom_data[:, 2:5]

        # ---- KMeans labels ----
        kmeans_label_path = os.path.join(label_dir, f"kmeans_labels_snapshot_{i}.npy")
        if os.path.exists(kmeans_label_path):
            pred_kmeans = np.load(kmeans_label_path)
        else:
            pred_kmeans = compute_2class_labels(atom_data)
            np.save(kmeans_label_path, pred_kmeans)

        boundary_kmeans = add_boundary_labels(pred_kmeans, positions, snapshot_idx=i)
        rich_wt_kmeans, poor_wt_kmeans = calculate_weight_percent(atom_data, boundary_kmeans)
        all_rich_kmeans.append(rich_wt_kmeans)
        all_poor_kmeans.append(poor_wt_kmeans)

        plot_middle_z_slices(positions, boundary_kmeans, snapshot_idx=i, save_dir="plots/kmeans")

        # ---- GNN labels ----
        pred_gnn, _, _ = infer(model, atom_data, device, threshold=0.03)
        boundary_gnn = add_boundary_labels(pred_gnn, positions, snapshot_idx=i)
        rich_wt_gnn, poor_wt_gnn = calculate_weight_percent(atom_data, boundary_gnn)
        all_rich_gnn.append(rich_wt_gnn)
        all_poor_gnn.append(poor_wt_gnn)

        plot_middle_z_slices(positions, boundary_gnn, snapshot_idx=i, save_dir="plots/gnn")

    def average_wt_percent_with_se(wt_list):
        from collections import defaultdict
        import numpy as np

        keys = set(k for d in wt_list for k in d.keys())
        means = {}
        ses = {}

        for k in keys:
            values = np.array([d.get(k, 0) for d in wt_list])
            mean = values.mean()
            se = values.std(ddof=1) / np.sqrt(len(values))
            means[k] = mean
            ses[k] = se

        return means, ses

    avg_kmeans_rich, se_kmeans_rich = average_wt_percent_with_se(all_rich_kmeans)
    avg_kmeans_poor, se_kmeans_poor = average_wt_percent_with_se(all_poor_kmeans)

    avg_gnn_rich, se_gnn_rich = average_wt_percent_with_se(all_rich_gnn)
    avg_gnn_poor, se_gnn_poor = average_wt_percent_with_se(all_poor_gnn)

    return {
        "kmeans": {
            "rich_all": all_rich_kmeans,
            "poor_all": all_poor_kmeans,
            "rich_avg": avg_kmeans_rich,
            "poor_avg": avg_kmeans_poor,
            "rich_se": se_kmeans_rich,
            "poor_se": se_kmeans_poor,
        },
        "gnn": {
            "rich_all": all_rich_gnn,
            "poor_all": all_poor_gnn,
            "rich_avg": avg_gnn_rich,
            "poor_avg": avg_gnn_poor,
            "rich_se": se_gnn_rich,
            "poor_se": se_gnn_poor,
        }
    }

