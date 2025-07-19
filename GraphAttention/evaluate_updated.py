import csv
from run import (
    run_evaluation_on_new_data,
    load_snapshots_from_dump,
    run_evaluation_on_new_data_v2,
    compute_weight_percent_from_dump_dual_se
)

print("Evaluation Started")

# --- Step 1: Load every 100th snapshot ---
snapshots = load_snapshots_from_dump("out.dump")
#snapshots = snapshots[200::40]  # subsample
#snapshots = [snapshots[-1]]
print("length-",len(snapshots))
#run_evaluation_on_new_data('gnn_model.pt',snapshots)

run_evaluation_on_new_data_v2('gnn_model.pt',snapshots)

# --- Step 2: Run KMeans + GNN labeling and weight percent calculation ---
results = compute_weight_percent_from_dump_dual_se(snapshots, model_path="gnn_model.pt")


# --- Step 3: Save per-snapshot and average weight % to CSV ---
def save_weight_percent_csv(weight_result, label_prefix):
    all_rich = weight_result["rich_all"]
    all_poor = weight_result["poor_all"]
    avg_rich = weight_result["rich_avg"]
    avg_poor = weight_result["poor_avg"]

    with open(f"{label_prefix}_rich_region.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Snapshot", "Fe", "Mg", "Si", "O", "N"])
        for i, wt in enumerate(all_rich):
            writer.writerow([i] + [wt.get(t, 0.0) for t in [1, 2, 3, 4, 5]])

    with open(f"{label_prefix}_poor_region.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Snapshot", "Fe", "Mg", "Si", "O", "N"])
        for i, wt in enumerate(all_poor):
            writer.writerow([i] + [wt.get(t, 0.0) for t in [1, 2, 3, 4, 5]])

    with open(f"{label_prefix}_avg_region.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Region", "Fe", "Mg", "Si", "O", "N"])
        writer.writerow(["Fe-rich"] + [avg_rich.get(t, 0.0) for t in [1, 2, 3, 4, 5]])
        writer.writerow(["Fe-poor"] + [avg_poor.get(t, 0.0) for t in [1, 2, 3, 4, 5]])


# --- Step 4: Export both methods to CSV ---
save_weight_percent_csv(results["kmeans"], "kmeans")
save_weight_percent_csv(results["gnn"], "gnn")

print("Weight percent exported to CSV for both KMeans and GNN.")

