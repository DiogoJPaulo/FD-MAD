import os
import argparse
import itertools
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle as sk_shuffle
from sklearn.metrics import roc_curve, accuracy_score, auc


# ----------------------------------------------------------------------
#                     Utility metrics
# ----------------------------------------------------------------------
def eer_from_scores(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    diff = fnr - fpr
    i1 = np.where(diff <= 0)[0][0]
    i0 = i1 - 1
    w = diff[i0] / (diff[i0] - diff[i1] + 1e-12)
    eer = fpr[i0] + w * (fpr[i1] - fpr[i0])
    eer_thr = thr[i0] + w * (thr[i1] - thr[i0])
    return eer, eer_thr, (fpr, tpr, thr)


# ----------------------------------------------------------------------
#                     MRF exact inference
# ----------------------------------------------------------------------
def build_fully_connected_edges(num_nodes):
    return [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]

def build_face_topology_edges(regions):
    idx = {name: i for i, name in enumerate(regions)}
    required = ["nose", "mouth", "left_eye", "right_eye"]
    missing = [r for r in required if r not in idx]
    if missing:
        raise ValueError(f"Missing regions: {missing}. Got: {regions}")

    nose = idx["nose"]
    mouth = idx["mouth"]
    le = idx["left_eye"]
    re = idx["right_eye"]
    return [(le, re), (nose, le), (nose, re), (nose, mouth)]

def mrf_prob_bonafide(unary_log, beta, edges):
    """
    unary_log: [R,2], log P(z_r=0/1 | x_r)
    Pairwise: penalize disagreements with beta (Potts model).
      Energy: sum unary_log[r,zr]  - beta * sum_{(i,j)} |zi-zj|
    Returns posterior expected mean(z).
    """
    R = unary_log.shape[0]
    configs = list(itertools.product([0, 1], repeat=R))

    logps = []
    for z in configs:
        z = np.asarray(z, dtype=np.int64)
        s = 0.0
        # unary
        for r in range(R):
            s += unary_log[r, z[r]]
        # pairwise (penalize disagreements)
        for (i, j) in edges:
            s -= beta * abs(z[i] - z[j])
        logps.append(s)

    logps = np.asarray(logps, dtype=np.float64)
    logps -= logps.max()
    ps = np.exp(logps)
    ps /= ps.sum() + 1e-12

    mean_z = np.asarray([np.mean(cfg) for cfg in configs], dtype=np.float64)
    return float((ps * mean_z).sum())

def score_samples_mrf(unary_log_N, beta, edges):
    """
    unary_log_N: [N,R,2]
    returns scores: [N]
    """
    N = unary_log_N.shape[0]
    scores = np.zeros(N, dtype=np.float64)
    for i in range(N):
        scores[i] = mrf_prob_bonafide(unary_log_N[i], beta=beta, edges=edges)
    return scores


# ----------------------------------------------------------------------
#                     Loading features
# ----------------------------------------------------------------------
def load_region_npz(features_out, region, split_name):
    """
    Expected:
      features_out/local/<region>/<split_name>.npz
    with arrays X, y.
    """
    path = os.path.join(features_out, "local", region, f"{split_name}.npz")
    data = np.load(path)
    return data["X"], data["y"]

def load_all_regions(features_out, regions, test_names):
    """
    Returns:
      train: dict region -> (X_tr,y_tr)
      tests: dict test_name -> dict region -> (X_te,y_te)
    """
    train = {}
    tests = {t: {} for t in test_names}

    y_tr_ref = None
    y_te_ref = {t: None for t in test_names}

    for r in regions:
        X_tr, y_tr = load_region_npz(features_out, r, "train")
        train[r] = (X_tr, y_tr)

        if y_tr_ref is None:
            y_tr_ref = y_tr
        elif not np.array_equal(y_tr_ref, y_tr):
            print(f"[WARN] y_train mismatch for region={r} (ordering differs).")

        for t in test_names:
            X_te, y_te = load_region_npz(features_out, r, f"test_{t}")
            tests[t][r] = (X_te, y_te)

            if y_te_ref[t] is None:
                y_te_ref[t] = y_te
            elif not np.array_equal(y_te_ref[t], y_te):
                print(f"[WARN] y_test mismatch for test={t}, region={r} (ordering differs).")

    return train, tests


# ----------------------------------------------------------------------
#                     Unary training + potentials
# ----------------------------------------------------------------------
def train_unary_models(regions, train_dict, pca_dim=32, seed=42):
    models = {}
    for r in regions:
        X_tr, y_tr = train_dict[r]
        X_tr, y_tr = sk_shuffle(X_tr, y_tr, random_state=seed)

        pipe = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=pca_dim),
            LogisticRegression(max_iter=5000),
        )
        pipe.fit(X_tr, y_tr)
        models[r] = pipe
        print(f"[INFO] trained unary: {r} | X_tr={X_tr.shape}")
    return models

def compute_unary_log(regions, models, X_dict):
    """
    X_dict: region -> X (N,D)
    Returns unary_log: [N,R,2]
    """
    R = len(regions)
    N = X_dict[regions[0]].shape[0]
    out = np.zeros((N, R, 2), dtype=np.float64)

    for r_idx, r in enumerate(regions):
        X = X_dict[r]
        if X.shape[0] != N:
            raise ValueError(f"N mismatch region {r}: {X.shape[0]} vs {N}")

        probs = models[r].predict_proba(X)  # columns correspond to classes_
        classes = models[r].named_steps["logisticregression"].classes_
        order = np.argsort(classes)  # ensures column 0 is class 0, column 1 is class 1
        probs = probs[:, order]

        out[:, r_idx, 0] = np.log(probs[:, 0] + 1e-8)
        out[:, r_idx, 1] = np.log(probs[:, 1] + 1e-8)

    return out


# ----------------------------------------------------------------------
#                     Beta tuning (optional)
# ----------------------------------------------------------------------
def tune_beta(beta_grid, unary_log_train, y_train, edges):
    best_beta, best_auc = beta_grid[0], -1e9
    for b in beta_grid:
        scores = score_samples_mrf(unary_log_train, beta=b, edges=edges)
        fpr, tpr, _ = roc_curve(y_train, scores, pos_label=1)
        a = auc(fpr, tpr)
        if a > best_auc:
            best_auc, best_beta = a, b
    return best_beta, best_auc


# ----------------------------------------------------------------------
#                     Evaluation
# ----------------------------------------------------------------------
def evaluate_scores(y, scores, name=""):
    fpr, tpr, thr = roc_curve(y, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    eer, eer_thr, _ = eer_from_scores(y, scores)
    y_pred = (scores >= eer_thr).astype(np.int32)
    acc = accuracy_score(y, y_pred)

    print(f"=== {name} ===")
    print(f"AUC: {roc_auc:.4f}")
    print(f"EER: {eer*100:.2f}% @ thr={eer_thr:.6f}")
    print(f"ACC: {acc*100:.2f}%\n")
    return {"auc": roc_auc, "eer": eer, "eer_thr": eer_thr, "acc": acc}


# ----------------------------------------------------------------------
#                     Save fusion model (optional)
# ----------------------------------------------------------------------
def save_mrf_model(save_dir, regions, models, edges, beta):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "regions.npy"), np.array(regions, dtype=object))
    np.save(os.path.join(save_dir, "edges.npy"), np.array(edges, dtype=object))
    np.save(os.path.join(save_dir, "beta.npy"), np.array(float(beta)))
    np.save(os.path.join(save_dir, "region_models.npy"), models, allow_pickle=True)
    print(f"[INFO] saved MRF package to: {save_dir}")


# ----------------------------------------------------------------------
#                     Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("Local-region MRF fusion evaluation (residual features)")
    ap.add_argument("--features-out", required=True, help="Root feature folder (contains local/<region>/train.npz etc.)")
    ap.add_argument("--regions", nargs="+", required=True, help="Regions to use e.g. mouth nose left_eye right_eye")
    ap.add_argument("--tests", nargs="+", required=True, help="Test names e.g. opencv facemorpher webmorpher mipgan_1 ...")

    ap.add_argument("--graph", choices=["fc", "face"], default="fc", help="MRF edges: fully-connected or face-topology")
    ap.add_argument("--beta", type=float, default=0.9, help="If set, use this beta (no tuning).")
    ap.add_argument("--tune-beta", action="store_true", help="Tune beta on TRAIN AUC (slow).")
    ap.add_argument("--beta-min", type=float, default=0.0)
    ap.add_argument("--beta-max", type=float, default=6.55)
    ap.add_argument("--beta-num", type=int, default=50)

    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--save-dir", default=None, help="If set, save unary models + edges + beta here")

    args = ap.parse_args()

    np.random.seed(args.seed)

    # Load features
    train_dict, tests_dict = load_all_regions(args.features_out, args.regions, args.tests)

    # Train unary per region
    models = train_unary_models(args.regions, train_dict, pca_dim=args.pca_dim, seed=args.seed)

    # Build edges
    if args.graph == "fc":
        edges = build_fully_connected_edges(len(args.regions))
    else:
        edges = build_face_topology_edges(args.regions)

    # Unary logs for train
    X_train_dict = {r: train_dict[r][0] for r in args.regions}
    y_train = train_dict[args.regions[0]][1]
    unary_log_train = compute_unary_log(args.regions, models, X_train_dict)

    # Choose beta
    if args.beta is not None:
        beta = float(args.beta)
        print(f"[INFO] Using fixed beta={beta}")
    elif args.tune_beta:
        grid = np.linspace(args.beta_min, args.beta_max, args.beta_num)
        beta, best_auc = tune_beta(grid, unary_log_train, y_train, edges)
        print(f"[INFO] Tuned beta={beta:.4f} (train AUC={best_auc:.4f})")
    else:
        beta = float(args.beta_max)
        print(f"[INFO] Using default beta={beta} (set --beta or --tune-beta to change)")

    # Optional save
    if args.save_dir:
        save_mrf_model(args.save_dir, args.regions, models, edges, beta)

    # Evaluate per test
    eers = []
    for t in args.tests:
        X_te_dict = {r: tests_dict[t][r][0] for r in args.regions}
        y_te = tests_dict[t][args.regions[0]][1]

        unary_log_te = compute_unary_log(args.regions, models, X_te_dict)
        scores = score_samples_mrf(unary_log_te, beta=beta, edges=edges)

        m = evaluate_scores(y_te, scores, name=f"MRF local fusion | test={t} | graph={args.graph} | beta={beta:.4f}")
        eers.append(m["eer"])

    mean_eer = float(np.mean(eers)) if len(eers) else float("nan")
    print(f"Mean EER over tests: {mean_eer*100:.3f}%")

if __name__ == "__main__":
    main()

""" Example usage:
python fd_mad_mrf.py  \
  --features-out features_residual \
  --regions mouth nose left_eye right_eye \
  --tests opencv facemorpher webmorpher mipgan_1 mipgan_2 mordiff \
  --graph fc \
  --beta 0.9 \
  --save-dir saved_mrf_local
"""