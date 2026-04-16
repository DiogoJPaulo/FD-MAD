import os
import argparse
import itertools
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, auc


# ============================================================
#                 Metrics
# ============================================================
def eer_from_scores(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    diff = fnr - fpr

    # first index where fnr <= fpr
    i1 = np.where(diff <= 0)[0][0]
    i0 = i1 - 1
    w = diff[i0] / (diff[i0] - diff[i1] + 1e-12)

    eer = fpr[i0] + w * (fpr[i1] - fpr[i0])
    eer_thr = thr[i0] + w * (thr[i1] - thr[i0])
    return eer, eer_thr, (fpr, tpr, thr)


def bpcer_at_apcer_op(fpr, fnmr, thr, apcer_op):
    order = np.argsort(fpr)
    fpr, fnmr, thr = fpr[order], fnmr[order], thr[order]

    i1 = np.searchsorted(fpr, apcer_op, side="right")
    if i1 == 0:
        return fnmr[0], thr[0]
    if i1 == len(fpr):
        return fnmr[-1], thr[-1]

    i0 = i1 - 1
    w = (apcer_op - fpr[i0]) / (fpr[i1] - fpr[i0] + 1e-12)
    bpcer = fnmr[i0] + w * (fnmr[i1] - fnmr[i0])
    thr_op = thr[i0] + w * (thr[i1] - thr[i0])
    return bpcer, thr_op


def evaluate_scores(y, scores, name=""):
    fpr, tpr, thr = roc_curve(y, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    eer, eer_thr, _ = eer_from_scores(y, scores)

    y_pred = (scores >= eer_thr).astype(np.int32)
    acc = accuracy_score(y, y_pred)

    bpcer1, thr1 = bpcer_at_apcer_op(fpr, 1 - tpr, thr, 0.01)
    bpcer20, thr20 = bpcer_at_apcer_op(fpr, 1 - tpr, thr, 0.20)

    print(f"=== {name} ===")
    print(f"AUC:  {roc_auc:.4f}")
    print(f"EER:  {eer * 100:.2f}% @ thr={eer_thr:.6f}")
    print(f"BPCER @ APCER=1%:  {bpcer1 * 100:.2f}%  @ thr={thr1:.20f}")
    print(f"BPCER @ APCER=20%: {bpcer20 * 100:.2f}%  @ thr={thr20:.20f}")
    print(f"ACC:  {acc * 100:.2f}%\n")

    return {"auc": roc_auc, "eer": eer, "eer_thr": eer_thr, "acc": acc, "bpcer1": bpcer1, "bpcer20": bpcer20}


# ============================================================
#                 MRF fusion core
# ============================================================
def mrf_sample_prob_bonafide(unary_log_potentials, beta, edges):
    """
    unary_log_potentials: [R,2] where:
      [r,0] = log P(z_r=0 | x_r)  (morph)
      [r,1] = log P(z_r=1 | x_r)  (bonafide)
    Pairwise term: reward agreement by +beta (same-label).
    """
    R = unary_log_potentials.shape[0]
    configs = list(itertools.product([0, 1], repeat=R))
    logps = []

    for cfg in configs:
        z = np.asarray(cfg, dtype=np.int64)
        logp = 0.0
        for r in range(R):
            logp += unary_log_potentials[r, z[r]]
        for (i, j) in edges:
            if z[i] == z[j]:
                logp += beta
        logps.append(logp)

    logps = np.asarray(logps, dtype=np.float64)
    logps -= logps.max()
    ps = np.exp(logps)
    ps /= ps.sum() + 1e-12

    mean_z = np.asarray([np.mean(cfg) for cfg in configs], dtype=np.float64)
    return float((ps * mean_z).sum())


def compute_unary_log_potentials(regions, region_models, X_dict):
    """
    regions: list[str]
    region_models: dict region -> sklearn pipeline (has LogisticRegression as last step)
    X_dict: dict region -> X (N,D)
    returns unary_log: [N,R,2]
    """
    R = len(regions)
    N = X_dict[regions[0]].shape[0]
    unary_log = np.zeros((N, R, 2), dtype=np.float64)

    for r_idx, region in enumerate(regions):
        model = region_models[region]
        X = X_dict[region]
        if X.shape[0] != N:
            raise ValueError(f"N mismatch for region {region}: {X.shape[0]} vs {N}")

        probs = model.predict_proba(X)  # [N,2]
        # robust class order
        if "logisticregression" in model.named_steps:
            classes = model.named_steps["logisticregression"].classes_
        else:
            classes = model.classes_
        order = np.argsort(classes)
        probs = probs[:, order]

        unary_log[:, r_idx, 0] = np.log(probs[:, 0] + 1e-8)
        unary_log[:, r_idx, 1] = np.log(probs[:, 1] + 1e-8)

    return unary_log


# ============================================================
#                 Loading helpers
# ============================================================
def load_npz(path):
    data = np.load(path)
    return data["X"], data["y"]


def global_test_npz(features_out, test_name):
    # expected: features_out/global/test_<name>.npz
    return os.path.join(features_out, "global", f"test_{test_name}.npz")


def local_test_npz(features_out, region, test_name):
    # expected: features_out/local/<region>/test_<name>.npz
    return os.path.join(features_out, "local", region, f"test_{test_name}.npz")


# ============================================================
#                 Model loading helpers
# ============================================================
def load_global_model(global_model_dir):
    """
    Robust loader for global model artifacts.

    Supports:
      A) preprocess_pipe.npy + svm.npy   (new names)
      B) global_preprocess_pipe.npy + global_svm.npy (old names that i had)
    And pipe formats:
      1) sklearn Pipeline saved as object -> np.load(...).item()
      2) numpy object array of length 1 containing pipeline
      3) numpy object array of length 2: [scaler, pca]
    """
    pipe_path_1 = os.path.join(global_model_dir, "preprocess_pipe.npy")
    svm_path_1 = os.path.join(global_model_dir, "svm.npy")

    pipe_path_2 = os.path.join(global_model_dir, "global_preprocess_pipe.npy")
    svm_path_2 = os.path.join(global_model_dir, "global_svm.npy")

    if os.path.exists(pipe_path_1) and os.path.exists(svm_path_1):
        pipe_raw = np.load(pipe_path_1, allow_pickle=True)
        clf = np.load(svm_path_1, allow_pickle=True).item()
        pipe = _coerce_pipe(pipe_raw)
        return pipe, clf

    if os.path.exists(pipe_path_2) and os.path.exists(svm_path_2):
        pipe_raw = np.load(pipe_path_2, allow_pickle=True)
        clf = np.load(svm_path_2, allow_pickle=True).item()
        pipe = _coerce_pipe(pipe_raw)
        return pipe, clf

    raise FileNotFoundError(f"Could not find global model files in: {global_model_dir}")


def _coerce_pipe(pipe_raw):
    """
    Convert loaded numpy object into something that has .transform(X).
    """
    # Case: already a sklearn Pipeline-like object
    if not isinstance(pipe_raw, np.ndarray):
        return pipe_raw

    # Case: ndarray of objects
    if pipe_raw.shape == ():  # 0-d array
        return pipe_raw.item()

    if pipe_raw.size == 1:
        return pipe_raw.reshape(-1)[0]

    # Common older case: [scaler, pca]
    if pipe_raw.size == 2:
        scaler = pipe_raw.reshape(-1)[0]
        pca = pipe_raw.reshape(-1)[1]

        class _SimplePipe:
            def __init__(self, scaler, pca):
                self.scaler = scaler
                self.pca = pca

            def transform(self, X):
                return self.pca.transform(self.scaler.transform(X))

        return _SimplePipe(scaler, pca)

    raise ValueError(
        f"Unrecognized preprocess pipe format: type={type(pipe_raw)}, shape={pipe_raw.shape}, size={pipe_raw.size}")


def load_local_mrf_model(local_model_dir):
    regions = np.load(os.path.join(local_model_dir, "regions.npy"), allow_pickle=True).tolist()
    region_models = np.load(os.path.join(local_model_dir, "region_models.npy"), allow_pickle=True).item()
    edges = np.load(os.path.join(local_model_dir, "edges.npy"), allow_pickle=True).tolist()
    beta = float(np.load(os.path.join(local_model_dir, "beta.npy"), allow_pickle=True))
    return regions, region_models, edges, beta


# ============================================================
#                 Main
# ============================================================
def main():
    ap = argparse.ArgumentParser("Fuse GLOBAL + LOCAL(MRF) scores")
    ap.add_argument("--features-out", required=True,
                    help="Root folder with global/ and local/<region>/ feature npz files")
    ap.add_argument("--tests", nargs="+", required=True,
                    help="Test names, e.g. opencv facemorpher webmorpher mipgan_1 mipgan_2 mordiff")

    ap.add_argument("--global-model-dir", required=True)
    ap.add_argument("--local-model-dir", required=True)

    ap.add_argument("--alpha", type=float, default=0.6,
                    help="Fusion: s_fused = alpha*s_global + (1-alpha)*s_local")
    ap.add_argument("--no-local", action="store_true", help="Only evaluate global model")
    ap.add_argument("--no-global", action="store_true", help="Only evaluate local MRF")

    args = ap.parse_args()

    # Load models
    pipe_g, clf_g = load_global_model(args.global_model_dir)
    regions, region_models, edges, beta = load_local_mrf_model(args.local_model_dir)

    print("[INFO] Loaded GLOBAL model from:", args.global_model_dir)
    print("[INFO] Loaded LOCAL MRF model from:", args.local_model_dir)
    print("[INFO] Regions:", regions)
    print("[INFO] beta:", beta)
    print("[INFO] edges:", edges)
    print("[INFO] alpha:", args.alpha, "\n")

    for t in args.tests:
        print("\n##############################")
        print("Test:", t)
        print("##############################")

        # -------- global scores --------
        s_global, y_global = None, None
        if not args.no_global:
            Xg, yg = load_npz(global_test_npz(args.features_out, t))
            y_global = yg

            probs = clf_g.predict_proba(pipe_g.transform(Xg))
            classes = clf_g.classes_.tolist()
            idx_bona = classes.index(1)
            s_global = probs[:, idx_bona].astype(np.float64)

            print(f"[INFO] Global: X={Xg.shape}, y={yg.shape}")

        # -------- local MRF scores --------
        s_local, y_local = None, None
        if not args.no_local:
            X_local_dict = {}
            y_local_ref = None
            for r in regions:
                Xr, yr = load_npz(local_test_npz(args.features_out, r, t))
                X_local_dict[r] = Xr
                if y_local_ref is None:
                    y_local_ref = yr
                elif not np.array_equal(y_local_ref, yr):
                    print(f"[WARN] Label mismatch across regions for test={t}. Using first region labels.")
            y_local = y_local_ref

            unary_log = compute_unary_log_potentials(regions, region_models, X_local_dict)
            N = unary_log.shape[0]
            s_local = np.zeros(N, dtype=np.float64)
            for i in range(N):
                s_local[i] = mrf_sample_prob_bonafide(unary_log[i], beta=beta, edges=edges)

            print(f"[INFO] Local: unary_log={unary_log.shape}")

        # choose y for eval
        y = y_global if y_global is not None else y_local
        if y_global is not None and y_local is not None and not np.array_equal(y_global, y_local):
            print("[WARN] Global vs Local labels mismatch. Using LOCAL labels for evaluation.")
            y = y_local

        # -------- evaluations --------
        if s_global is not None:
            evaluate_scores(y, s_global, name=f"{t} - GLOBAL only")

        if s_local is not None:
            evaluate_scores(y, s_local, name=f"{t} - LOCAL (MRF) only")

        if (s_global is not None) and (s_local is not None) and (not args.no_global) and (not args.no_local):
            s_fused = args.alpha * s_global + (1.0 - args.alpha) * s_local
            evaluate_scores(y, s_fused, name=f"{t} - FUSED (alpha={args.alpha:.2f})")


if __name__ == "__main__":
    main()

""" EXAMPLE USAGE:
python fd_mad_fuse_scores.py \
  --features-out features_residual \
  --tests opencv facemorpher webmorpher mipgan_1 mipgan_2 mordiff \
  --global-model-dir models/global_residual \
  --local-model-dir saved_mrf_local \
  --alpha 0.6
"""
