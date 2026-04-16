import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

import torch
import torch.fft as fft

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils import shuffle as sk_shuffle
from sklearn.metrics import roc_curve, auc


# ============================================================
#                 PyTorch/CUDA Fourier Extractor
# ============================================================
class FourierBandsTorch:
    def __init__(self, n_bands=500, log_scale=True, patch_size=None, device=None, include_phase=False,
                 dtype=torch.float32):
        self.n_bands = n_bands
        self.include_phase = include_phase
        self.log_scale = log_scale
        self.patch_size = patch_size  # if 10 => 10x10 grid
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype
        # caches to avoid re-building radial grids
        self._r_cache = {}  # key: (H,W,device) -> r tensor
        self._bins_cache = {}  # key: (H,W,device) -> bins tensor

    def _get_r_and_bins(self, H, W, device):
        key_r = (H, W, device, self.dtype)  # r depends on dtype as well
        if key_r not in self._r_cache:
            y = torch.arange(H, device=device, dtype=self.dtype)
            x = torch.arange(W, device=device, dtype=self.dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            cy, cx = H // 2, W // 2
            r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            self._r_cache[key_r] = r
        r = self._r_cache[key_r]

        # cap bands to the number of 1-px rings
        max_r = r.max()
        K = min(self.n_bands, int(torch.floor(max_r).item()) + 1)
        key_bins = (H, W, device, self.dtype, K)
        if key_bins not in self._bins_cache:
            bins = torch.linspace(0, max_r, K + 1, device=device, dtype=self.dtype)
            self._bins_cache[key_bins] = bins
        bins = self._bins_cache[key_bins]
        K = bins.numel() - 1  # derive K from cached bins
        return r, bins, K

    @staticmethod
    def _scatter_mean(vals, idx, K):
        sums = torch.zeros(K, device=vals.device, dtype=vals.dtype)
        cnts = torch.zeros(K, device=vals.device, dtype=vals.dtype)
        sums.scatter_add_(0, idx, vals)
        cnts.scatter_add_(0, idx, torch.ones_like(vals))
        return sums / (cnts + 1e-8)

    def _fft_band_energy_single(self, img2d_t):
        H, W = img2d_t.shape
        F = fft.fftshift(fft.fft2(img2d_t))
        mag = torch.abs(F)
        phase = torch.angle(F)

        if self.log_scale:
            mag = torch.log1p(mag)
        r, bins, K = self._get_r_and_bins(H, W, img2d_t.device)
        band_idx = torch.bucketize(r.reshape(-1), bins[1:])  # 0..K-1
        bands_mag = self._scatter_mean(mag.reshape(-1), band_idx, K)

        if self.include_phase:
            sin_phase = torch.sin(phase)
            cos_phase = torch.cos(phase)
            bands_sin = self._scatter_mean(sin_phase.reshape(-1), band_idx, K)
            bands_cos = self._scatter_mean(cos_phase.reshape(-1), band_idx, K)
            bands = torch.cat([bands_mag, bands_sin, bands_cos], dim=0)
        else:
            bands = bands_mag

        return bands, mag

    def _fft_band_energy_single_residuals(self, img2d_t):
        """
        Compute ring-mean log-power spectrum and subtract a fitted 1/f baseline.
        Returns (bands_residuals, mag)
        """
        H, W = img2d_t.shape
        F = fft.fftshift(fft.fft2(img2d_t))
        mag = torch.abs(F)

        # work in log space
        logP = torch.log1p(mag) if self.log_scale else torch.log(mag + 1e-12)

        # radial averaging
        r, bins, K = self._get_r_and_bins(H, W, img2d_t.device)
        band_idx = torch.bucketize(r.reshape(-1), bins[1:])  # 0..K-1
        bands_logP = self._scatter_mean(logP.reshape(-1), band_idx, K)

        # --- compute residuals (baseline-corrected log power) ---
        # convert to numpy for simplicity
        bands_np = bands_logP.detach().cpu().numpy()
        # plot fourier image with the rings
        # plt.figure(figsize=(5, 5))
        # plt.imshow(logP.detach().cpu().numpy(), cmap='magma')
        # plt.contour(r.detach().cpu().numpy(), bins.detach().cpu().numpy(), colors='blue', linewidths=0.5)
        # plt.title("Log-Fourier Magnitude")
        # plt.colorbar()
        # plt.show()
        # plt.close()

        f = 0.5 * (bins[1:] + bins[:-1]).detach().cpu().numpy()  # f is the mid-point frequencies of each band
        # f = bins[1:].detach().cpu().numpy()  # alternative: use band edges
        # print(f)  # DEBUG
        # sys.exit(0)  # DEBUG
        # I could have used f = bins[1:].detach().cpu().numpy() directly, but mid-point is more accurate
        # while bands_np is the log-power at each band, f is the frequency, meaning we expect logP ~ a + b log f
        mask = (f > 0.5) & np.isfinite(bands_np)
        if mask.sum() >= 3:  # need at least 3 points to fit
            # fit logP = a + b log f
            A = np.vstack([np.ones_like(f[mask]), np.log(f[mask])]).T
            a, b = np.linalg.lstsq(A, bands_np[mask], rcond=None)[0]
            baseline = a + b * np.log(np.maximum(f, 1e-6))
            residuals = bands_np - baseline
            # plt.plot(f, bands_np, linestyle='-', color='blue', label='Radial log-power')
            # plt.plot(f, baseline, linestyle='--', color='red')
            # plt.plot(f, residuals, linestyle='-', color='green')
            # plt.title("Radial log-power spectrum after baseline removal")
            # plt.show()
            # plt.close()
            # sys.exit()
        else:
            residuals = bands_np * 0

        bands_residuals = torch.from_numpy(residuals).to(img2d_t.device, dtype=img2d_t.dtype)
        return bands_residuals, mag

    @torch.inference_mode()
    def transform_batch(self, imgs_uint8_list):
        B = len(imgs_uint8_list)
        H, W = imgs_uint8_list[0].shape
        t = torch.empty((B, H, W), device=self.device, dtype=self.dtype)
        for i, im in enumerate(imgs_uint8_list):
            t[i] = torch.from_numpy(im).to(self.device, dtype=self.dtype) / 255.0

        if self.patch_size is None:
            out = []
            for i in range(B):
                b, _ = self._fft_band_energy_single_residuals(t[i])
                # b = self._pad_or_trunc(b, self.n_bands)
                H, W = t[i].shape
                K = self.effective_K(H, W)  # or (ph, pw) in the patch case
                b = b[..., :K]  # truncate only; no padding
                out.append(b)
            return torch.stack(out, 0).detach().cpu().numpy().astype("float32")

        ps = int(self.patch_size)
        ph, pw = H // ps, W // ps
        t = t[:, :ps * ph, :ps * pw]
        t_blocks = t.view(B, ps, ph, ps, pw).permute(0, 1, 3, 2, 4).reshape(B, ps * ps, ph, pw)

        F = fft.fftshift(fft.fft2(t_blocks), dim=(-2, -1))
        mag = torch.abs(F)
        if self.log_scale:
            mag = torch.log1p(mag)

        r, bins, K = self._get_r_and_bins(ph, pw, t.device)
        idx = torch.bucketize(r.reshape(-1), bins[1:])  # [ph*pw]
        idx = idx.view(1, 1, -1).expand(B, ps * ps, -1)  # [B,P,ph*pw]

        mags_flat = mag.reshape(B, ps * ps, -1)
        sums = torch.zeros(B, ps * ps, K, device=mag.device, dtype=mag.dtype)
        cnts = torch.zeros_like(sums)
        sums.scatter_add_(2, idx, mags_flat)
        cnts.scatter_add_(2, idx, torch.ones_like(mags_flat))
        bands = (sums / (cnts + 1e-8))
        K = self.effective_K(H, W)  # or (ph, pw) in the patch case
        # bands = self._pad_or_trunc(bands, self.n_bands)
        bands = bands[..., :K]  # truncate only; no padding
        bands = bands.reshape(B, -1)  # [B, P*self.n_bands]
        return bands.detach().cpu().numpy().astype("float32")

    def _pad_or_trunc(self, bands, target):
        # bands: [..., K]
        K = bands.shape[-1]
        if K == target:
            return bands
        if K > target:
            return bands[..., :target]
        pad = torch.zeros(*bands.shape[:-1], target - K, device=bands.device, dtype=bands.dtype)
        return torch.cat([bands, pad], dim=-1)

    def effective_K(self, H, W):
        with torch.no_grad():
            _, _, K = self._get_r_and_bins(H, W, self.device)
        return K


# ============================================================
# Helpers
# ============================================================

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def balance_df(df):
    idx_bf = np.where(df["label"] == "bonafide")[0]
    idx_m = np.where(df["label"] != "bonafide")[0]
    n = min(len(idx_bf), len(idx_m))
    idx_bf = np.random.choice(idx_bf, n, replace=False)
    idx_m = np.random.choice(idx_m, n, replace=False)
    return df.iloc[np.concatenate([idx_bf, idx_m])].sample(frac=1).reset_index(drop=True)


def load_resize_bgr(path, size):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.resize(bgr, size)


def rewrite_to_region(df, region):
    """
      subj/file -> subj/<region>/file
    If region is None => unchanged (global).
    """
    if region is None:
        return df

    def rw(x):
        parts = x.split("/")
        if len(parts) < 2:
            raise ValueError(f"Unexpected image_path format: {x}")
        return parts[0] + f"/{region}/" + parts[1]

    out = df.copy()
    out["image_path"] = out["image_path"].apply(rw)
    return out


def extract_features_batched(df, images_root, FB, batch_size=256, resize_hw=(500, 500)):
    N = len(df)
    rows = list(df.itertuples(index=False))

    X_list, y_list = [], []
    pbar = tqdm(total=N, desc="Extracting (CUDA)" if str(FB.device).startswith("cuda") else "Extracting (CPU)")

    i = 0
    while i < N:
        j = min(i + batch_size, N)
        paths = [os.path.join(images_root, getattr(r, "image_path")) for r in rows[i:j]]
        labels = [getattr(r, "label") for r in rows[i:j]]

        Rs, Gs, Bs = [], [], []
        for p in paths:
            bgr = load_resize_bgr(p, resize_hw)
            b, g, r = cv2.split(bgr)
            Rs.append(r.astype(np.uint8, copy=False))
            Gs.append(g.astype(np.uint8, copy=False))
            Bs.append(b.astype(np.uint8, copy=False))

        FR = FB.transform_batch(Rs)
        FG = FB.transform_batch(Gs)
        FBb = FB.transform_batch(Bs)

        feats = np.concatenate([FR, FG, FBb], axis=1).astype(np.float32)
        y = np.array([1 if lab == "bonafide" else 0 for lab in labels], dtype=np.int32)

        X_list.append(feats)
        y_list.append(y)

        pbar.update(j - i)
        i = j

    pbar.close()
    return np.concatenate(X_list, 0), np.concatenate(y_list, 0)


def out_dir(base_out, scope, region=None):
    # scope: global | local
    if scope == "global":
        return os.path.join(base_out, "global")
    if scope == "local":
        if not region:
            raise ValueError("local scope requires --region")
        return os.path.join(base_out, "local", region)
    raise ValueError(scope)


def save_npz(base_out, scope, region, name, X, y):
    d = out_dir(base_out, scope, region)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"{name}.npz"), X=X, y=y)


def load_npz(base_out, scope, region, name):
    d = out_dir(base_out, scope, region)
    data = np.load(os.path.join(d, f"{name}.npz"))
    return data["X"], data["y"]


# ============================================================
# Train + Eval (GLOBAL ONLY)
# ============================================================
def eer_from_scores(y_true, scores):  # This is the correct EER calculation
    # Single ROC using positive-class scores
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    # Find where FNR crosses FPR and interpolate
    diff = fnr - fpr
    # indices around the zero crossing
    i1 = np.where(diff <= 0)[0][0]  # first index where fnr <= fpr
    i0 = i1 - 1
    # Linear interpolation parameter
    w = diff[i0] / (diff[i0] - diff[i1])  # in [0,1]
    eer = fpr[i0] + w * (fpr[i1] - fpr[i0])
    eer_thr = thr[i0] + w * (thr[i1] - thr[i0])

    return eer, eer_thr, (fpr, tpr, thr)


def bpcer_at_apcer_op(fpr, fnmr, thr, apcer_op):
    """
    Return (bpcer_at_op, threshold_at_op) for the lowest BPCER such that APCER <= apcer_op.
    Interpolates between adjacent ROC points.
    """
    # Ensure FPR nondecreasing
    order = np.argsort(fpr)
    fpr, fnmr, thr = fpr[order], fnmr[order], thr[order]

    # Index just to the right of the target (first fpr > op), choose segment [i0, i1]
    i1 = np.searchsorted(fpr, apcer_op, side='right')
    if i1 == 0:
        return fnmr[0], thr[0]  # operating point left of curve
    if i1 == len(fpr):
        return fnmr[-1], thr[-1]  # operating point right of curve

    i0 = i1 - 1
    # Linear interpolate on FPR to hit apcer_op
    w = (apcer_op - fpr[i0]) / (fpr[i1] - fpr[i0])
    bpcer = fnmr[i0] + w * (fnmr[i1] - fnmr[i0])
    thr_op = thr[i0] + w * (thr[i1] - thr[i0])
    return bpcer, thr_op


def train_and_eval_global(features_out, model_out, test_names, pca_dim=2, C=1.0, kernel="rbf", seed=42):
    seed_everything(seed)

    # Load GLOBAL residual features
    X_tr, y_tr = load_npz(features_out, "global", None, "train")
    X_tr, y_tr = sk_shuffle(X_tr, y_tr, random_state=seed)

    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        PCA(n_components=pca_dim),
    )
    F_tr = pipe.fit_transform(X_tr)

    clf = SVC(C=C, kernel=kernel, probability=True)
    clf.fit(F_tr, y_tr)

    # Save model (optional but useful)
    os.makedirs(model_out, exist_ok=True)
    np.save(os.path.join(model_out, "preprocess_pipe.npy"), pipe, allow_pickle=True)
    np.save(os.path.join(model_out, "svm.npy"), clf, allow_pickle=True)

    # Evaluate on GLOBAL residual features
    print("\n=== EVAL on GLOBAL residual features ===")
    for name in test_names:
        X_te, y_te = load_npz(features_out, "global", None, f"test_{name}")
        F_te = pipe.transform(X_te)

        classes = clf.classes_.tolist()
        probs_bonafide = clf.predict_proba(F_te)[:, classes.index(1)]

        fpr, tpr, thr = roc_curve(y_te, probs_bonafide, pos_label=1)
        eer, _, _ = eer_from_scores(y_te, probs_bonafide)
        # If 0 is morph and 1 is bonafide, then:
        apcer1, _ = bpcer_at_apcer_op(fpr, 1 - tpr, thr, 0.01)
        apcer20, _ = bpcer_at_apcer_op(fpr, 1 - tpr, thr, 0.20)
        print(f"{name}: AUC={auc(fpr, tpr):.6f}, EER={eer*100:.2f}%, BPCER@APCER=1%: {apcer1*100:.2f}%, BPCER@APCER=20%: {apcer20*100:.2f}%")

    return pipe, clf


# ============================================================
# Commands
# ============================================================

def cmd_extract(args):
    """
    Extraction supports:
      --do-global
      --do-local --regions mouth nose left_eye right_eye
    In one run it is possible to do both global + many locals.
    """
    seed_everything(args.seed)

    # Init extractor
    FB = FourierBandsTorch(
        n_bands=args.n_bands,
        log_scale=True,
        include_phase=False,
        patch_size=None,
        dtype=torch.float32,
    )
    print("Using device:", FB.device)

    # Load train df
    df_tr = pd.read_csv(os.path.join(args.train_root_global, args.train_csv))
    if args.balance_train:
        df_tr = balance_df(df_tr)

    # Parse test pairs: name=csv
    tests = []
    for item in args.test:
        n, c = item.split("=", 1)
        tests.append((n, c))

    def extract_and_save(scope, region_or_none, train_root, test_root):
        # scope: "global" or "local"
        region = None if scope == "global" else region_or_none

        tag = "GLOBAL" if scope == "global" else f"LOCAL region={region}"
        print(f"\n[EXTRACT] {tag}")

        # Train
        df_tr_scoped = rewrite_to_region(df_tr, region)
        X_tr, y_tr = extract_features_batched(
            df_tr_scoped, train_root, FB,
            batch_size=args.batch_size,
            resize_hw=(args.img_size, args.img_size),
        )
        save_npz(args.features_out, scope, region, "train", X_tr, y_tr)
        print(f"[OK] saved: {scope}{'' if region is None else '/' + region}/train")

        # Tests
        for name, csvfile in tests:
            df_te = pd.read_csv(os.path.join(args.test_root_global, csvfile))
            df_te_scoped = rewrite_to_region(df_te, region)

            X_te, y_te = extract_features_batched(
                df_te_scoped, test_root, FB,
                batch_size=args.batch_size,
                resize_hw=(args.img_size, args.img_size),
            )
            save_npz(args.features_out, scope, region, f"test_{name}", X_te, y_te)
            print(f"[OK] saved: {scope}{'' if region is None else '/' + region}/test_{name}")

    # ------------- GLOBAL extraction -------------
    if args.do_global:
        extract_and_save("global", None, args.train_root_global, args.test_root_global)

    # ------------- MULTI-REGION LOCAL extraction -------------
    if args.do_local:
        if not args.regions or len(args.regions) == 0:
            raise ValueError("--do-local requires --regions (one or more regions)")

        for region in args.regions:
            extract_and_save("local", region, args.train_root_local, args.test_root_local)


def cmd_train_eval(args):
    """
    trains on GLOBAL residual features and tests on GLOBAL residual features.
    """
    test_names = args.test_names
    train_and_eval_global(
        features_out=args.features_out,
        model_out=args.model_out,
        test_names=test_names,
        pca_dim=args.pca_dim,
        C=args.C,
        kernel=args.kernel,
        seed=args.seed,
    )


def build_parser():
    p = argparse.ArgumentParser("Residual Fourier pipeline (extract global/local, train+eval global only)")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="Generate residual features (global and/or local)")
    e.add_argument("--train-root-global", required=True)
    e.add_argument("--train-root-local", required=True)
    e.add_argument("--test-root-global", required=True)
    e.add_argument("--test-root-local", required=True)
    e.add_argument("--train-csv", required=True)
    e.add_argument("--test", nargs="+", required=True,
                   help="Pairs name=csv e.g. opencv=labels_opencv.csv facemorpher=labels_facemorpher.csv")
    e.add_argument("--features-out", required=True)

    e.add_argument("--do-global", action="store_true", help="Extract global features")
    e.add_argument("--do-local", action="store_true", help="Extract local features for one or more regions")
    e.add_argument("--regions", nargs="+", default=None,
                   help="Used only with --do-local. Example: --regions mouth nose left_eye right_eye")

    e.add_argument("--n-bands", type=int, default=500)
    e.add_argument("--batch-size", type=int, default=512)
    e.add_argument("--img-size", type=int, default=500)
    e.add_argument("--balance-train", action="store_true")
    e.add_argument("--seed", type=int, default=42)

    te = sub.add_parser("train-eval", help="Train on GLOBAL residual features and eval on GLOBAL residual features")
    te.add_argument("--features-out", required=True)
    te.add_argument("--model-out", required=True)
    te.add_argument("--test-names", nargs="+", required=True,
                    help="Names used in extract: e.g. opencv facemorpher webmorpher mipgan_1 mipgan_2 mordiff")
    te.add_argument("--pca-dim", type=int, default=2)
    te.add_argument("--C", type=float, default=1.0)
    te.add_argument("--kernel", type=str, default="rbf")
    te.add_argument("--seed", type=int, default=42)

    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "extract":
        if not args.do_global and not args.do_local:
            raise ValueError("extract: choose at least one of --do-global / --do-local")
        cmd_extract(args)
    elif args.cmd == "train-eval":
        cmd_train_eval(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()

""" Example usage:
# 1) Extract residual features (both global and local)
python fd_mad_features.py extract \
--train-root-global path/to/your/train/dataset/aligned \
--train-root-local path/to/your/train/dataset/regions \
--train-csv train.csv \
--test-root-global path/to/your/test/dataset/aligned \
--test-root-local   path/to/your/test/dataset/regions \
--test opencv=labels_opencv.csv facemorpher=labels_facemorpher.csv webmorpher=labels_webmorph.csv mipgan_1=labels_mipgan_1.csv mipgan_2=labels_mipgan_2.csv mordiff=labels_mordiff.csv \
--features-out features_residual \
--do-global \
--do-local \
--regions mouth nose left_eye right_eye \
--balance-train

# 2) Train + Eval on GLOBAL residual features
python fd_mad_features.py train-eval \
  --features-out features_residual \
  --model-out models/global_residual \
  --test-names opencv facemorpher webmorpher mipgan_1 mipgan_2 mordiff
  
  
# The extraction of features will generate the following struc~ture:
features_residual/
  global/
    train.npz
    test_opencv.npz
    ...
  local/
    mouth/
      train.npz
      test_opencv.npz
      ...
    nose/
    left_eye/
    right_eye/
"""
