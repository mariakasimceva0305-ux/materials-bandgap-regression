from pathlib import Path
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import re, json, hashlib, math, functools, time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
from catboost import CatBoostRegressor
import joblib

ROOT = Path(__file__).resolve().parent
ELEMENTS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()


def parse(s):
    if not isinstance(s, str):
        return None
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", s)
    if "".join(a + b for a, b in tokens) != s or not tokens:
        return None
    counts = {}
    for e, n in tokens:
        if e not in ELEMENTS:
            return None
        counts[e] = counts.get(e, 0) + int(n or 1)
    if min(counts.values()) <= 0:
        return None
    total = sum(counts.values())
    g = functools.reduce(math.gcd, counts.values())
    identity = ";".join(f"{e}:{counts[e] // g}" for e in sorted(counts))
    return [counts.get(e, 0) / total for e in ELEMENTS], identity


def metrics(y, p):
    return dict(
        mae_eV=mean_absolute_error(y, p),
        rmse_eV=float(np.sqrt(mean_squared_error(y, p))),
        r2=r2_score(y, p),
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    SRC = args.data
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SRC)
    selected = []
    features = []
    groups = []
    for i, row in raw.iterrows():
        parsed = parse(row["formula"])
        if (
            parsed is None
            or pd.isna(row["band_gap_value"])
            or row["band_gap_value"] < 0
        ):
            continue
        selected.append(i)
        features.append(parsed[0])
        groups.append(parsed[1])
    X = np.array(features)
    y = raw.loc[selected, "band_gap_value"].to_numpy(float)
    groups = np.array(groups)
    # Exact material IDs cannot straddle splits; composition grouping is stricter.
    train, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            X, y, groups
        )
    )
    assert not set(groups[train]) & set(groups[test])
    folds = list(GroupKFold(n_splits=5).split(X[train], y[train], groups[train]))
    for a, b in folds:
        assert not set(groups[train][a]) & set(groups[train][b])
    models = {
        "median": DummyRegressor(strategy="median"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "catboost": CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.04,
            loss_function="RMSE",
            random_seed=42,
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                alpha=0.1,
                max_iter=500,
                random_state=42,
                early_stopping=False,
            ),
        ),
    }
    protocol = dict(
        task="Predict DFT band gap from elemental fractions; not battery-performance prediction",
        source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
        raw_rows=len(raw),
        labelled_parseable_rows=len(y),
        unique_compositions=len(set(groups)),
        train_n=len(train),
        test_n=len(test),
        seed=42,
        selection="lowest mean 5-fold grouped CV MAE on training set",
        features=ELEMENTS,
        notes=[
            "No target-derived features or UUID/source identifiers.",
            "No target metric threshold or test-driven tuning.",
            "Fixed candidate configurations; negative predictions clipped to physically admissible zero for all models.",
            "Composition-only features do not distinguish polymorphs.",
            "Historical resume R2 is not a target or verified baseline.",
        ],
    )
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "source_row": selected,
            "composition": groups,
            "split": ["test" if i in set(test) else "train" for i in range(len(y))],
        }
    ).to_csv(OUT / "split_manifest.csv", index=False)
    cv = []
    for name, model in models.items():
        for fold, (a, b) in enumerate(folds):
            t = time.time()
            m = clone(model)
            m.fit(X[train][a], y[train][a])
            p = np.maximum(0, m.predict(X[train][b]))
            row = dict(
                model=name,
                fold=fold,
                **metrics(y[train][b], p),
                seconds=time.time() - t,
            )
            cv.append(row)
            print(json.dumps(row), flush=True)
        pd.DataFrame(cv).to_csv(OUT / "cv_results.csv", index=False)
    best = pd.DataFrame(cv).groupby("model").mae_eV.mean().idxmin()
    # Evaluate the predeclared candidates once; select deployment candidate by CV only.
    results = []
    predictions = pd.DataFrame(
        {
            "source_row": np.array(selected)[test],
            "composition": groups[test],
            "target_eV": y[test],
        }
    )
    for name, model in models.items():
        m = clone(model)
        m.fit(X[train], y[train])
        p = np.maximum(0, m.predict(X[test]))
        predictions[name] = p
        results.append(
            dict(model=name, selected_by_cv=name == best, **metrics(y[test], p))
        )
        if name == best:
            joblib.dump(m, OUT / "selected_model.joblib")
    predictions.to_csv(OUT / "test_predictions.csv", index=False)
    pd.DataFrame(results).to_csv(OUT / "test_results.csv", index=False)
    # Composition-cluster bootstrap, so repeated polymorphs are sampled together.
    rng = np.random.default_rng(2026)
    ug = np.unique(groups[test])
    idx = {g: np.flatnonzero(groups[test] == g) for g in ug}
    scores = []
    for _ in range(1000):
        ix = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
        scores.append(metrics(y[test][ix], predictions[best].to_numpy()[ix]))
    ci = {
        k: np.quantile([r[k] for r in scores], [0.025, 0.975]).tolist()
        for k in scores[0]
    }
    (OUT / "summary.json").write_text(
        json.dumps(
            dict(
                selected_model=best,
                test_results=results,
                selected_cluster_bootstrap_95=ci,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print("COMPLETE", best, json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
