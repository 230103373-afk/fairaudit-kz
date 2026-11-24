"""FairAudit-KZ prototype: audit -> explain -> mitigate pipeline (TRL 3-4 proof of concept).

Dataset: UCI Adult (48,842 records), protected attribute = sex.
Run:  python fairaudit.py --data ../data/adult.data --out results
"""
import argparse, json, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer

warnings.filterwarnings("ignore")
COLS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
        "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
CAT = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "native_country"]
NUM = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
PROXIES = ["relationship", "marital_status"]  # identified by the SHAP/proxy audit step


def load(path):
    df = pd.read_csv(path, header=None, names=COLS, skipinitialspace=True, na_values="?").dropna()
    y = df["income"].str.contains(">50K").astype(int).values
    s = (df["sex"] == "Female").astype(int).values  # 1 = unprivileged group
    return df.drop(columns=["income", "fnlwgt"]), y, s


def prep(cat, num):
    return ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
                              ("n", StandardScaler(), num)])


def metrics(y, p, s):
    r1, r0 = p[s == 1].mean(), p[s == 0].mean()  # selection rates female / male
    tpr = [p[(s == g) & (y == 1)].mean() for g in (0, 1)]
    fpr = [p[(s == g) & (y == 0)].mean() for g in (0, 1)]
    return dict(acc=accuracy_score(y, p), dp_diff=abs(r0 - r1), di_ratio=r1 / r0 if r0 else np.nan,
                eo_diff=max(abs(tpr[0] - tpr[1]), abs(fpr[0] - fpr[1])), sel_f=r1, sel_m=r0)


def reweigh(y, s):
    """Kamiran & Calders (2012) reweighing: w(g,c) = P(g)P(c)/P(g,c)."""
    w = np.ones(len(y))
    for g in (0, 1):
        for c in (0, 1):
            m = (s == g) & (y == c)
            w[m] = (s == g).mean() * (y == c).mean() / m.mean()
    return w


def run_seed(X, y, s, seed):
    Xtr, Xte, ytr, yte, str_, ste = train_test_split(X, y, s, test_size=0.3, stratify=y, random_state=seed)
    feats_all = CAT + ["sex"], NUM
    feats_unaware = CAT, NUM
    feats_proxy = [c for c in CAT if c not in PROXIES], NUM
    gbm = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    lr = lambda: LogisticRegression(max_iter=2000)
    out = {}

    def fit(cols, est, w=None):
        pipe = Pipeline([("p", prep(*cols)), ("m", est)])
        pipe.fit(Xtr[cols[0] + cols[1]], ytr, m__sample_weight=w) if w is not None else pipe.fit(Xtr[cols[0] + cols[1]], ytr)
        return pipe

    # B0 baseline models (sex used as feature)
    for name, est in [("LR baseline", lr), ("GBM baseline", gbm)]:
        m = fit(feats_all, est())
        out[name] = metrics(yte, m.predict(Xte[feats_all[0] + feats_all[1]]), ste)
    # M1 fairness through unawareness
    m = fit(feats_unaware, gbm())
    out["GBM unaware (drop sex)"] = metrics(yte, m.predict(Xte[CAT + NUM]), ste)
    # M2 SHAP-guided proxy removal
    cols = feats_proxy[0] + feats_proxy[1]
    m = fit(feats_proxy, gbm())
    out["GBM proxy-aware (SHAP-guided)"] = metrics(yte, m.predict(Xte[cols]), ste)
    # M3 reweighing (pre-processing)
    m = fit(feats_unaware, gbm(), w=reweigh(ytr, str_))
    out["GBM + reweighing"] = metrics(yte, m.predict(Xte[CAT + NUM]), ste)
    # M4 exponentiated gradient, demographic parity (in-processing)
    P = prep(*feats_unaware).fit(Xtr[CAT + NUM])
    eg = ExponentiatedGradient(LogisticRegression(max_iter=2000), DemographicParity(difference_bound=0.02))
    eg.fit(P.transform(Xtr[CAT + NUM]), ytr, sensitive_features=str_)
    out["LR + ExpGrad (DP)"] = metrics(yte, eg.predict(P.transform(Xte[CAT + NUM]), random_state=seed), ste)
    # M5 threshold optimizer on GBM (post-processing)
    base = fit(feats_unaware, gbm())
    to = ThresholdOptimizer(estimator=base, constraints="demographic_parity", prefit=True, predict_method="predict_proba")
    to.fit(Xtr[CAT + NUM], ytr, sensitive_features=str_)
    out["GBM + ThresholdOpt (DP)"] = metrics(yte, to.predict(Xte[CAT + NUM], sensitive_features=ste, random_state=seed), ste)
    # M6 PROPOSED: FairAudit pipeline = reweighing + tolerance-tuned threshold optimisation.
    # tol is selected on a validation split (never on test): largest-accuracy tol with DI >= 0.8.
    Xa, Xv, ya, yv, sa, sv = train_test_split(Xtr, ytr, str_, test_size=0.25, stratify=ytr, random_state=seed)
    pipe = Pipeline([("p", prep(*feats_unaware)), ("m", gbm())]).fit(Xa[CAT + NUM], ya, m__sample_weight=reweigh(ya, sa))
    best = None
    for tol in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        t = ThresholdOptimizer(estimator=pipe, constraints="demographic_parity", prefit=True,
                               predict_method="predict_proba", tol=tol).fit(Xa[CAT + NUM], ya, sensitive_features=sa)
        mv = metrics(yv, t.predict(Xv[CAT + NUM], sensitive_features=sv, random_state=seed), sv)
        if mv["di_ratio"] >= 0.8 and (best is None or mv["acc"] > best[0]):
            best = (mv["acc"], t, tol)
    out["FairAudit (proposed)"] = metrics(yte, best[1].predict(Xte[CAT + NUM], sensitive_features=ste, random_state=seed), ste)
    out["FairAudit (proposed)"]["tol"] = best[2]
    return out


def proxy_audit(X, s, seed=0):
    """How well do the remaining features predict sex? AUC near 0.5 = no proxy."""
    res = {}
    for label, cat in [("all features (no sex)", CAT), ("without relationship & marital status",
                                                         [c for c in CAT if c not in PROXIES])]:
        Xtr, Xte, s_tr, s_te = train_test_split(X[cat + NUM], s, test_size=0.3, random_state=seed, stratify=s)
        m = Pipeline([("p", prep(cat, NUM)), ("m", HistGradientBoostingClassifier(random_state=seed))]).fit(Xtr, s_tr)
        res[label] = roc_auc_score(s_te, m.predict_proba(Xte)[:, 1])
    return res


def shap_audit(X, y, s, outdir, seed=0):
    import shap
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    P = prep(CAT, NUM).fit(Xtr[CAT + NUM])
    names = list(P.get_feature_names_out())
    m = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(P.transform(Xtr[CAT + NUM]), ytr)
    sample = P.transform(Xte[CAT + NUM].sample(2000, random_state=seed))
    sv = shap.TreeExplainer(m).shap_values(sample)
    imp = pd.Series(np.abs(sv).mean(0), index=names)
    groups = {c: imp[[n for n in names if n.startswith(f"c__{c}_")]].sum() for c in CAT}
    groups.update({c: imp[f"n__{c}"] for c in NUM})
    g = pd.Series(groups).sort_values(ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    colors = ["#c0392b" if k in PROXIES else "#5b7aa8" for k in g.index]
    ax.barh([k.replace("_", " ") for k in g.index], g.values, color=colors)
    ax.set_xlabel("mean |SHAP value| (log-odds)")
    ax.set_title("SHAP audit of the 'sex-blind' model: proxies in red", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_shap.png"), dpi=200); plt.close(fig)
    return g.sort_values(ascending=False).round(4).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/adult.data"); ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    X, y, s = load(a.data)
    print(f"Records: {len(y)}, positive rate: {y.mean():.3f}, female share: {s.mean():.3f}")
    rows = []
    for seed in range(a.seeds):
        for k, v in run_seed(X, y, s, seed).items():
            rows.append(dict(method=k, seed=seed, **v))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "raw_results.csv"), index=False)
    order = list(dict.fromkeys(df["method"]))
    summ = df.groupby("method")[["acc", "dp_diff", "di_ratio", "eo_diff"]].agg(["mean", "std"]).loc[order]
    summ.to_csv(os.path.join(a.out, "summary.csv")); print(summ.round(4))
    base = summ.loc["GBM baseline"]
    crit = {}
    for mth in order:
        r = summ.loc[mth]
        crit[mth] = dict(dp_reduction_pct=float(100 * (1 - r[("dp_diff", "mean")] / base[("dp_diff", "mean")])),
                         acc_drop_pp=float(100 * (base[("acc", "mean")] - r[("acc", "mean")])),
                         di_ok=bool(r[("di_ratio", "mean")] >= 0.8))
        c = crit[mth]; c["success"] = c["dp_reduction_pct"] >= 50 and c["acc_drop_pp"] <= 2 and c["di_ok"]
    json.dump(dict(proxy_auc=proxy_audit(X, s), shap_top10=shap_audit(X, y, s, a.out), criteria=crit),
              open(os.path.join(a.out, "audit.json"), "w"), indent=2)
    # trade-off figure
    fig, ax = plt.subplots(figsize=(6, 3.8))
    b_acc, b_dp = base[("acc", "mean")], base[("dp_diff", "mean")]
    ax.axhspan(b_acc - 0.02, 0.875, xmin=0, xmax=(b_dp * 0.5) / 0.2, color="#1e8449", alpha=0.08)
    ax.text(0.004, 0.873, "success zone:\n>=50% smaller DP gap,\n<=2 pp accuracy loss", fontsize=7, color="#1e8449", va="top")
    offs = {"GBM baseline": (5, 4), "GBM unaware (drop sex)": (-20, -12), "LR baseline": (-30, 8)}
    for mth in order:
        r = summ.loc[mth]; ok = crit[mth]["success"]
        ax.errorbar(r[("dp_diff", "mean")], r[("acc", "mean")], xerr=r[("dp_diff", "std")], yerr=r[("acc", "std")],
                    fmt="o", ms=7, color="#1e8449" if ok else "#7f8c8d", capsize=2)
        ax.annotate(mth, (r[("dp_diff", "mean")], r[("acc", "mean")]), fontsize=7, xytext=offs.get(mth, (6, 3)),
                    textcoords="offset points", fontweight="bold" if ok else "normal")
    ax.set_xlim(0, 0.2); ax.set_ylim(0.825, 0.875)
    ax.set_xlabel("Demographic parity difference (lower = fairer)"); ax.set_ylabel("Accuracy (test)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Fairness-accuracy trade-off, UCI Adult (5 seeds, mean +/- sd)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_tradeoff.png"), dpi=200); plt.close(fig)
    print(json.dumps(json.load(open(os.path.join(a.out, "audit.json"))), indent=1))


if __name__ == "__main__":
    main()
