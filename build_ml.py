"""Offline machine-learning step: group the 57,209 fall narratives into "story groups".

Pipeline: clean text -> TF-IDF -> TruncatedSVD (PCA-style reduction) -> K-Means -> 2D PCA map.
Run once:  python build_ml.py   (outputs are saved in data/ and committed, so the app does not refit anything)
"""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

from agent import mechanism_text

SEED = 42
LABELS = {
 "0": {
  "label": "Nursing-home falls and transfers",
  "note": "Notes that mention a nursing home or facility; 96% are in the care-facility location category."
 },
 "1": {
  "label": "Fell to the floor, often unwitnessed",
  "note": "Falls to the floor, often found by someone or not seen; bathrooms and floors come up often."
 },
 "2": {
  "label": "Got up or out of bed, then hit the head",
  "note": "Falls that include hitting the head while getting up."
 },
 "3": {
  "label": "Falls getting out of bed",
  "note": "Bed exits, including at facilities."
 },
 "4": {
  "label": "Tripped or fell and struck the head",
  "note": "Falls with a head strike."
 },
 "5": {
  "label": "Everyday trips, steps, and slips (largest group)",
  "note": "A broad group: trips, stairs, steps, chairs, and many other situations."
 },
 "6": {
  "label": "Fell onto the hip",
  "note": "Notes that describe landing on the hip or hip pain; the highest share ended in hospital admission."
 },
 "7": {
  "label": "Lost balance",
  "note": "Notes that say the person lost their balance, often while doing a routine task."
 }
}  # hand-written names for the 8 groups, chosen after reading each group's top words and notes
KEEP = ["case_number", "treatment_date", "age_years", "sex_label", "location_label", "disposition_label",
        "diagnosis_label", "admitted", "narrative"]

df = pd.read_csv("../neiss2024_older_adult_falls.csv")[KEEP]
print("rows", len(df))

# 1. Preprocessing: keep only how the fall happened (text before "DX"), drop ages, expand abbreviations.
text = df["narrative"].map(mechanism_text)
extra_stop = {"fall", "fell", "falls", "falling", "patient", "presents", "present", "pt", "dx", "yof", "yom", "old", "year", "years"}
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5, sublinear_tf=True, max_features=20000,
                      stop_words=list(ENGLISH_STOP_WORDS | extra_stop))
X = vec.fit_transform(text)
print("tf-idf matrix", X.shape)

# 2. Reduce to 50 dimensions (TruncatedSVD is PCA for sparse text), then length-normalize.
svd = TruncatedSVD(n_components=50, random_state=SEED)
Z = Normalizer().fit_transform(svd.fit_transform(X))
explained = float(svd.explained_variance_ratio_.sum())
print("variance kept by 50 components", round(explained, 3))

# 3. Choose k with the silhouette score on a random sample (higher = better separated groups).
rng = np.random.RandomState(SEED)
sample = rng.choice(len(Z), 8000, replace=False)
scores = {}
for k in range(5, 13):
    labels = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit_predict(Z)
    scores[k] = float(silhouette_score(Z[sample], labels[sample]))
    print("k", k, "silhouette", round(scores[k], 4))
K = 8  # chosen for interpretability: silhouette is low and fairly flat across k, so I use the smallest k that gives distinct, readable groups
km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Z)
group = km.labels_

# 4. Evidence on limitations.
stab = []
for seed in (7, 99, 2024):
    other = KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(Z)
    stab.append(float(adjusted_rand_score(group, other)))

themes = {  # simple keyword rules, used only to check whether groups line up with themes I can read directly
    "dizzy_faint": r"DIZZ|LIGHTHEAD|SYNCOP|FAINT|PASSED OUT|BLACKED OUT",
    "found_down": r"\bFOUND\b|UNWITNESSED|UNABLE TO GET UP|COULD NOT GET UP|CAN.?T GET UP",
    "trip_slip": r"\bTRIP|\bSLIP|\bSLID",
    "bed": r"\bBED\b",
    "bathroom_toilet": r"BATHROOM|SHOWER|\bTUB\b|TOILET|COMMODE",
    "stairs": r"STAIR|\bSTEPS?\b",
    "lost_balance": r"LOST (HER|HIS|THEIR)? ?BALANCE|OFF BALANCE|LEGS GAVE|WEAK",
}
up = df["narrative"].str.upper()
primary = pd.Series("none", index=df.index)
for name, pat in reversed(list(themes.items())):  # first theme in the dict wins
    primary[up.str.contains(pat, regex=True)] = name
has_theme = primary != "none"
nmi = float(normalized_mutual_info_score(primary[has_theme], group[has_theme.values]))

# How much does the group say about hospital admission? (descriptive check, not a model for use)
y = df["admitted"].values
oh = np.eye(K)[group]
tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y)
auc_group = float(roc_auc_score(y[te], LogisticRegression(max_iter=500).fit(oh[tr], y[tr]).predict_proba(oh[te])[:, 1]))
age = df["age_years"].fillna(df["age_years"].median()).values.reshape(-1, 1)
auc_age = float(roc_auc_score(y[te], LogisticRegression(max_iter=500).fit(age[tr], y[tr]).predict_proba(age[te])[:, 1]))

# 5. Group profiles and top terms, for naming the groups.
terms = np.array(vec.get_feature_names_out())
profiles = []
for g in range(K):
    m = group == g
    centroid = np.asarray(X[m].mean(axis=0)).ravel()
    top = terms[np.argsort(centroid)[::-1][:10]].tolist()
    sub = df[m]
    profiles.append({
        "group": g, "size": int(m.sum()), "share": round(float(m.mean()), 4),
        "top_terms": top,
        "admitted_rate": round(float(sub["admitted"].mean()), 4),
        "median_age": float(sub["age_years"].median()),
        "home_share": round(float((sub["location_label"] == "Home").mean()), 4),
        "other_public_share": round(float((sub["location_label"] == "Other public property").mean()), 4),
        "theme_mix": {k: round(float((primary[m] == k).mean()), 3) for k in themes},
    })

# 6. 2D map for the app (PCA of the 50-d vectors).
xy = PCA(n_components=2, random_state=SEED).fit_transform(Z)
df["group"], df["x"], df["y"] = group, xy[:, 0].round(4), xy[:, 1].round(4)
df.to_csv("data/falls.csv.gz", index=False, compression="gzip")  # the app loads this one file

metrics = {
    "n_narratives": int(len(df)), "tfidf_vocabulary": int(X.shape[1]), "svd_components": 50,
    "variance_kept_by_svd": round(explained, 4), "k_chosen": K,
    "silhouette_by_k": {str(k): round(v, 4) for k, v in scores.items()},
    "seed_stability_ARI": [round(v, 3) for v in stab], "seed_stability_mean": round(float(np.mean(stab)), 3),
    "nmi_vs_keyword_themes": round(nmi, 3), "narratives_with_a_keyword_theme": int(has_theme.sum()),
    "auc_admission_from_group_only": round(auc_group, 3), "auc_admission_from_age_only": round(auc_age, 3),
    "overall_admitted_rate": round(float(df["admitted"].mean()), 4),
}
json.dump({"metrics": metrics, "groups": profiles, "labels": LABELS}, open("data/ml_info.json", "w"), indent=1)
print(json.dumps(metrics, indent=1))
for p in profiles:
    print(p["group"], p["size"], p["admitted_rate"], p["top_terms"][:8], p["home_share"], p["other_public_share"])
