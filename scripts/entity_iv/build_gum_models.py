#!/usr/bin/env python3
"""Fit entity transition cost models on GUM for two channels, ROLE and FORM, used by
calculate_entity_iv.py for the d3_transition_form / d3_transition_role_mlp / d3_transition_form_mlp channels.
Usage: uv run python build_gum_models.py --gum <GUM dep dir> --output gum_transition_models.json
(torch is needed for training only; the scorer reads the saved weights with numpy.)

Fit entity transition cost models on GUM for two channels, ROLE and FORM,
each with two estimators: an additively smoothed bigram table (the existing
approach) and a one-hidden-layer MLP.

Observation unit (same as build_gum_transition_cost.py): every mention paired
with the same entity's previous mention anywhere earlier in the document.

    previous state  ->  current state
    role:  fine inventory of entity_roles (12 states) or coarse (4)
    form:  form_inventory.FORMS (15 states), derived from tokens

Bigram:  P(cur | prev) = (N(prev,cur) + a) / (N(prev,.) + a*K), a = 0.5, cost = -log2 P.
         Conditions only on the previous state of the SAME channel.
MLP:     P(cur | x), x = [one-hot prev role (fine), one-hot prev form, one-hot gap
         bucket, log1p(gap), log1p(#earlier mentions)], one hidden layer (ReLU),
         softmax output. It sees both channels' previous state and recency, which
         a bigram table cannot. Cost = -log2 P.

A 5-fold split BY DOCUMENT reports held-out cost in bits per transition for each
estimator, and for an MLP restricted to the bigram's information (previous
state of the same channel only) to separate "better estimator" from "more
information". The final models are refit on all documents (MLP with a 10%
document dev split for early stopping) and written to one JSON. MLP weights are
stored as plain lists so the scorer needs only numpy.
"""
from __future__ import annotations
import argparse, json, math, random
from collections import Counter, defaultdict
import numpy as np

import build_gum_transition_cost as B          # GUM parser + entity span reader (read-only)
from entity_roles import INVENTORIES, mention_head, mention_role, project_state
from form_inventory import FORMS, mention_form

GAP_BUCKETS = ("same", "1", "2-3", "4-6", "7+")


def gap_bucket(gap: int) -> str:
    if gap <= 0: return "same"
    if gap == 1: return "1"
    if gap <= 3: return "2-3"
    if gap <= 6: return "4-6"
    return "7+"


def document_observations(text: str, role_inventory: str):
    """(prev_role, prev_form, gap, n_prev, cur_role, cur_form) for each non-first mention."""
    roles = INVENTORIES[role_inventory]
    mentions = []
    for s_idx, tokens in enumerate(B.parse_conllu(text)):
        for start, end, eid in B.entity_spans(tokens):
            h = mention_head(tokens, start, end)
            role = project_state(mention_role(tokens, h, role_inventory), roles) if roles else mention_role(tokens, h, role_inventory)
            mentions.append((s_idx, start, end, eid, role, mention_form(tokens, start, end)))
    mentions.sort(key=lambda m: (m[0], m[1], -m[2], m[3]))
    last, count, obs = {}, Counter(), []
    for s_idx, _s, _e, eid, role, form in mentions:
        if eid in last:
            p_role, p_form, p_sent = last[eid]
            obs.append((p_role, p_form, s_idx - p_sent, count[eid], role, form))
        last[eid] = (role, form, s_idx)
        count[eid] += 1
    return obs


def bigram_costs(pairs, states, alpha=0.5):
    counts = Counter(pairs)
    table, probs = {}, {}
    for p in states:
        row = sum(counts[(p, c)] for c in states)
        den = row + alpha * len(states)
        probs[p] = {c: (counts[(p, c)] + alpha) / den for c in states}
        table[p] = {c: -math.log2(probs[p][c]) for c in states}
    return table, probs, counts


class FeatureSpec:
    def __init__(self, roles, forms, inputs=("role", "form", "gap", "count")):
        self.roles, self.forms, self.inputs = list(roles), list(forms), tuple(inputs)
        self.r_ix = {r: i for i, r in enumerate(self.roles)}
        self.f_ix = {f: i for i, f in enumerate(self.forms)}
        self.g_ix = {g: i for i, g in enumerate(GAP_BUCKETS)}
        self.dim = (len(roles) if "role" in inputs else 0) + (len(forms) if "form" in inputs else 0) \
            + (len(GAP_BUCKETS) + 1 if "gap" in inputs else 0) + (1 if "count" in inputs else 0)

    def vector(self, p_role, p_form, gap, n_prev):
        x = np.zeros(self.dim, dtype=np.float32); o = 0
        if "role" in self.inputs:
            x[o + self.r_ix.get(p_role, self.r_ix.get("other", 0))] = 1; o += len(self.roles)
        if "form" in self.inputs:
            x[o + self.f_ix.get(p_form, self.f_ix["other"])] = 1; o += len(self.forms)
        if "gap" in self.inputs:
            x[o + self.g_ix[gap_bucket(gap)]] = 1; x[o + len(GAP_BUCKETS)] = math.log1p(max(gap, 0)); o += len(GAP_BUCKETS) + 1
        if "count" in self.inputs:
            x[o] = math.log1p(n_prev)
        return x

    def to_json(self):
        return {"roles": self.roles, "forms": self.forms, "inputs": list(self.inputs),
                "gap_buckets": list(GAP_BUCKETS), "dim": self.dim}


def train_mlp(X, y, n_out, Xdev, ydev, hidden=64, epochs=80, lr=1e-3, wd=1e-4, seed=0, batch=256):
    import torch, torch.nn as nn
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, n_out))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.long)
    Xd, yd = torch.tensor(Xdev), torch.tensor(ydev, dtype=torch.long)
    best, best_state, patience = float("inf"), None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad(); loss = lossf(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            dev = lossf(model(Xd), yd).item()
        if dev < best - 1e-4:
            best, patience = dev, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 8: break
    model.load_state_dict(best_state)
    W1, b1 = model[0].weight.detach().numpy(), model[0].bias.detach().numpy()
    W2, b2 = model[2].weight.detach().numpy(), model[2].bias.detach().numpy()
    return {"W1": W1.tolist(), "b1": b1.tolist(), "W2": W2.tolist(), "b2": b2.tolist(),
            "hidden": hidden, "dev_loss_nats": best, "epochs_run": ep + 1}


def mlp_log2_probs(weights, X):
    H = np.maximum(0, X @ np.asarray(weights["W1"], dtype=np.float32).T + np.asarray(weights["b1"], dtype=np.float32))
    Z = H @ np.asarray(weights["W2"], dtype=np.float32).T + np.asarray(weights["b2"], dtype=np.float32)
    Z = Z - Z.max(axis=1, keepdims=True)
    logp = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
    return logp / math.log(2)


def evaluate(obs_train, obs_test, channel, states, spec_full, spec_same, seed):
    """Held-out mean cost (bits/transition) of bigram, MLP(same-channel prev only), MLP(full)."""
    ci = 4 if channel == "role" else 5      # index of current state
    pi = 0 if channel == "role" else 1      # index of previous state of the same channel
    s_ix = {s: i for i, s in enumerate(states)}
    table, _, _ = bigram_costs([(o[pi], o[ci]) for o in obs_train], states)
    bigram = np.mean([table[o[pi]][o[ci]] for o in obs_test])
    out = {"bigram": float(bigram)}
    for name, spec in (("mlp_same", spec_same), ("mlp_full", spec_full)):
        docs = None
        X = np.stack([spec.vector(o[0], o[1], o[2], o[3]) for o in obs_train]); y = np.array([s_ix[o[ci]] for o in obs_train])
        Xt = np.stack([spec.vector(o[0], o[1], o[2], o[3]) for o in obs_test]); yt = np.array([s_ix[o[ci]] for o in obs_test])
        n_dev = max(1, len(X) // 10)
        w = train_mlp(X[n_dev:], y[n_dev:], len(states), X[:n_dev], y[:n_dev], seed=seed)
        lp = mlp_log2_probs(w, Xt)
        out[name] = float(-lp[np.arange(len(yt)), yt].mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gum", required=True, help="GUM release directory (dep/*.conllu) or zip")
    ap.add_argument("--output", required=True)
    ap.add_argument("--role-inventory", default="fine", choices=("fine", "coarse"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()
    roles = list(INVENTORIES[args.role_inventory]); forms = list(FORMS)

    docs = []
    for name, text in B.iter_gum_documents(args.gum):
        docs.append((name, document_observations(text, args.role_inventory)))
    n_obs = sum(len(o) for _, o in docs)
    print(f"GUM documents {len(docs)}, transitions {n_obs}", flush=True)

    spec_full = FeatureSpec(roles, forms)
    spec_role_only = FeatureSpec(roles, forms, inputs=("role",))
    spec_form_only = FeatureSpec(roles, forms, inputs=("form",))

    # ---- document-level cross-validation
    rng = random.Random(0); order = list(range(len(docs))); rng.shuffle(order)
    folds = [order[i::args.folds] for i in range(args.folds)]
    cv = {"role": defaultdict(list), "form": defaultdict(list)}
    for k, test_ix in enumerate(folds):
        test = set(test_ix)
        tr = [o for i, (_, obs) in enumerate(docs) if i not in test for o in obs]
        te = [o for i, (_, obs) in enumerate(docs) if i in test for o in obs]
        for channel, states, spec_same in (("role", roles, spec_role_only), ("form", forms, spec_form_only)):
            r = evaluate(tr, te, channel, states, spec_full, spec_same, seed=k)
            for kk, v in r.items(): cv[channel][kk].append(v)
            print(f"fold {k} {channel}: " + " ".join(f"{kk}={v:.3f}" for kk, v in r.items()), flush=True)
    cv_summary = {ch: {est: {"mean_bits": float(np.mean(v)), "sd": float(np.std(v))} for est, v in d.items()} for ch, d in cv.items()}
    print("\nheld-out bits per transition (mean over folds):")
    for ch in cv_summary:
        print(f"  {ch}: " + ", ".join(f"{est} {m['mean_bits']:.3f}±{m['sd']:.3f}" for est, m in cv_summary[ch].items()))

    # ---- final fit on all documents
    all_obs = [o for _, obs in docs for o in obs]
    out = {"schema_version": 1, "source": "GUM dep/*.conllu", "role_inventory": args.role_inventory,
           "roles": roles, "forms": forms, "smoothing": 0.5, "cost_unit": "bits",
           "document_count": len(docs), "transition_count": n_obs, "cv": cv_summary, "channels": {}}
    for channel, states, ci, pi in (("role", roles, 4, 0), ("form", forms, 5, 1)):
        table, probs, counts = bigram_costs([(o[pi], o[ci]) for o in all_obs], states)
        out["channels"][f"{channel}_bigram"] = {"states": states, "transition_costs": table,
                                                 "row_totals": {p: sum(counts[(p, c)] for c in states) for p in states}}
        s_ix = {s: i for i, s in enumerate(states)}
        dev_docs = set(order[: max(1, len(docs) // 10)])
        tr = [o for i, (_, obs) in enumerate(docs) if i not in dev_docs for o in obs]
        dv = [o for i, (_, obs) in enumerate(docs) if i in dev_docs for o in obs]
        X = np.stack([spec_full.vector(*o[:4]) for o in tr]); y = np.array([s_ix[o[ci]] for o in tr])
        Xd = np.stack([spec_full.vector(*o[:4]) for o in dv]); yd = np.array([s_ix[o[ci]] for o in dv])
        w = train_mlp(X, y, len(states), Xd, yd, hidden=args.hidden, seed=0)
        out["channels"][f"{channel}_mlp"] = {"states": states, "features": spec_full.to_json(), "weights": w}
        print(f"final {channel}_mlp: dev loss {w['dev_loss_nats'] / math.log(2):.3f} bits, epochs {w['epochs_run']}")
    B.atomic_json(args.output, out)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
