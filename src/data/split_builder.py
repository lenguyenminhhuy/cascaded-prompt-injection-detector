"""Build the 5 cascade-pid splits (PLAN.md Chunk 6, SCHEMA.md §1).

Splits and the questions they answer:
  train               in-channel, in-domain detection training (class-balanced)
  cal                 calibration; oversampled toward ambiguous/hard near-threshold
  test_in_dist        in-channel, in-domain; realistic deployment imbalance
  test_cross_channel  (payload_family x channel) cells reserved ENTIRELY
  test_cross_domain   domain != sql; never trained

Holdout discipline:
  - cross_domain is decided purely by domain (pre-stamped by the synthetic slice).
  - cross_channel reserves whole (family, channel) cells of POSITIVES, plus a
    matched sample of channel-matched negatives, so the split has a real FPR.
  - multi-intent items are excluded from cross_channel to keep the signal clean.
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict

from src.data.schema import Sample

log = logging.getLogger(__name__)


class SplitBuilder:
    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg = cfg
        self.seed = seed
        self.holdout_cells = {tuple(c) for c in cfg.get("cross_channel_holdout_cells", [])}
        self.exclude_mi = cfg.get("exclude_multi_intent_from_cross_channel", True)
        idc = cfg.get("in_dist", {})
        self.ratios = idc.get("ratios", {"train": 0.7, "cal": 0.15, "test_in_dist": 0.15})
        self.test_inj_ratio = idc.get("test_in_dist_injection_ratio", 0.15)
        self.train_inj_ratio = idc.get("train_injection_ratio", 0.5)
        self.diff_weights = cfg.get("calibration", {}).get(
            "difficulty_weights", {"ambiguous": 3.0, "hard": 2.0, "easy": 1.0})

    # ------------------------------------------------------------------ #
    def build(self, pool: list[Sample]) -> dict[str, list[Sample]]:
        rng = random.Random(self.seed)
        out: dict[str, list[Sample]] = defaultdict(list)

        # 1) cross-domain: anything not sql.
        sql, cross_domain = [], []
        for s in pool:
            (cross_domain if s.domain != "sql" else sql).append(s)
        for s in cross_domain:
            s.split = "test_cross_domain"
        out["test_cross_domain"] = cross_domain

        # 2) cross-channel: reserve whole positive cells + matched negatives.
        def is_holdout(s: Sample) -> bool:
            if s.label != "injected":
                return False
            if self.exclude_mi and "multi-intent" in (s.notes or ""):
                return False
            return (s.payload_family, s.channel) in self.holdout_cells

        cc_pos = [s for s in sql if is_holdout(s)]
        holdout_channels = {ch for (_, ch) in self.holdout_cells}
        rest = [s for s in sql if not is_holdout(s)]

        benign_by_channel = defaultdict(list)
        for s in rest:
            if s.label == "benign":
                benign_by_channel[s.channel].append(s)

        cc_neg = []
        n_pos_by_ch = Counter(s.channel for s in cc_pos)
        for ch in sorted(holdout_channels):  # sorted: stable RNG consumption order
            pool_b = benign_by_channel.get(ch, [])
            rng.shuffle(pool_b)
            # reserve ~ as many benign as positives in that channel (min 10), capped at 40% of channel benign
            want = min(max(n_pos_by_ch.get(ch, 0), 10), int(len(pool_b) * 0.4))
            cc_neg.extend(pool_b[:want])
        cc_neg_ids = {id(s) for s in cc_neg}
        for s in cc_pos + cc_neg:
            s.split = "test_cross_channel"
        out["test_cross_channel"] = cc_pos + cc_neg

        in_dist = [s for s in rest if id(s) not in cc_neg_ids]

        # 3) in-distribution -> cal / test_in_dist / train
        self._split_in_dist(rng, in_dist, out)

        # stamp + validate
        final: dict[str, list[Sample]] = {}
        for name, items in out.items():
            for s in items:
                s.split = name
                s.validate()
            final[name] = items
        return final

    # ------------------------------------------------------------------ #
    def _split_in_dist(self, rng, in_dist, out):
        inj = [s for s in in_dist if s.label == "injected"]
        ben = [s for s in in_dist if s.label == "benign"]
        rng.shuffle(inj)
        rng.shuffle(ben)
        n = len(in_dist)

        cal_n = int(n * self.ratios["cal"])
        test_n = int(n * self.ratios["test_in_dist"])

        # --- calibration: difficulty-weighted draw across BOTH classes ---------
        cal = self._weighted_draw(rng, inj + ben, cal_n)
        cal_ids = {id(s) for s in cal}
        inj = [s for s in inj if id(s) not in cal_ids]
        ben = [s for s in ben if id(s) not in cal_ids]
        out["cal"] = cal

        # --- test_in_dist: realistic imbalance (few positives) -----------------
        t_pos = max(1, int(test_n * self.test_inj_ratio))
        t_neg = max(1, test_n - t_pos)
        test = inj[:t_pos] + ben[:t_neg]
        rng.shuffle(test)
        out["test_in_dist"] = test
        inj = inj[t_pos:]
        ben = ben[t_neg:]

        # --- train: class-balanced by undersampling the MAJORITY class ---------
        # Hit train_inj_ratio exactly given the two pools, undersampling whichever
        # class is in surplus (positives here, since benign is the scarcer class).
        r = self.train_inj_ratio
        n_pos, n_neg = len(inj), len(ben)
        if 0 < r < 1:
            # max positives supportable by the benign pool, and vice-versa
            pos_cap = min(n_pos, int(round(n_neg * r / (1 - r))))
            neg_cap = int(round(pos_cap * (1 - r) / r))
        else:
            pos_cap, neg_cap = n_pos, n_neg
        train = inj[:pos_cap] + ben[:neg_cap]
        rng.shuffle(train)
        out["train"] = train
        log.info("train balance: %d pos / %d neg (target inj ratio %.2f)",
                 pos_cap, neg_cap, r)

    def _weighted_draw(self, rng, items, k):
        if k <= 0 or not items:
            return []
        k = min(k, len(items))
        pool = list(items)
        weights = [self.diff_weights.get(s.difficulty, 1.0) for s in pool]
        chosen, chosen_ids = [], set()
        # weighted sampling without replacement
        while len(chosen) < k and pool:
            total = sum(weights)
            r = rng.uniform(0, total)
            upto = 0.0
            for i, w in enumerate(weights):
                upto += w
                if upto >= r:
                    s = pool.pop(i)
                    weights.pop(i)
                    if id(s) not in chosen_ids:
                        chosen.append(s)
                        chosen_ids.add(id(s))
                    break
        return chosen
