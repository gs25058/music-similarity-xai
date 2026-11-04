"""Full-song analysis and cross-song window similarity with monotone chains."""
import numpy as np

from .audio import load_audio, windows
from .features import ALL_FACTORS


def analyze_song(path, backend, window=10.0, hop=5.0, structure=None):
    audio = load_audio(path)
    encoded = []
    for chunk in windows(audio, window, hop):
        embeddings, info = backend.encode(chunk)
        normalized, zero_factors = {}, []
        for factor, vector in embeddings.items():
            vector = np.asarray(vector, dtype=np.float32)
            if not np.isfinite(vector).all():
                raise FloatingPointError(f"Nonfinite {backend.name}/{factor} embedding")
            norm = float(np.linalg.norm(vector))
            if norm < 1e-8:
                zero_factors.append(factor)
            else:
                normalized[factor] = vector/norm
        embeddings = normalized
        if zero_factors:
            info["unavailable_zero_embedding_factors"] = zero_factors
        encoded.append(dict(**chunk.metadata(), embeddings=embeddings, music_information=info))
    sections = structure(audio) if structure else dict(status="unavailable", backend="none", segments=[],
                                                        reason="SongFormer not requested")
    return dict(path=audio.path, duration=audio.duration, sample_rate=audio.sample_rate,
                windows=encoded, structure=sections)


def _contexts(song, interval):
    return [s for s in song["structure"].get("segments", [])
            if s["start"] < interval["end"] and s["end"] > interval["start"]]


def compare_songs(a, b, backend, threshold=.8, top_k=20, weights=None, min_coverage=.5):
    """Cosine for all window pairs; grouping never uses verse/chorus labels.

    Kept cells are bounded per A-window. Monotone chains support diagonal,
    1:2 and 2:1 steps for modest tempo changes; no unconstrained rectangles.
    Scores remain raw cosines in [-1,1], not calibrated probabilities.
    """
    if not -1 <= threshold <= 1 or top_k < 1:
        raise ValueError("Require cosine threshold in [-1,1] and top_k >= 1")
    weights = weights if weights is not None else {factor: (0. if factor == "timbre" else 1.) for factor in backend.supported}
    if not weights or any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Need positive, nonnegative factor weights")
    if any(key not in backend.supported for key in weights):
        raise ValueError("Weights include factors unavailable in this backend")
    scores, masks = {}, {}
    for factor in backend.supported:
        dimension = next((len(w["embeddings"][factor]) for song in (a, b) for w in song["windows"]
                          if factor in w["embeddings"]), None)
        if dimension is None:
            continue
        mats, valid = [], []
        for song in (a, b):
            mats.append(np.stack([w["embeddings"].get(factor, np.zeros(dimension)) for w in song["windows"]]))
            valid.append(np.array([factor in w["embeddings"] for w in song["windows"]]))
        # Full matrix intentionally retained for transparent small/medium song evaluation.
        scores[factor] = np.clip(mats[0] @ mats[1].T, -1, 1)
        masks[factor] = valid[0][:, None] & valid[1][None, :]
    shape = (len(a["windows"]), len(b["windows"]))
    numerator, denominator = np.zeros(shape), np.zeros(shape)
    for factor, score in scores.items():
        weight = weights.get(factor, 0.)
        numerator += score * masks[factor]*weight
        denominator += masks[factor]*weight
    combined = np.divide(numerator, denominator, out=np.full(shape, -2.), where=denominator>0)
    cells = {}
    for i, row in enumerate(combined):
        for j in np.argsort(-row, kind="stable")[:min(5, len(row))]:
            wa, wb = a["windows"][i], b["windows"][j]
            coverage = min(wa["actual_duration"]/min(wa["padded_duration"], a["duration"]),
                           wb["actual_duration"]/min(wb["padded_duration"], b["duration"]))
            if row[j] >= threshold and coverage >= min_coverage:
                cells[(i, int(j))] = float(row[j])
    # Greedy best chains in a directed acyclic graph. Each cell used once.
    chains = []
    while cells and len(chains) < top_k:
        paths, values = {}, {}
        for (i, j), value in sorted(cells.items()):
            predecessors = [(i-di, j-dj) for di, dj in ((1, 1), (1, 2), (2, 1))]
            previous = max((p for p in predecessors if p in paths), key=lambda p: values[p], default=None)
            paths[(i, j)] = (paths[previous] if previous else [])+[(i, j)]
            values[(i, j)] = value + (values[previous] if previous else 0)
        end = max(values, key=values.get)
        chain = paths[end]
        chains.append((chain, np.mean([cells[c] for c in chain])))
        for cell in chain:
            del cells[cell]
    # Keep the strongest local window as well as its longer temporal chain.
    # A chain can span a transition; its union must not erase a precise match.
    candidates = []
    for chain, score in chains:
        peak = max(chain, key=lambda cell: combined[cell])
        candidates.append(([peak], float(combined[peak]), "peak_window"))
        if len(chain) > 1:
            candidates.append((chain, float(score), "monotone_chain"))
    candidates.sort(key=lambda c: (-c[1], c[0][0], 0 if c[2] == "peak_window" else 1))
    evidence = []
    for index, (chain, score, candidate_kind) in enumerate(candidates[:top_k]):
        a_windows = [a["windows"][i] for i, j in chain]
        b_windows = [b["windows"][j] for i, j in chain]
        ai = dict(start=min(w["start"] for w in a_windows), end=max(w["end"] for w in a_windows))
        bi = dict(start=min(w["start"] for w in b_windows), end=max(w["end"] for w in b_windows))
        factor_scores = {}
        for factor in ALL_FACTORS:
            values = [float(scores[factor][i, j]) for i, j in chain if factor in scores and masks[factor][i, j]]
            factor_scores[factor] = dict(status="available" if values else "unavailable",
                                         cosine=float(np.mean(values)) if values else None)
        evidence.append(dict(evidence_id=f"E{index+1:03}", candidate_kind=candidate_kind, a=ai, b=bi, combined_cosine=float(score),
                             factors=factor_scores, window_pairs=[dict(a=i, b=j) for i, j in chain],
                             music_information=dict(a=[w["music_information"] for w in a_windows],
                                                    b=[w["music_information"] for w in b_windows]),
                             structure_context=dict(a=_contexts(a, ai), b=_contexts(b, bi),
                                                    a_status=a["structure"]["status"], b_status=b["structure"]["status"],
                                                    role="context_only_not_similarity_evidence")))
    return dict(evidence=evidence, factor_weights=weights, threshold=threshold,
                min_window_coverage=min_coverage,
                total_window_pairs=int(np.prod(shape)), score_semantics="raw_cosine_not_probability",
                matrix_maxima={factor: float(matrix[masks[factor]].max()) if masks[factor].any() else None
                               for factor, matrix in scores.items()},
                limitations=["window_resolution_limits_boundary_precision", "thresholds_require_validation_calibration",
                             "phrase_structure_similarity_unavailable_structure_labels_are_context_only"])


def public_song(song):
    return {**song, "windows": [{k: v for k, v in w.items() if k != "embeddings"} for w in song["windows"]]}
