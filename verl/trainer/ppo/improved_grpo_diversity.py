"""
This module implements enhanced diversity-aware GRPO advantage computation.

It is designed to be used in reinforcement learning fine-tuning for search‑enabled
LLMs where each sample may consist of multiple search queries and retrieved
documents across multiple rounds.  The core idea is to use log‑determinant
metrics to measure both semantic diversity (embedding space) and outcome
diversity (retrieved documents) and integrate these metrics into the reward and
advantage calculation.  Semantic diversity encourages the model to explore
different ways of phrasing a query when it is already correct, while outcome
diversity encourages exploration when the current answer is incorrect by
rewarding the retrieval of new documents.

The module provides two main functions:

* ``compute_diversity_scores_from_meta`` – extracts per‑sample embeddings and
  document IDs from ``DataProto.meta_info``, computes per‑round and per‑sample
  diversity using log‑det determinants, and returns aggregated diversity scores
  along with gate values for group‑level diversity.  It supports multi‑round
  interactions, leave‑one‑out margins, local history margins, and time
  discounting.

* ``compute_advantage`` – wraps a modified GRPO advantage calculation that
  adds an outcome diversity bonus to each sample's scalar reward and scales
  the resulting advantages by semantic diversity.  It relies on the
  ``compute_diversity_scores_from_meta`` function and seamlessly integrates
  with existing GRPO pipelines by writing the computed advantages and
  returns back to the ``DataProto`` batch.

The implementation is heavily commented to clarify the rationale and
mathematical properties of each step.  It uses numerically stable log‑det
computation via Cholesky factorisation and handles missing embeddings or
documents gracefully.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import math
import torch
import torch.nn.functional as F
from verl import DataProto  # type: ignore

# -----------------------------------------------------------------------------
# Hyperparameters
#
# These defaults control how strongly diversity influences the reward and
# advantage.  They can be overridden by passing parameters into the diversity
# functions or by reading values from a configuration object if integrated
# elsewhere.
#
# - alpha_D : multiplicative factor for the outcome diversity bonus added to
#   the scalar reward for each sample.  A positive value encourages
#   exploration when the answer is incorrect by rewarding retrieval of new
#   documents.  If set too high, it may bias learning away from correct
#   answers.
# - beta_E  : multiplicative factor for the semantic diversity scale applied
#   to the advantages.  A positive value encourages the model to produce
#   different phrasing of the query when the answer is correct.
# - zeta    : time discount factor for semantic diversity (0 < zeta ≤ 1).
#   Smaller values emphasise early rounds in multi‑round interactions.
# - eta     : time discount factor for outcome diversity (0 < eta ≤ 1).
#   Similar to ``zeta`` but applied to document diversity.
# - w_local / w_cross : weights used to blend local (within‑sample) and
#   cross‑group diversity for each round.  Local diversity measures how much
#   new information a sample adds relative to its own history, while
#   cross‑group diversity measures how much it adds relative to other samples
#   in the group.
# - s_min / s_max : bounds for clipping the semantic diversity scale applied
#   to the advantages.  Clipping prevents extremely large or small scales.
# - jitter  : small diagonal offset used in log‑det computations to ensure
#   numerical stability.
DEFAULTS: Dict[str, float] = {
    "alpha_D": 0.3,
    "beta_E": 0.4,
    "zeta": 0.5,
    "eta": 0.5,
    "w_local": 0.6,
    "w_cross": 0.4,
    "s_min": 0.8,
    "s_max": 1.3,
    "jitter": 1e-6,
}


# Low 配置：较低的多样性鼓励
LOW: Dict[str, float] = {
    "alpha_D": 0.1,    # 文档多样性影响较小
    "beta_E": 0.2,     # 语义多样性影响较小
    "zeta": 0.3,       # 强调更早的轮次，但仍较弱
    "eta": 0.3,        # 对文档多样性的时间折扣较强
    "w_local": 0.7,     # 更多的本地多样性
    "w_cross": 0.3,     # 跨组多样性较小
    "s_min": 0.5,       # 限制语义多样性的影响范围
    "s_max": 1.0,       # 语义多样性最大值较小
    "jitter": 1e-6,
}

# Medium 配置：适中的多样性鼓励
MEDIUM: Dict[str, float] = {
    "alpha_D": 0.3,    # 文档多样性影响适中
    "beta_E": 0.4,     # 语义多样性适中
    "zeta": 0.5,       # 平衡早期和后期的多样性
    "eta": 0.5,        # 文档多样性时间折扣适中
    "w_local": 0.6,     # 本地多样性和跨组多样性均衡
    "w_cross": 0.4,     # 跨组多样性适中
    "s_min": 0.7,       # 限制语义多样性影响范围
    "s_max": 1.3,       # 最大值适中
    "jitter": 1e-6,
}

# Large 配置：极高的多样性鼓励
LARGE: Dict[str, float] = {
    "alpha_D": 0.5,    # 强烈鼓励文档多样性
    "beta_E": 0.6,     # 强烈鼓励语义多样性
    "zeta": 0.7,       # 更加注重早期轮次的多样性
    "eta": 0.7,        # 强调文档多样性的时间折扣
    "w_local": 0.5,     # 本地多样性和跨组多样性平衡
    "w_cross": 0.5,     # 跨组多样性较强
    "s_min": 0.8,       # 限制语义多样性影响范围
    "s_max": 1.5,       # 语义多样性最大值较大
    "jitter": 1e-6,
}

# -----------------------------------------------------------------------------
# Utility functions
#
# The following helpers simplify common operations such as moving tensors to
# the same device, computing Gram matrices, performing log‑determinant
# calculations, and normalising scores to the [0, 1) range.  Each function is
# documented to explain its usage.

def _to_device(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Cast tensor ``t`` to the device and dtype of ``like``.

    This helper ensures that scalar additions and multiplications happen on
    the same device and with the same precision as the input rewards or
    embeddings.  Without such casting, PyTorch will raise errors or silently
    move data between CPU and GPU.
    """
    return t.to(device=like.device, dtype=like.dtype)


def _l2_normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2‑normalise each row vector of ``x`` in place.

    ``eps`` prevents division by zero when normalising zero vectors.  This
    normalisation is essential when using the dot‑product kernel to ensure
    that ``x @ x.T`` corresponds to cosine similarity.  Without normalisation,
    the kernel would capture both magnitude and direction, which is
    undesirable for measuring semantic similarity.
    """
    return F.normalize(x, p=2, dim=-1, eps=eps)


def _cosine_psd_gram(embs: torch.Tensor) -> torch.Tensor:
    """Compute a positive‑semidefinite Gram matrix using the cosine kernel.

    Each vector is first L2‑normalised and then multiplied with its
    transpose.  The result is a symmetric, PSD matrix representing pairwise
    cosine similarities.  If ``embs`` is empty, an empty matrix is returned.
    """
    if embs.numel() == 0:
        return embs.new_zeros((0, 0))
    z = _l2_normalize_rows(embs)
    return z @ z.t()


def _rbf_psd_gram(embs: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """Compute a positive‑semidefinite Gram matrix using the RBF kernel.

    The RBF (Gaussian) kernel measures similarity as ``exp(-||x - y||^2 / (2σ^2))``.
    The resulting Gram matrix is PSD by construction.  The parameter
    ``sigma`` controls the bandwidth: smaller values yield more localised
    similarity; larger values lead to broader similarity regions.
    """
    if embs.numel() == 0:
        return embs.new_zeros((0, 0))
    # Efficiently compute pairwise squared distances using broadcasting.
    sq = (embs ** 2).sum(dim=1, keepdim=True)
    dist2 = sq + sq.t() - 2.0 * (embs @ embs.t())
    return torch.exp(-dist2 / (2.0 * (sigma ** 2)))


def _logdet_I_plus_K(K: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    """Compute ``log det(I + K)`` in a numerically stable manner.

    The input ``K`` must be symmetric positive semidefinite (PSD).  We add
    ``jitter`` to the diagonal to avoid singularity, then perform a Cholesky
    decomposition ``A = L @ L.T`` where ``A = I + K + jitter*I``.  The log
    determinant is twice the sum of log diagonals of ``L``.  If ``K`` is
    empty, the result is 0 (the determinant of a 0‑dimensional identity is 1,
    whose log is 0).
    """
    n = K.size(0)
    if n == 0:
        return K.new_tensor(0.0)
    # Copy to avoid modifying the original K.  Adding to the diagonal ensures
    # ``I + K`` remains SPD even if K has zero eigenvalues.
    A = K.clone()
    A.diagonal().add_(1.0 + jitter)
    L = torch.linalg.cholesky(A)
    return 2.0 * torch.log(torch.diagonal(L)).sum()


def _leave_one_out_logdet_increment(K: torch.Tensor, i: int, jitter: float = 1e-6) -> torch.Tensor:
    """Compute the marginal log‑det increment when removing item ``i``.

    This function computes ``log det(I + K) - log det(I + K_{-i})`` by
    explicitly removing the ``i``‑th row and column from ``K`` to form
    ``K_{-i}``.  It is simple but O(n^3) in the worst case because it
    recomputes the log determinant for each removal.  For typical GRPO group
    sizes (tens of samples), this approach is acceptable.  For very large
    groups, a more efficient implementation using Schur complements would be
    preferable.
    """
    n = K.size(0)
    if n == 0:
        return K.new_tensor(0.0)
    logdet_full = _logdet_I_plus_K(K, jitter=jitter)
    if n == 1:
        # Removing the only element results in the empty set whose log det is 0.
        return logdet_full
    mask = torch.ones(n, dtype=torch.bool, device=K.device)
    mask[i] = False
    K_minus = K[mask][:, mask]
    logdet_minus = _logdet_I_plus_K(K_minus, jitter=jitter)
    return logdet_full - logdet_minus


def _doc_logdet_count(C: List[str]) -> float:
    """Compute the log determinant of ``I + K`` for document IDs under a
    Kronecker‑ID kernel.

    Under this kernel, each unique ID corresponds to a basis vector and is
    orthogonal to the others.  The Gram matrix ``K_C`` is thus diagonal with
    ones for each unique ID.  The determinant of ``I + K_C`` is ``2^{|U|}``
    where ``U`` is the set of unique IDs.  The log determinant is then
    ``|U| * log(2)``.
    """
    return math.log(2.0) * len(set(C))


def _doc_ldig_cross_round(sample_ids: List[str], C_all: List[str]) -> float:
    """Compute the log‑det increment for a sample relative to the group union.

    ``sample_ids`` is the set of IDs retrieved by one sample in a given
    round.  ``C_all`` is the union of all IDs retrieved by all samples in
    that round.  Removing ``sample_ids`` from the union reduces the number
    of unique IDs by ``|U_C| - |U_{C\Si}|``.  Each unique ID contributes
    ``log 2`` to the log‑det, so the increment is the difference in counts
    times ``log 2``.
    """
    if not sample_ids:
        return 0.0
    set_all = set(C_all)
    set_si = set(sample_ids)
    set_minus = set_all - set_si
    return math.log(2.0) * (len(set_all) - len(set_minus))


def _doc_ldig_local_round(hist_before: List[str], add_now: List[str]) -> float:
    """Compute the local log‑det increment within a sample across rounds.

    ``hist_before`` is the list of document IDs retrieved by a sample in
    previous rounds.  ``add_now`` is the list of IDs retrieved in the current
    round.  Under the Kronecker‑ID kernel, the log‑det increment equals the
    number of new unique IDs introduced by ``add_now`` times ``log 2``.
    """
    if not add_now:
        return 0.0
    unique_before = set(hist_before)
    unique_after = unique_before | set(add_now)
    return math.log(2.0) * (len(unique_after) - len(unique_before))


def _build_group_index(index: Any) -> Dict[Any, List[int]]:
    """Group sample indices by a group identifier ``index``.

    ``index`` can be any iterable (list, numpy array, etc.) where each
    element identifies which group a sample belongs to.  Samples with the
    same identifier are aggregated into the same list.
    """
    groups = defaultdict(list)
    for i, key in enumerate(index):
        groups[key].append(i)
    return groups


# -----------------------------------------------------------------------------
# Diversity score computation
#
# This function encapsulates all the logic required to transform raw per‑round
# embeddings and document IDs into normalised per‑sample diversity scores.
# It supports arbitrary group sizes and variable round counts per sample.

def compute_diversity_scores_from_meta(
    data: DataProto,
    zeta: float = DEFAULTS["zeta"],
    eta: float = DEFAULTS["eta"],
    w_local: float = DEFAULTS["w_local"],
    w_cross: float = DEFAULTS["w_cross"],
    use_rbf: bool = True,
    rbf_sigma: float = 0.5,
    jitter: float = DEFAULTS["jitter"],
) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[float], Dict[str, Any]]:
    """
    Compute semantic and document diversity scores for each sample from
    ``data.meta_info``.

    Parameters
    ----------
    data : DataProto
        Must contain ``meta_info['retrieved_id_lists_per_sample']`` and
        ``meta_info['search_query_embeddings_per_sample']``.  Each entry in
        these lists corresponds to one sample and contains per‑round data.
    zeta : float
        Discount factor for semantic diversity across rounds.  Values < 1
        emphasise early rounds.
    eta : float
        Discount factor for document diversity across rounds.
    w_local : float
        Weight for local (within‑sample) diversity.  The remaining weight
        ``(1 - w_local)`` is assigned to cross‑group diversity.
    w_cross : float
        Weight for cross‑group diversity.  Usually ``w_cross = 1 - w_local``.
    use_rbf : bool
        If True, use an RBF kernel to construct embedding Gram matrices
        instead of the cosine kernel.  RBF is more expensive but sometimes
        yields better distance sensitivity.
    rbf_sigma : float
        Bandwidth parameter for the RBF kernel when ``use_rbf=True``.
    jitter : float
        Small diagonal offset to ensure numerical stability in log‑det
        computations.

    Returns
    -------
    S_E : torch.Tensor of shape (bs,)
        Normalised semantic diversity score for each sample.  Values are in
        [0, 1), with larger values indicating greater diversity relative to
        other samples and across rounds.
    S_D : torch.Tensor of shape (bs,)
        Normalised document diversity score for each sample.  Computed
        analogously to ``S_E`` but in the space of retrieved documents.
    Gamma_E_per_sample : List[float]
        Group‑level gate values for semantic diversity, broadcast to each
        sample.  Larger values indicate the group as a whole lacks diversity.
    Gamma_D_per_sample : List[float]
        Group‑level gate values for document diversity.
    debug_info : Dict[str, Any]
        Diagnostic information including raw scores and hyperparameters.  This
        dictionary is safe to write back into ``data.meta_info``.
    """
    # breakpoint()
    # Extract meta information.  If the required keys are missing, an
    # AssertionError will be raised, signalling incorrect usage.
    mi = data.meta_info
    ids_per_sample: List[List[List[str]]] = mi.get('retrieved_id_lists_per_sample')
    embs_per_sample: List[List[Optional[List[float]]]] = mi.get('search_query_embeddings_per_sample')
    assert ids_per_sample is not None, "meta_info['retrieved_id_lists_per_sample'] is required"
    assert embs_per_sample is not None, "meta_info['search_query_embeddings_per_sample'] is required"

    bsz = len(ids_per_sample)
    assert bsz == len(embs_per_sample), "per-sample ids and embeddings length mismatch"

    # Group samples by their unique identifier (uid).  Samples with the same uid
    # belong to the same group and will be normalised together when computing
    # advantages.  The uid is stored in data.non_tensor_batch['uid'] by the
    # caller, matching the usage in compute_grpo_outcome_advantage.
    uid = data.non_tensor_batch['uid']
    groups = _build_group_index(uid)

    # Determine the maximum number of rounds across all samples.  Some samples
    # may have fewer rounds (fewer queries) than others.  We will iterate up
    # to this maximum and gracefully handle missing data for shorter samples.
    rounds_per_sample = [max(len(ids_per_sample[i]), len(embs_per_sample[i])) for i in range(bsz)]
    R_max = 0 if bsz == 0 else max(rounds_per_sample)

    # Allocate per‑sample output tensors.  These live on the same device as the
    # token-level rewards so that subsequent operations in compute_advantage
    # remain on the correct device.  If there are no samples, these remain
    # empty.
    device_like = data.batch['token_level_rewards']
    S_E = torch.zeros(bsz, dtype=device_like.dtype, device=device_like.device)
    S_D = torch.zeros(bsz, dtype=device_like.dtype, device=device_like.device)
    Gamma_E_per_sample: List[float] = [0.0] * bsz
    Gamma_D_per_sample: List[float] = [0.0] * bsz

    # Precompute per‑round mapping of embeddings and documents.  For each
    # (group_uid, r), we collect the indices of samples with non‑None
    # embeddings and stack them into a matrix E_r.  Similarly, we collect
    # document IDs into a flat list docs_all.  These are used to build the
    # group‑level Gram matrices and compute log‑determinants.
    emb_round_map: Dict[Tuple[Any, int], Tuple[List[int], torch.Tensor]] = {}
    doc_round_map: Dict[Tuple[Any, int], List[str]] = {}

    for gkey, idxs in groups.items():
        for r in range(R_max):
            emb_list: List[torch.Tensor] = []
            emb_owner: List[int] = []
            docs_all: List[str] = []
            for i in idxs:
                # Extract embedding if present for this round.  None indicates
                # that the sample did not perform a query in this round.
                if r < len(embs_per_sample[i]) and embs_per_sample[i][r] is not None:
                    e = torch.tensor(embs_per_sample[i][r], dtype=torch.float32)
                    emb_list.append(e)
                    emb_owner.append(i)
                # Extract documents for this round.  The list may be empty
                # (meaning no documents were retrieved).  We still extend
                # docs_all with an empty list to maintain shape.
                if r < len(ids_per_sample[i]) and len(ids_per_sample[i][r]) > 0:
                    docs_all.extend(ids_per_sample[i][r])
            if len(emb_list) > 0:
                E_r = torch.stack(emb_list, dim=0)
                emb_round_map[(gkey, r)] = (emb_owner, E_r)
            if len(docs_all) > 0:
                doc_round_map[(gkey, r)] = docs_all

    # Prepare per‑round containers to accumulate per‑sample diversity values.
    # These lists have length ``R_max`` and store tuples of (sample_index, score)
    # for each round.  At the end of processing, these lists will be folded
    # into final scores with time discounting.
    sE_per_round: List[List[Tuple[int, float]]] = [[] for _ in range(R_max)]
    sD_per_round: List[List[Tuple[int, float]]] = [[] for _ in range(R_max)]

    # Gate values per group: we collect the raw gate for each round and group,
    # then average (or discount) them later.  They are stored per group key.
    gate_E_round_values: Dict[Any, List[float]] = {gkey: [] for gkey in groups.keys()}
    gate_D_round_values: Dict[Any, List[float]] = {gkey: [] for gkey in groups.keys()}

    # Histories for each sample.  These accumulate embeddings and document IDs
    # across rounds to compute local log‑det increments.  For embeddings we
    # store a list of tensors; for documents we store a list of strings.
    local_hist_E: Dict[int, List[torch.Tensor]] = {i: [] for i in range(bsz)}
    local_hist_D: Dict[int, List[str]] = {i: [] for i in range(bsz)}

    # Iterate over every group and every round.  Within each iteration, we
    # compute:
    #   1. The overall group diversity (log-det) and gate values.
    #   2. The cross-group marginal contributions for each sample with data in
    #      this round.
    #   3. The local (within-sample) marginal contributions relative to the
    #      sample's own history.
    #   4. A blended per-sample score sE_{i}^{(r)} or sD_{i}^{(r)} using
    #      weights w_local and w_cross, and normalisation x/(1+x).
    for gkey, idxs in groups.items():
        for r in range(R_max):
            # --- Semantic overall diversity and gate ---
            if (gkey, r) in emb_round_map:
                owners, E_r = emb_round_map[(gkey, r)]
                # Construct Gram matrix from embeddings.  Switch between RBF
                # and cosine kernels based on ``use_rbf``.
                if use_rbf:
                    K = _rbf_psd_gram(E_r, sigma=rbf_sigma)
                else:
                    K = _cosine_psd_gram(E_r)
                K = K.to(device_like.device)
                # Compute F_E^(r) = log det(I + K).  This measures the volume
                # of the subspace spanned by the embeddings up to scale.
                F_E_r = _logdet_I_plus_K(K, jitter=jitter).item()
                # Convert the raw log-det to a gate value in [0,1].  A small
                # log-det implies low diversity, so Gamma_E is large.
                Gamma_E_r = 1.0 - (F_E_r / (1.0 + F_E_r)) if F_E_r >= 0 else 1.0
                gate_E_round_values[gkey].append(Gamma_E_r)
                # Compute cross-group marginal contributions.  For each sample
                # in this round we remove its vector from the Gram and observe
                # how much the log-det changes.  This captures how unique the
                # vector is relative to the rest of the group.
                for k_i, sample_i in enumerate(owners):
                    m_val = _leave_one_out_logdet_increment(K, k_i, jitter=jitter).item()
                    # Local (within-sample) marginal: compare current embedding
                    # to the embeddings this sample has produced in earlier rounds.
                    cur_e = E_r[k_i]
                    hist_e = local_hist_E[sample_i]
                    if not hist_e:
                        # When there is no history, the local log-det of the
                        # singleton {e} equals log det(I + [e·e^T]).  This is
                        # easily computed by the helper.
                        ell_val = _logdet_I_plus_K(cur_e.unsqueeze(0) @ cur_e.unsqueeze(0).t(), jitter=jitter).item()
                    else:
                        E_before = torch.stack(hist_e, dim=0)
                        if use_rbf:
                            K_before = _rbf_psd_gram(E_before, sigma=rbf_sigma)
                            K_after = _rbf_psd_gram(torch.cat([E_before, cur_e.unsqueeze(0)], dim=0), sigma=rbf_sigma)
                        else:
                            K_before = _cosine_psd_gram(E_before)
                            K_after = _cosine_psd_gram(torch.cat([E_before, cur_e.unsqueeze(0)], dim=0))
                        ell_val = (
                            _logdet_I_plus_K(K_after, jitter=jitter) - _logdet_I_plus_K(K_before, jitter=jitter)
                        ).item()
                    # Update history for this sample.  We append the current
                    # embedding after computing the local increment to avoid
                    # double counting in subsequent rounds.
                    local_hist_E[sample_i].append(cur_e.detach())
                    # Blend cross-group and local contributions.  Each is
                    # normalised via x/(1+x) into [0,1) so they remain on
                    # comparable scales.  The weights ``w_local`` and
                    # ``w_cross`` control their relative importance.
                    cross_norm = m_val / (1.0 + m_val) if m_val > 0 else 0.0
                    local_norm = ell_val / (1.0 + ell_val) if ell_val > 0 else 0.0
                    sE_val = w_local * local_norm + w_cross * cross_norm
                    sE_per_round[r].append((sample_i, float(sE_val)))
            # --- Document overall diversity and gate ---
            if (gkey, r) in doc_round_map:
                docs_all = doc_round_map[(gkey, r)]
                # Overall log-det for documents is simply the count of unique
                # IDs times log(2).
                F_D_r = _doc_logdet_count(docs_all)
                Gamma_D_r = 1.0 - (F_D_r / (1.0 + F_D_r)) if F_D_r >= 0 else 1.0
                gate_D_round_values[gkey].append(Gamma_D_r)
                # For each sample in the group compute cross-group and local
                # contributions.  Note that ``ids_per_sample[i][r]`` may be
                # missing or empty, so we guard accordingly.
                for i in idxs:
                    Si_r = ids_per_sample[i][r] if r < len(ids_per_sample[i]) else []
                    mD_val = _doc_ldig_cross_round(Si_r, docs_all)
                    ellD_val = _doc_ldig_local_round(local_hist_D[i], Si_r)
                    # Update history.  Documents do not require detaching.
                    local_hist_D[i].extend(Si_r)
                    # Blend contributions and normalise.  Use the same
                    # ``w_local`` and ``w_cross`` as for semantic diversity.
                    cross_norm = mD_val / (1.0 + mD_val) if mD_val > 0 else 0.0
                    local_norm = ellD_val / (1.0 + ellD_val) if ellD_val > 0 else 0.0
                    sD_val = w_local * local_norm + w_cross * cross_norm
                    sD_per_round[r].append((i, float(sD_val)))

    # Aggregate per-round scores into per-sample scores using exponential
    # discounting.  ``zeta`` controls how much recent rounds contribute to
    # semantic diversity; similarly, ``eta`` for documents.  ``denom_E`` and
    # ``denom_D`` normalise the discounted sums.  If ``R_max`` is 0 (no
    # rounds), the denominators remain 1 to avoid division by zero.
    denom_E = sum([zeta ** r for r in range(R_max)]) if R_max > 0 else 1.0
    denom_D = sum([eta ** r for r in range(R_max)]) if R_max > 0 else 1.0
    numE = torch.zeros(bsz, dtype=device_like.dtype, device=device_like.device)
    numD = torch.zeros(bsz, dtype=device_like.dtype, device=device_like.device)
    for r in range(R_max):
        for (i, sEi) in sE_per_round[r]:
            numE[i] += (zeta ** r) * torch.tensor(sEi, dtype=device_like.dtype, device=device_like.device)
        for (i, sDi) in sD_per_round[r]:
            numD[i] += (eta ** r) * torch.tensor(sDi, dtype=device_like.dtype, device=device_like.device)
    if R_max > 0:
        S_E = numE / denom_E
        S_D = numD / denom_D

    # Compute group-level gate values: average per-round gate per group, then
    # broadcast to each sample in the group.  If a group has no diversity
    # data in a particular channel, its gate remains 0.0.
    for gkey, idxs in groups.items():
        if gate_E_round_values[gkey]:
            Gamma_E_val = float(sum(gate_E_round_values[gkey]) / len(gate_E_round_values[gkey]))
        else:
            Gamma_E_val = 0.0
        if gate_D_round_values[gkey]:
            Gamma_D_val = float(sum(gate_D_round_values[gkey]) / len(gate_D_round_values[gkey]))
        else:
            Gamma_D_val = 0.0
        for i in idxs:
            Gamma_E_per_sample[i] = Gamma_E_val
            Gamma_D_per_sample[i] = Gamma_D_val

    # Bundle debug info for inspection.  These lists contain raw floats which
    # may be useful for plotting or analysing diversity contributions.  They
    # should not influence the training pipeline if not used.
    debug_info = {
        'S_E': S_E.detach().cpu().tolist(),
        'S_D': S_D.detach().cpu().tolist(),
        'Gamma_E': Gamma_E_per_sample,
        'Gamma_D': Gamma_D_per_sample,
        'params': {
            'zeta': zeta,
            'eta': eta,
            'w_local': w_local,
            'w_cross': w_cross,
            'jitter': jitter,
        }
    }
    return S_E, S_D, Gamma_E_per_sample, Gamma_D_per_sample, debug_info


# -----------------------------------------------------------------------------
# Modified GRPO outcome advantage with bonus
#
# This helper extends the original GRPO implementation by allowing a per‑sample
# reward bonus to be added before normalising within groups.  It retains the
# same interface and behaviour as ``core_algos.compute_grpo_outcome_advantage``
# when ``reward_add_per_sample`` is None, ensuring backwards compatibility.

def compute_grpo_outcome_advantage_with_bonus(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: Any,
    reward_add_per_sample: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the GRPO advantage and return values with an optional reward bonus.

    Parameters
    ----------
    token_level_rewards : torch.Tensor of shape (bs, T)
        Token-level reward tensor.  Typically only the EOS token is non‑zero
        when using outcome supervision.
    eos_mask : torch.Tensor of shape (bs, T)
        Mask indicating positions of response tokens.  Advantage is only
        applied on these positions; other positions receive zero.
    index : Any iterable of length ``bs``
        Group assignment for each sample.  Samples with the same index are
        normalised together (mean & std) when computing the advantage.
    reward_add_per_sample : Optional[torch.Tensor] of shape (bs,)
        Additional scalar reward added to each sample before group
        normalisation.  When None, no bonus is added.
    epsilon : float
        Small constant for numerical stability in division.

    Returns
    -------
    advantages : torch.Tensor of shape (bs, T)
        Normalised advantages broadcasted across the response length and
        masked by ``eos_mask``.
    returns : torch.Tensor of shape (bs, T)
        Identical to ``advantages`` for outcome‑only supervision.
    """
    response_length = token_level_rewards.shape[-1]
    # Sum token-level rewards per sample.  In outcome supervision, there is
    # exactly one non-zero reward per sample (at EOS), but we do not rely on
    # this fact in case multiple tokens carry rewards.
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)
    # Add bonus if provided.  ``reward_add_per_sample`` must be on the same
    # device and dtype as ``scores``; we rely on _to_device in caller.
    if reward_add_per_sample is not None:
        scores = scores + reward_add_per_sample
    # Group scores by uid.  Compute per‑group mean and standard deviation.
    id2score = defaultdict(list)
    for i, key in enumerate(index):
        id2score[key].append(scores[i].detach().cpu())
    id2mean: Dict[Any, torch.Tensor] = {}
    id2std: Dict[Any, torch.Tensor] = {}
    for key, vals in id2score.items():
        tvals = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else torch.tensor(vals)
        if tvals.numel() == 1:
            # Only one sample in group: mean 0, std 1 to avoid division by zero.
            id2mean[key] = torch.tensor(0.0)
            id2std[key] = torch.tensor(1.0)
        else:
            id2mean[key] = tvals.mean()
            # Unbiased=False gives the population std; clamp ensures non-zero.
            id2std[key] = tvals.std(unbiased=False).clamp_min(1e-8)
    # Standardise scores per sample according to its group.
    scores_std = scores.clone()
    for i, key in enumerate(index):
        mu = id2mean[key].to(scores_std.device, scores_std.dtype)
        sigma = id2std[key].to(scores_std.device, scores_std.dtype)
        scores_std[i] = (scores_std[i] - mu) / (sigma + epsilon)
    # Broadcast standardised scores to the sequence dimension.  Only response
    # positions (identified by ``eos_mask``) carry the advantage; other
    # positions remain zero.
    scores_std = scores_std.unsqueeze(-1).expand(-1, response_length) * eos_mask
    return scores_std, scores_std


# -----------------------------------------------------------------------------
# Top-level function integrating diversity into advantage computation
#
# This wrapper replaces the original ``compute_advantage`` function.  It first
# computes semantic and document diversity scores, uses them to adjust the
# reward and scale the advantages, and finally writes the results back to the
# ``DataProto`` object.  Additional debug information is stored in
# ``data.meta_info`` for analysis.

def compute_advantage_diverse(
    data: DataProto,
    adv_estimator: str,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
) -> DataProto:
    # breakpoint()
    """
    Compute advantages for GRPO with diversity adjustments.

    This function wraps the standard GRPO advantage computation by adding
    document diversity bonuses to the scalar rewards and scaling the
    resulting advantages by semantic diversity.  The diversity scores are
    extracted from ``data.meta_info``.  It writes back the computed
    advantages and returns to ``data.batch`` and stores intermediate values
    in ``data.meta_info`` for debugging.

    Parameters
    ----------
    data : DataProto
        Contains ``batch`` and ``meta_info``.  ``meta_info`` must include
        per-sample lists of document IDs and embeddings.
    adv_estimator : str
        Placeholder for API compatibility; ignored by this implementation.
    gamma : float
        Unused here; included for signature compatibility.
    lam : float
        Unused here; included for signature compatibility.
    num_repeat : int
        Unused here; included for signature compatibility.

    Returns
    -------
    DataProto
        The input ``data`` with updated ``advantages`` and ``returns`` in
        ``data.batch``.  Also carries additional debug info in
        ``data.meta_info``.
    """
    token_level_rewards = data.batch['token_level_rewards']
    index = data.non_tensor_batch['uid']
    responses = data.batch['responses']
    response_length = responses.size(-1)
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]
    # 1. Compute semantic and document diversity scores and gates.
    S_E, S_D, Gamma_E_list, Gamma_D_list, dbg = compute_diversity_scores_from_meta(data)
    # 2. Construct the reward bonus per sample for outcome diversity.  We
    # multiply by alpha_D and the group-level gate Gamma_D.  Higher Gamma_D
    # indicates the group lacks diversity, thus amplifying the bonus.
    Gamma_D = torch.tensor(Gamma_D_list, dtype=S_D.dtype, device=S_D.device)
    reward_add = DEFAULTS['alpha_D'] * Gamma_D * S_D
    #
    # # DRA-GRPO版本，注意，无法对整个输出计算embedding，用我们的代替
    # reward_add = DEFAULTS['beta_E'] * S_E
    # # 此外，不调整adv的缩放
    # 3. Compute GRPO advantages and returns with the reward bonus.  We pass
    # ``reward_add`` to the helper to add it before group normalisation.
    advantages, returns = compute_grpo_outcome_advantage_with_bonus(
        token_level_rewards=token_level_rewards,
        eos_mask=response_mask,
        index=index,
        reward_add_per_sample=reward_add,
        epsilon=1e-6,
    )
    # 4. Scale advantages by semantic diversity.  We clip the scale into
    # [s_min, s_max] to avoid extreme updates.  We use the group-level
    # Gamma_E to increase the effect when the group is homogeneous.
    Gamma_E = torch.tensor(Gamma_E_list, dtype=S_E.dtype, device=S_E.device)
    scale = 1.0 + DEFAULTS['beta_E'] * Gamma_E * S_E
    scale = torch.clamp(scale, min=DEFAULTS['s_min'], max=DEFAULTS['s_max'])
    # Expand to match advantage tensor shape and apply elementwise.
    scale_time = scale.unsqueeze(-1).expand_as(advantages)
    # advantages = advantages * scale_time
    advantages = torch.where(advantages < 0, advantages, advantages * scale_time)

    # 5. Write results back into the batch.  In outcome-only settings, returns
    # equal advantages.  If desired, compute returns separately for other
    # advantage estimators.
    data.batch['advantages'] = advantages
    data.batch['returns'] = returns
    # 6. Save diagnostic info for analysis.  These fields are optional and
    # should not interfere with training if left unused.
    data.meta_info['diversity_semantic_S_E'] = dbg['S_E']
    data.meta_info['diversity_document_S_D'] = dbg['S_D']
    data.meta_info['diversity_gamma_E'] = dbg['Gamma_E']
    data.meta_info['diversity_gamma_D'] = dbg['Gamma_D']
    data.meta_info['diversity_reward_add'] = reward_add.detach().cpu().tolist()
    data.meta_info['diversity_semantic_scale'] = scale.detach().cpu().tolist()
    data.meta_info['diversity_params'] = dbg['params']
    return data
