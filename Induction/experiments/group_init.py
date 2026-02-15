import torch
import torch.nn.functional as F
from tqdm import tqdm

def compute_pairwise_distances(x, y):
    x_norm = (x**2).sum(1, keepdim=True)
    y_norm = (y**2).sum(1, keepdim=True).t()
    return x_norm + y_norm - 2 * (x @ y.t())

def median_heuristic(x, y):
    dists = compute_pairwise_distances(x, y)
    n = dists.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=dists.device)
    return dists[mask].median()

def rbf_kernel(x, y, bandwidth):
    dists = compute_pairwise_distances(x, y)
    k = torch.exp(-dists / (2 * bandwidth))
    return k

@torch.no_grad()
def compute_mmd2(X, Y, K_XX_term, bandwidth, device=None):
    K_YY = rbf_kernel(Y, Y, bandwidth)  # (m, m)
    sum_K_YY = K_YY.sum() - K_YY.diagonal().sum()
    K_YY_term = sum_K_YY / (Y.size(0) * (Y.size(0) - 1))
    K_XY_term = rbf_kernel(X, Y, bandwidth)  # (n, m)

    mmd2_val = K_XX_term + K_YY_term - 2 * K_XY_term.mean()
    return torch.clamp(mmd2_val, min=0)

def greedy_group_selection(groups, instructions, X, N_init):
    groups = [group.to(torch.float32) for group in groups]
    X = X.to(torch.float32)
    N_groups = len(groups)
    bandwidth = median_heuristic(X, X)

    K_XX = rbf_kernel(X, X, bandwidth)
    K_XX = K_XX.sum() - K_XX.diagonal().sum()
    K_XX_term = K_XX / (X.size(0) * (X.size(0) - 1))

    selected_instructions = []
    remaining_idx = list(range(N_groups))
    union_selected = None
    for _ in tqdm(range(N_init), desc="Group selection"):
        remaining_sizes = torch.tensor([len(groups[j]) for j in remaining_idx], dtype=X.dtype, device=X.device)
        size_scores_remaining = remaining_sizes / remaining_sizes.max()        
        candidate_mmd = dict()
        for j in remaining_idx:
            Y = groups[j] if union_selected is None else torch.cat([union_selected, groups[j]], dim=0)
            candidate_mmd[j] = compute_mmd2(X, Y, K_XX_term, bandwidth, X.device)

        max_mmd = max(candidate_mmd.values())

        best_score = -float('inf')
        best_j = None

        for j in remaining_idx:
            spread_score = 1 - (candidate_mmd[j] / max_mmd) if max_mmd > 0 else 0.0
            idx_in_remaining = remaining_idx.index(j)
            total_score = size_scores_remaining[idx_in_remaining] + spread_score            
            if total_score > best_score:
                best_score = total_score
                best_j = j

        selected_instructions.append(instructions[best_j])
        union_selected = groups[best_j] if union_selected is None else torch.cat([union_selected, groups[best_j]], dim=0)
        remaining_idx.remove(best_j)

    return selected_instructions