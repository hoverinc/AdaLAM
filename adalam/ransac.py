from .utils import draw_first_k_couples, batch_2x2_inv, batch_2x2_ellipse, arange_sequence, piecewise_arange
import torch
from typing import Optional

@torch.jit.script
def stable_sort_residuals(residuals, ransidx):
    logres = torch.log(residuals + 1e-10)
    minlogres = torch.min(logres)
    maxlogres = torch.max(logres)

    sorting_score = ransidx.unsqueeze(0).float() + 0.99 * (logres - minlogres) / (maxlogres - minlogres)

    sorting_idxes = torch.argsort(sorting_score, dim=-1)  # (niters, numsamples)

    iters_range = torch.arange(residuals.shape[0], device=residuals.device)

    return residuals[iters_range.unsqueeze(-1), sorting_idxes], sorting_idxes

@torch.jit.script
def group_sum_and_cumsum(scores_mat, end_group_idx): #, group_idx: torch.Tensor=None):
    cumulative_scores = torch.cumsum(scores_mat, dim=1)
    ending_cumusums = cumulative_scores[:, end_group_idx]
    shifted_ending_cumusums = torch.cat(
        [torch.zeros(size=(ending_cumusums.shape[0], 1), dtype=ending_cumusums.dtype, device=scores_mat.device),
         ending_cumusums[:, :-1]], dim=1)
    grouped_sums = ending_cumusums - shifted_ending_cumusums

    return grouped_sums, cumulative_scores, shifted_ending_cumusums
    #if group_idx is not None:
    #    grouped_cumsums = cumulative_scores - shifted_ending_cumusums[:, group_idx]
    #    return grouped_sums, grouped_cumsums
    #return grouped_sums, None

@torch.jit.script
def confidence_based_inlier_selection(residuals, ransidx, rdims, idxoffsets, min_confidence: int):
    numransacs = rdims.shape[0]
    numiters = residuals.shape[0]

    sorted_res, sorting_idxes = stable_sort_residuals(residuals, ransidx)
    sorted_res_sqr = sorted_res ** 2

    too_perfect_fits = sorted_res_sqr <= 1e-8
    end_rans_indexing = torch.cumsum(rdims, dim=0)-1

    #NOTE: why the fuck is this getting converted to half and then float?
    #_, inv_indices, res_dup_counts = torch.unique_consecutive(sorted_res_sqr.half().float(), dim=1, return_counts=True, return_inverse=True)
    _, inv_indices, res_dup_counts = torch.unique_consecutive(sorted_res_sqr.float(), dim=1, return_counts=True, return_inverse=True)

    duplicates_per_sample = res_dup_counts[inv_indices]
    inlier_weights = (1./duplicates_per_sample).repeat(numiters, 1)
    inlier_weights[too_perfect_fits] = 0.

    balanced_rdims, cumsum_scores, shift_end = group_sum_and_cumsum(inlier_weights, end_rans_indexing) #, ransidx)
    weights_cumsums = cumsum_scores - shift_end[:, ransidx]
    progressive_inl_rates = weights_cumsums / (balanced_rdims.repeat_interleave(rdims, dim=1)).float()

    good_inl_mask = (sorted_res_sqr * min_confidence <= progressive_inl_rates) | too_perfect_fits

    inlier_weights[~good_inl_mask] = 0.
    inlier_counts_matrix, _, _ = group_sum_and_cumsum(inlier_weights, end_rans_indexing)

    inl_counts, inl_iters = torch.max(inlier_counts_matrix, dim=0)

    relative_inl_idxes = arange_sequence(inl_counts)
    inl_ransidx = torch.arange(numransacs).repeat_interleave(inl_counts)
    inl_sampleidx = sorting_idxes[inl_iters.repeat_interleave(inl_counts),
                                  idxoffsets[inl_ransidx] + relative_inl_idxes]
    highest_accepted_sqr_residuals = sorted_res_sqr[inl_iters, idxoffsets + inl_counts - 1]
    expected_extra_inl = balanced_rdims[inl_iters, torch.arange(numransacs)].float() * highest_accepted_sqr_residuals
    return inl_ransidx, inl_sampleidx, inl_counts, inl_iters, 1.-expected_extra_inl/inl_counts.float()

@torch.jit.script
def sample_padded_inliers(xsamples, ysamples, inlier_counts, inl_ransidx, inl_sampleidx, numransacs: int):
    maxinliers = torch.max(inlier_counts).item()
    padded_inlier_x = torch.zeros(size=(numransacs, int(maxinliers), 2))
    padded_inlier_y = torch.zeros(size=(numransacs, int(maxinliers), 2))

    padded_inlier_x[inl_ransidx, piecewise_arange(inl_ransidx)] = xsamples[inl_sampleidx]
    padded_inlier_y[inl_ransidx, piecewise_arange(inl_ransidx)] = ysamples[inl_sampleidx]

    return padded_inlier_x, padded_inlier_y

@torch.jit.script
def ransac(xsamples: torch.Tensor, ysamples: torch.Tensor, rdims: torch.Tensor, DET_THR: int, MIN_CONFIDENCE: int, iters: int=128, refit: bool=True):

    numransacs = rdims.shape[0]
    numsamples = xsamples.shape[0]
    ransidx = torch.arange(numransacs).repeat_interleave(rdims)
    idxoffsets = torch.cat([torch.tensor([0]), torch.cumsum(rdims[:-1], dim=0)], dim=0)

    rand_samples_rel = draw_first_k_couples(iters, rdims)
    rand_samples_abs = rand_samples_rel + idxoffsets
    sampled_x = torch.transpose(xsamples[rand_samples_abs], dim0=1,
                                dim1=2)  # (niters, 2, numransacs, 2) -> (niters, numransacs, 2, 2)
    sampled_y = torch.transpose(ysamples[rand_samples_abs], dim0=1, dim1=2)

    # minimal fit for sampled_x @ A^T = sampled_y
    affinities_fit = torch.transpose(batch_2x2_inv(sampled_x, check_dets=True) @ sampled_y, -1, -2)
    if not refit:
        eigenvals, eigenvecs = batch_2x2_ellipse(affinities_fit)
        bad_ones = (eigenvals[..., 1] < 1/DET_THR**2) | (eigenvals[..., 0] > DET_THR**2)
        affinities_fit[bad_ones] = torch.eye(2)
    y_pred = (affinities_fit[:, ransidx] @ xsamples.unsqueeze(-1)).squeeze(-1)

    residuals = torch.norm(y_pred - ysamples, dim=-1)  # (niters, numsamples)

    inl_ransidx, inl_sampleidx, \
    inl_counts, inl_iters, \
    comp_inl_counts = confidence_based_inlier_selection(residuals, ransidx,
                            rdims, idxoffsets, min_confidence=MIN_CONFIDENCE)

    if len(inl_sampleidx) == 0:
        # If no inliers have been found, there is nothing to re-fit!
        refit = False

    if not refit:
        return inl_sampleidx, \
               affinities_fit[inl_iters, torch.arange(inl_iters.shape[0])], \
               comp_inl_counts, inl_counts

    # Organize inliers found into a matrix for efficient GPU re-fitting.
    # Cope with the irregular number of inliers per sample by padding with zeros
    padded_inlier_x, padded_inlier_y = sample_padded_inliers(xsamples, ysamples, inl_counts, inl_ransidx, inl_sampleidx,
                                                             numransacs)

    # A @ pad_x.T = pad_y.T
    # A = pad_y.T @ pad_x @ (pad_x.T @ pad_x)^-1
    refit_affinity = padded_inlier_y.transpose(-2, -1) @ padded_inlier_x @ batch_2x2_inv(
        padded_inlier_x.transpose(-2, -1) @ padded_inlier_x, check_dets=True)

    # Filter out degenerate affinities with large scale changes
    eigenvals, eigenvecs = batch_2x2_ellipse(refit_affinity)
    bad_ones = (eigenvals[..., 1] < 1/DET_THR**2) | (eigenvals[..., 0] > DET_THR**2)
    refit_affinity[bad_ones] = torch.eye(2)
    y_pred = (refit_affinity[ransidx] @ xsamples.unsqueeze(-1)).squeeze(-1)

    residuals = torch.norm(y_pred - ysamples, dim=-1)

    inl_ransidx, inl_sampleidx, \
    inl_counts, inl_iters, comp_inl_counts = confidence_based_inlier_selection(residuals.unsqueeze(0), ransidx,
                            rdims, idxoffsets, min_confidence=MIN_CONFIDENCE)
    return inl_sampleidx, refit_affinity, comp_inl_counts, inl_counts
