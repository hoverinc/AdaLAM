from typing import Tuple
import math

import torch

from .ransac import ransac
from .utils import dist_matrix, orientation_diff

@torch.jit.script
def select_seeds(dist1: torch.Tensor, R1: float, scores1: torch.Tensor, n1: int, fnn12: torch.Tensor, mnn: torch.Tensor):
    im1neighmap = dist1 < R1 ** 2  # (n1, n1)
    # find out who scores higher than whom
    im1scorescomp = scores1.unsqueeze(1) > scores1.unsqueeze(0)  # (n1, n1)
    # find out who scores higher than all of its neighbors: seed points
    if mnn is not None:
        im1bs = (~ torch.any(im1neighmap & im1scorescomp & mnn.unsqueeze(0), dim=1)) & mnn & (scores1 < 0.8 ** 2) # (n1,)
    else:
        im1bs =(~ torch.any(im1neighmap & im1scorescomp, dim=1)) & (scores1 < 0.8 ** 2)

    # collect all seeds in both images and the 1NN of the seeds of the other image
    im1seeds = torch.where(im1bs)[0]  # (n1bs) index format
    im2seeds = fnn12[im1bs]  # (n1bs) index format
    return im1seeds, im2seeds

@torch.jit.script
def extract_neighborhood_sets(dist1: torch.Tensor, im1seeds: torch.Tensor, im2seeds: torch.Tensor,
                              k1: torch.Tensor, k2: torch.Tensor, R1: float, R2: float, fnn12: torch.Tensor,
                              SEARCH_EXP: int, MIN_INLIERS: int):
    dst1 = dist1[im1seeds, :]
    #TODO: testing for an error that only shows up on-device
    dst2 = dist_matrix(k2[fnn12[im1seeds]].float(), k2[fnn12].float())
    local_neighs_mask = (dst1 < (SEARCH_EXP * R1) ** 2) \
                        & (dst2 < (SEARCH_EXP * R2) ** 2)

    #if ORIENTATION_THR is not None and ORIENTATION_THR < 180:
    #    relo = orientation_diff(o1, o2[fnn12])
    #    orientation_diffs = torch.abs(orientation_diff(relo.unsqueeze(0), relo[im1seeds].unsqueeze(1)))
    #    local_neighs_mask = local_neighs_mask & (orientation_diffs < ORIENTATION_THR)
    #if SCALE_RATE_THR is not None and SCALE_RATE_THR < 10:
    #    rels = s2[fnn12] / s1
    #    scale_rates = rels[im1seeds].unsqueeze(1) / rels.unsqueeze(0)
    #    local_neighs_mask = local_neighs_mask & (scale_rates < SCALE_RATE_THR) \
    #                        & (scale_rates > 1 / SCALE_RATE_THR)  # (ns, n1)

    numn1 = torch.sum(local_neighs_mask, dim=1)
    valid_seeds = numn1 >= MIN_INLIERS

    local_neighs_mask = local_neighs_mask[valid_seeds, :]

    rdims = numn1[valid_seeds]

    return local_neighs_mask, rdims, im1seeds[valid_seeds], im2seeds[valid_seeds]


@torch.jit.script
def extract_local_patterns(fnn12, fnn_to_seed_local_consistency_map_corr, k1, k2, im1seeds, im2seeds, scores):
    ransidx, tokp1 = torch.where(fnn_to_seed_local_consistency_map_corr)
    tokp2 = fnn12[tokp1]

    im1abspattern = k1[tokp1]
    im2abspattern = k2[tokp2]

    im1loc = im1abspattern - k1[im1seeds[ransidx]]
    im2loc = im2abspattern - k2[im2seeds[ransidx]]

    expanded_local_scores = scores[tokp1] + ransidx.float() #.type(scores.dtype)

    sorting_perm = torch.argsort(expanded_local_scores)

    im1loc = im1loc[sorting_perm]
    im2loc = im2loc[sorting_perm]
    tokp1 = tokp1[sorting_perm]
    tokp2 = tokp2[sorting_perm]

    return im1loc, im2loc, ransidx, tokp1, tokp2

@torch.jit.script
def adalam_core(k1 : torch.Tensor, k2 : torch.Tensor, fnn12 : torch.Tensor,
                scores1 : torch.Tensor, mnn : torch.Tensor,
                im1shape : Tuple[int, int], im2shape : Tuple[int, int],
                AREA_RATIO: int,
                SEARCH_EXP: int,
                RANSAC_ITERS: int,
                MIN_INLIERS: int,
                MIN_CONF: int,
                REFIT: bool,
                DET_THR: int,
                MIN_CONFIDENCE: int):

    #if im1shape is None:
    #    k1mins, _ = torch.min(k1, dim=0)
    #    k1maxs, _ = torch.max(k1, dim=0)
    #    im1shape = (k1maxs - k1mins).cpu().numpy()
    #if im2shape is None:
    #    k2mins, _ = torch.min(k2, dim=0)
    #    k2maxs, _ = torch.max(k2, dim=0)
    #    im2shape = (k2maxs - k2mins).cpu().numpy()

    R1 = math.sqrt((im1shape[0] * im1shape[1]) / AREA_RATIO / 3.14159265)
    R2 = math.sqrt((im2shape[0] * im2shape[1]) / AREA_RATIO / 3.14159265)

    n1 = k1.shape[0]
    n2 = k2.shape[0]

    #TODO: testing for an error that only shows up on-device
    dist1 = dist_matrix(k1.float(), k1.float())
    im1seeds, im2seeds = select_seeds(dist1, R1, scores1, n1, fnn12, mnn)

    local_neighs_mask, rdims, im1seeds, im2seeds = extract_neighborhood_sets(dist1,
                                                                             im1seeds, im2seeds,
                                                                             k1, k2, R1, R2, fnn12,
                                                                             SEARCH_EXP, MIN_INLIERS)

    if rdims.shape[0] == 0:
        # No seed point survived. Just output ratio-test matches. This should happen very rarely.
        absolute_im1idx = torch.where(scores1 < 0.8 ** 2)[0]
        absolute_im2idx = fnn12[absolute_im1idx]
        nonseed_matches = torch.stack([absolute_im1idx, absolute_im2idx], dim=1)
        return nonseed_matches

    im1loc, im2loc, ransidx, tokp1, tokp2 = extract_local_patterns(fnn12,
                                                                   local_neighs_mask,
                                                                   k1, k2, im1seeds,
                                                                   im2seeds, scores1)
    im1loc = im1loc / (R1 * SEARCH_EXP)
    im2loc = im2loc / (R2 * SEARCH_EXP)
    inlier_idx, _, inl_count_sign, inlier_counts = ransac(im1loc, im2loc, rdims,
                                                          DET_THR, MIN_CONFIDENCE,
                                                          iters=RANSAC_ITERS, refit=REFIT)

    ics = inl_count_sign[ransidx[inlier_idx]]
    ica = inlier_counts[ransidx[inlier_idx]].float()
    passed_inliers_mask = (ics >= (1-1/MIN_CONF)) & (ica * ics >= MIN_INLIERS)
    accepted_inliers = inlier_idx[passed_inliers_mask]

    absolute_im1idx = tokp1[accepted_inliers]
    absolute_im2idx = tokp2[accepted_inliers]

    # inlier_seeds_idx = torch.unique(ransidx[accepted_inliers])

    # absolute_im1idx = torch.cat([absolute_im1idx, im1seeds[inlier_seeds_idx]])
    # absolute_im2idx = torch.cat([absolute_im2idx, im2seeds[inlier_seeds_idx]])

    final_matches = torch.stack([absolute_im1idx, absolute_im2idx], dim=1)
    if final_matches.shape[0] > 1:
        return torch.unique(final_matches, dim=0)
    return final_matches
