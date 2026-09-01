"""Submission-legal constants for camera-only XY offset estimation.

These duplicate the public fairness domain so the policy does not import
evaluator sampling, seeds, or shifted configs.
"""
from __future__ import annotations

XY_MIN_M = -0.01
XY_MAX_M = +0.01
SUPPORT_COUPLED_PARTS = frozenset({"gear_60teeth", "rod_16mm", "bolt_8mm"})

BUNDLE_VERSION = 1
DEFAULT_BUFFER_FRAMES = 5
# Mean-pool buffered RGB/depth into one composite before matching the
# nominal reference. Suppresses temporal render/denoiser flicker that
# otherwise makes single head frames look blurry/noisy.
AVERAGE_BUFFERED_FRAMES = True
DEFAULT_TEMPLATE_HALF_PX = 24
DEFAULT_ROI_MARGIN_PX = 6
DEPTH_DISAGREE_M = 0.008
BOARD_CONFIDENCE_MIN = 0.15
PART_NCC_MIN = 0.25
# World-XY disagreement (m) below which ECC and ORB board estimates are averaged.
BOARD_CONSENSUS_MAX_M = 0.0035
ORB_MIN_INLIERS = 12
NCC_SCALES = (1.0, 0.9, 1.1)
NUMERICAL_CLAMP_EPS_M = 5e-4
