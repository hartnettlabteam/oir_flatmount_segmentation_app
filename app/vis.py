from typing import Dict, Optional, Tuple

import numpy as np
import cv2


COLORS = {
	"retina": (255, 255, 255),
	"nv": (255, 0, 0),
	"vo": (255, 255, 0),
}


def overlay_mask(gray: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.5) -> np.ndarray:
	"""Blend a binary mask onto grayscale image with low memory overhead."""
	rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
	mask_bool = mask.astype(bool)
	if not np.any(mask_bool):
		return rgb

	out = rgb.copy()
	inv_alpha = 1.0 - float(alpha)
	for ch, ch_color in enumerate(color):
		channel = out[..., ch]
		blended = (inv_alpha * channel[mask_bool] + float(alpha) * float(ch_color)).astype(np.uint8)
		channel[mask_bool] = blended
	return out


def blend_three(gray: np.ndarray, mr: np.ndarray, mnv: np.ndarray, mvo: np.ndarray) -> np.ndarray:
	img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
	img = overlay_mask(gray, mr, COLORS["retina"], alpha=0.25)
	img = overlay_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mnv, COLORS["nv"], alpha=0.5)
	img = overlay_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mvo, COLORS["vo"], alpha=0.5)
	return img


def compare_panel(
	gray: np.ndarray,
	gt: Dict[str, np.ndarray],
	pred: Dict[str, np.ndarray],
	dice: Dict[str, float],
	title: Optional[str] = None,
	figsize: Tuple[int, int] = (12, 16),
	save_path: Optional[str] = None,
):
	# Optional utility for offline analysis; keep runtime deps light for app packaging.
	matplotlib = __import__("matplotlib")
	matplotlib.use("Agg")
	plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])

	# 4 rows x 2 cols: GT on left, Pred on right
	fig, axes = plt.subplots(4, 2, figsize=figsize)

	# Retina
	axes[0, 0].imshow(overlay_mask(gray, gt["retina"], COLORS["retina"], 0.25)[..., ::-1])
	axes[0, 0].set_title("GT Retina")
	axes[0, 0].axis("off")
	axes[0, 1].imshow(overlay_mask(gray, pred["retina"], COLORS["retina"], 0.25)[..., ::-1])
	axes[0, 1].set_title(f"Pred Retina (Dice {dice['retina']:.3f})")
	axes[0, 1].axis("off")

	# NV
	axes[1, 0].imshow(overlay_mask(gray, gt["nv"], COLORS["nv"], 0.5)[..., ::-1])
	axes[1, 0].set_title("GT NV")
	axes[1, 0].axis("off")
	axes[1, 1].imshow(overlay_mask(gray, pred["nv"], COLORS["nv"], 0.5)[..., ::-1])
	axes[1, 1].set_title(f"Pred NV (Dice {dice['nv']:.3f})")
	axes[1, 1].axis("off")

	# VO
	axes[2, 0].imshow(overlay_mask(gray, gt["vo"], COLORS["vo"], 0.5)[..., ::-1])
	axes[2, 0].set_title("GT VO")
	axes[2, 0].axis("off")
	axes[2, 1].imshow(overlay_mask(gray, pred["vo"], COLORS["vo"], 0.5)[..., ::-1])
	axes[2, 1].set_title(f"Pred VO (Dice {dice['vo']:.3f})")
	axes[2, 1].axis("off")

	# All overlaid
	axes[3, 0].imshow(blend_three(gray, gt["retina"], gt["nv"], gt["vo"])[..., ::-1])
	axes[3, 0].set_title("GT All Overlaid")
	axes[3, 0].axis("off")
	axes[3, 1].imshow(blend_three(gray, pred["retina"], pred["nv"], pred["vo"])[..., ::-1])
	axes[3, 1].set_title("Pred All Overlaid")
	axes[3, 1].axis("off")

	if title:
		fig.suptitle(title)
	plt.tight_layout()
	if save_path is None:
		save_path = "comparison.png"
	plt.savefig(save_path, dpi=200)
	plt.close(fig)
