#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.2.4",
#   "pillow==11.1.0",
# ]
# ///
"""Unified WaterUI visual conformance & showcase gallery evaluator.

Evaluates canonical `<suite>/<case>/<stage>.png` snapshots captured under
`WATERUI_TEST_ARTIFACTS_DIR` across GPU (Lavapipe Vulkan), optional CPU
(Cherenkov CPU), and optional committed baselines:
- Computes NVIDIA LDR-FLIP perceptual error metrics (`flip_mean`, `flip_p95`,
  `flip_max`, `max_block_flip_mean`) and 3x3 box-filtered `max_local_error`,
  matching `cherenkov-oracle`'s `metrics.rs`.
- Emits FLIP Magma error heatmap PNGs (`FLIP.h` `MapMagma` 256-entry LUT).
- Detects blank or completely uniform captures.
- Emits `report.json`, a self-contained static `index.html` rendered from
  `gallery_template.html`, and a `$GITHUB_STEP_SUMMARY` Markdown table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_PPD: float = 0.7 * (3840.0 / 0.7) * (math.pi / 180.0)
GQC: float = 0.7
GPC: float = 0.4
GPT: float = 0.95
GW: float = 0.082

A1: tuple[float, float, float] = (1.0, 1.0, 34.1)
B1: tuple[float, float, float] = (0.0047, 0.0053, 0.04)
A2: tuple[float, float, float] = (0.0, 0.0, 13.5)
B2: tuple[float, float, float] = (1.0e-5, 1.0e-5, 0.025)

INV_ILLUMINANT = np.array([1.052156925, 1.0, 0.918357670], dtype=np.float64)
ILLUMINANT = np.array([0.950428545, 1.0, 1.088900371], dtype=np.float64)

SRGB_TO_XYZ = np.array(
    [
        [0.4123907992659594, 0.357584339383878, 0.1804807884018343],
        [0.2126390058715104, 0.715168678767756, 0.0721923153607337],
        [0.0193308187155918, 0.119194779794626, 0.9505321522496607],
    ],
    dtype=np.float64,
)

P3_TO_XYZ = np.array(
    [
        [0.4865709486482162, 0.2656676931690931, 0.1982172852343625],
        [0.2289745640697488, 0.6917385218365062, 0.0792869140937450],
        [0.0, 0.0451133818589026, 1.043944368900976],
    ],
    dtype=np.float64,
)

XYZ_TO_P3 = np.linalg.inv(P3_TO_XYZ)
SRGB_TO_P3 = XYZ_TO_P3 @ SRGB_TO_XYZ

# Verbatim 256-entry MapMagma LUT from NVlabs/flip FLIP.h (BSD-3-Clause).
MAGMA_LUT = np.array(
    [
        [0.001462, 0.000466, 0.013866], [0.002258, 0.001295, 0.018331],
        [0.003279, 0.002305, 0.023708], [0.004512, 0.003490, 0.029965],
        [0.005950, 0.004843, 0.037130], [0.007588, 0.006356, 0.044973],
        [0.009426, 0.008022, 0.052844], [0.011465, 0.009828, 0.060750],
        [0.013708, 0.011771, 0.068667], [0.016156, 0.013840, 0.076603],
        [0.018815, 0.016026, 0.084584], [0.021692, 0.018320, 0.092610],
        [0.024792, 0.020715, 0.100676], [0.028123, 0.023201, 0.108787],
        [0.031696, 0.025765, 0.116965], [0.035520, 0.028397, 0.125209],
        [0.039608, 0.031090, 0.133515], [0.043830, 0.033830, 0.141886],
        [0.048062, 0.036607, 0.150327], [0.052320, 0.039407, 0.158841],
        [0.056615, 0.042160, 0.167446], [0.060949, 0.044794, 0.176129],
        [0.065330, 0.047318, 0.184892], [0.069764, 0.049726, 0.193735],
        [0.074257, 0.052017, 0.202660], [0.078815, 0.054184, 0.211667],
        [0.083446, 0.056225, 0.220755], [0.088155, 0.058133, 0.229922],
        [0.092949, 0.059904, 0.239164], [0.097833, 0.061531, 0.248477],
        [0.102815, 0.063010, 0.257854], [0.107899, 0.064335, 0.267289],
        [0.113094, 0.065492, 0.276784], [0.118405, 0.066479, 0.286321],
        [0.123833, 0.067295, 0.295879], [0.129380, 0.067935, 0.305443],
        [0.135053, 0.068391, 0.315000], [0.140858, 0.068654, 0.324538],
        [0.146785, 0.068738, 0.334011], [0.152839, 0.068637, 0.343404],
        [0.159018, 0.068354, 0.352688], [0.165308, 0.067911, 0.361816],
        [0.171713, 0.067305, 0.370771], [0.178212, 0.066576, 0.379497],
        [0.184801, 0.065732, 0.387973], [0.191460, 0.064818, 0.396152],
        [0.198177, 0.063862, 0.404009], [0.204935, 0.062907, 0.411514],
        [0.211718, 0.061992, 0.418647], [0.218512, 0.061158, 0.425392],
        [0.225302, 0.060445, 0.431742], [0.232077, 0.059889, 0.437695],
        [0.238826, 0.059517, 0.443256], [0.245543, 0.059352, 0.448436],
        [0.252220, 0.059415, 0.453248], [0.258857, 0.059706, 0.457710],
        [0.265447, 0.060237, 0.461840], [0.271994, 0.060994, 0.465660],
        [0.278493, 0.061978, 0.469190], [0.284951, 0.063168, 0.472451],
        [0.291366, 0.064553, 0.475462], [0.297740, 0.066117, 0.478243],
        [0.304081, 0.067835, 0.480812], [0.310382, 0.069702, 0.483186],
        [0.316654, 0.071690, 0.485380], [0.322899, 0.073782, 0.487408],
        [0.329114, 0.075972, 0.489287], [0.335308, 0.078236, 0.491024],
        [0.341482, 0.080564, 0.492631], [0.347636, 0.082946, 0.494121],
        [0.353773, 0.085373, 0.495501], [0.359898, 0.087831, 0.496778],
        [0.366012, 0.090314, 0.497960], [0.372116, 0.092816, 0.499053],
        [0.378211, 0.095332, 0.500067], [0.384299, 0.097855, 0.501002],
        [0.390384, 0.100379, 0.501864], [0.396467, 0.102902, 0.502658],
        [0.402548, 0.105420, 0.503386], [0.408629, 0.107930, 0.504052],
        [0.414709, 0.110431, 0.504662], [0.420791, 0.112920, 0.505215],
        [0.426877, 0.115395, 0.505714], [0.432967, 0.117855, 0.506160],
        [0.439062, 0.120298, 0.506555], [0.445163, 0.122724, 0.506901],
        [0.451271, 0.125132, 0.507198], [0.457386, 0.127522, 0.507448],
        [0.463508, 0.129893, 0.507652], [0.469640, 0.132245, 0.507809],
        [0.475780, 0.134577, 0.507921], [0.481929, 0.136891, 0.507989],
        [0.488088, 0.139186, 0.508011], [0.494258, 0.141462, 0.507988],
        [0.500438, 0.143719, 0.507920], [0.506629, 0.145958, 0.507806],
        [0.512831, 0.148179, 0.507648], [0.519045, 0.150383, 0.507443],
        [0.525270, 0.152569, 0.507192], [0.531507, 0.154739, 0.506895],
        [0.537755, 0.156894, 0.506551], [0.544015, 0.159033, 0.506159],
        [0.550287, 0.161158, 0.505719], [0.556571, 0.163269, 0.505230],
        [0.562866, 0.165368, 0.504692], [0.569172, 0.167454, 0.504105],
        [0.575490, 0.169530, 0.503466], [0.581819, 0.171596, 0.502777],
        [0.588158, 0.173652, 0.502035], [0.594508, 0.175701, 0.501241],
        [0.600868, 0.177743, 0.500394], [0.607238, 0.179779, 0.499492],
        [0.613617, 0.181811, 0.498536], [0.620005, 0.183840, 0.497524],
        [0.626401, 0.185867, 0.496456], [0.632805, 0.187893, 0.495332],
        [0.639216, 0.189921, 0.494150], [0.645633, 0.191952, 0.492910],
        [0.652056, 0.193986, 0.491611], [0.658483, 0.196027, 0.490253],
        [0.664915, 0.198075, 0.488836], [0.671349, 0.200133, 0.487358],
        [0.677786, 0.202203, 0.485819], [0.684224, 0.204286, 0.484219],
        [0.690661, 0.206384, 0.482558], [0.697098, 0.208501, 0.480835],
        [0.703532, 0.210638, 0.479049], [0.709962, 0.212797, 0.477201],
        [0.716387, 0.214982, 0.475290], [0.722805, 0.217194, 0.473316],
        [0.729216, 0.219437, 0.471279], [0.735616, 0.221713, 0.469180],
        [0.742004, 0.224025, 0.467018], [0.748378, 0.226377, 0.464794],
        [0.754737, 0.228772, 0.462509], [0.761077, 0.231214, 0.460162],
        [0.767398, 0.233705, 0.457755], [0.773695, 0.236249, 0.455289],
        [0.779968, 0.238851, 0.452765], [0.786212, 0.241514, 0.450184],
        [0.792427, 0.244242, 0.447543], [0.798608, 0.247040, 0.444848],
        [0.804752, 0.249911, 0.442102], [0.810855, 0.252861, 0.439305],
        [0.816914, 0.255895, 0.436461], [0.822926, 0.259016, 0.433573],
        [0.828886, 0.262229, 0.430644], [0.834791, 0.265540, 0.427671],
        [0.840636, 0.268953, 0.424666], [0.846416, 0.272473, 0.421631],
        [0.852126, 0.276106, 0.418573], [0.857763, 0.279857, 0.415496],
        [0.863320, 0.283729, 0.412403], [0.868793, 0.287728, 0.409303],
        [0.874176, 0.291859, 0.406205], [0.879464, 0.296125, 0.403118],
        [0.884651, 0.300530, 0.400047], [0.889731, 0.305079, 0.397002],
        [0.894700, 0.309773, 0.393995], [0.899552, 0.314616, 0.391037],
        [0.904281, 0.319610, 0.388137], [0.908884, 0.324755, 0.385308],
        [0.913354, 0.330052, 0.382563], [0.917689, 0.335500, 0.379915],
        [0.921884, 0.341098, 0.377376], [0.925937, 0.346844, 0.374959],
        [0.929845, 0.352734, 0.372677], [0.933606, 0.358764, 0.370541],
        [0.937221, 0.364929, 0.368567], [0.940687, 0.371224, 0.366762],
        [0.944006, 0.377643, 0.365136], [0.947180, 0.384178, 0.363701],
        [0.950210, 0.390820, 0.362468], [0.953099, 0.397563, 0.361438],
        [0.955849, 0.404400, 0.360619], [0.958464, 0.411324, 0.360014],
        [0.960949, 0.418323, 0.359630], [0.963310, 0.425390, 0.359469],
        [0.965549, 0.432519, 0.359529], [0.967671, 0.439703, 0.359810],
        [0.969680, 0.446936, 0.360311], [0.971582, 0.454210, 0.361030],
        [0.973381, 0.461520, 0.361965], [0.975082, 0.468861, 0.363111],
        [0.976690, 0.476226, 0.364466], [0.978210, 0.483612, 0.366025],
        [0.979645, 0.491014, 0.367783], [0.981000, 0.498428, 0.369734],
        [0.982279, 0.505851, 0.371874], [0.983485, 0.513280, 0.374198],
        [0.984622, 0.520713, 0.376698], [0.985693, 0.528148, 0.379371],
        [0.986700, 0.535582, 0.382210], [0.987646, 0.543015, 0.385210],
        [0.988533, 0.550446, 0.388365], [0.989363, 0.557873, 0.391671],
        [0.990138, 0.565296, 0.395122], [0.990871, 0.572706, 0.398714],
        [0.991558, 0.580107, 0.402441], [0.992196, 0.587502, 0.406299],
        [0.992785, 0.594891, 0.410283], [0.993326, 0.602275, 0.414390],
        [0.993834, 0.609644, 0.418613], [0.994309, 0.616999, 0.422950],
        [0.994738, 0.624350, 0.427397], [0.995122, 0.631696, 0.431951],
        [0.995480, 0.639027, 0.436607], [0.995810, 0.646344, 0.441361],
        [0.996096, 0.653659, 0.446213], [0.996341, 0.660969, 0.451160],
        [0.996580, 0.668256, 0.456192], [0.996775, 0.675541, 0.461314],
        [0.996925, 0.682828, 0.466526], [0.997077, 0.690088, 0.471811],
        [0.997186, 0.697349, 0.477182], [0.997254, 0.704611, 0.482635],
        [0.997325, 0.711848, 0.488154], [0.997351, 0.719089, 0.493755],
        [0.997351, 0.726324, 0.499428], [0.997341, 0.733545, 0.505167],
        [0.997285, 0.740772, 0.510983], [0.997228, 0.747981, 0.516859],
        [0.997138, 0.755190, 0.522806], [0.997019, 0.762398, 0.528821],
        [0.996898, 0.769591, 0.534892], [0.996727, 0.776795, 0.541039],
        [0.996571, 0.783977, 0.547233], [0.996369, 0.791167, 0.553499],
        [0.996162, 0.798348, 0.559820], [0.995932, 0.805527, 0.566202],
        [0.995680, 0.812706, 0.572645], [0.995424, 0.819875, 0.579140],
        [0.995131, 0.827052, 0.585701], [0.994851, 0.834213, 0.592307],
        [0.994524, 0.841387, 0.598983], [0.994222, 0.848540, 0.605696],
        [0.993866, 0.855711, 0.612482], [0.993545, 0.862859, 0.619299],
        [0.993170, 0.870024, 0.626189], [0.992831, 0.877168, 0.633109],
        [0.992440, 0.884330, 0.640099], [0.992089, 0.891470, 0.647116],
        [0.991688, 0.898627, 0.654202], [0.991332, 0.905763, 0.661309],
        [0.990930, 0.912915, 0.668481], [0.990570, 0.920049, 0.675675],
        [0.990175, 0.927196, 0.682926], [0.989815, 0.934329, 0.690198],
        [0.989434, 0.941470, 0.697519], [0.989077, 0.948604, 0.704863],
        [0.988717, 0.955742, 0.712242], [0.988367, 0.962878, 0.719649],
        [0.988033, 0.970012, 0.727077], [0.987691, 0.977154, 0.734536],
        [0.987387, 0.984288, 0.742002], [0.987053, 0.991438, 0.749504],
    ],
    dtype=np.float64,
)


@dataclass
class ComparisonMetrics:
    flip_mean: float
    flip_p95: float
    flip_max: float
    max_block_flip_mean: float
    max_local_error: float
    heatmap_path: str | None = None


@dataclass
class StageRecord:
    suite: str
    case: str
    stage: str
    width: int | None = None
    height: int | None = None
    gpu_path: str | None = None
    cpu_path: str | None = None
    baseline_path: str | None = None
    gpu_non_blank: bool | None = None
    cpu_non_blank: bool | None = None
    missing: bool = False
    cpu_comparison: ComparisonMetrics | None = None
    baseline_comparison: ComparisonMetrics | None = None
    passed: bool = True
    failures: list[str] = field(default_factory=list)


def srgb_to_linear(encoded: np.ndarray) -> np.ndarray:
    """Convert sRGB [0, 1] float array to linear light."""
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power((encoded + 0.055) / 1.055, 2.4),
    )


def load_rgba_png(path: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    """Load an image as RGBA8 and premultiplied linear Display P3 float64.

    Returns `(rgba8, premul_p3_rgba, is_non_blank)` where:
    - `rgba8` has shape `(H, W, 4)` uint8
    - `premul_p3_rgba` has shape `(H, W, 4)` float64 in linear Display P3
    - `is_non_blank` is True iff alpha is non-zero somewhere and pixels are not uniform
    """
    with Image.open(path) as img:
        rgba8 = np.asarray(img.convert("RGBA"), dtype=np.uint8)
    alpha_nonzero = bool(np.any(rgba8[..., 3] > 0))
    first_pixel = rgba8[0, 0]
    not_uniform = bool(np.any(rgba8 != first_pixel))
    is_non_blank = alpha_nonzero and not_uniform

    norm = rgba8.astype(np.float64) / 255.0
    lin_srgb = srgb_to_linear(norm[..., :3])
    alpha = norm[..., 3:4]
    premul_srgb = lin_srgb * alpha
    premul_p3_rgb = np.einsum("ij,...j->...i", SRGB_TO_P3, premul_srgb)
    premul_p3_rgba = np.concatenate([premul_p3_rgb, alpha], axis=-1)
    return rgba8, premul_p3_rgba, is_non_blank


def _xyz_to_ycxcz(xyz: np.ndarray) -> np.ndarray:
    scaled = xyz * INV_ILLUMINANT
    y = 116.0 * scaled[..., 1] - 16.0
    cx = 500.0 * (scaled[..., 0] - scaled[..., 1])
    cz = 200.0 * (scaled[..., 1] - scaled[..., 2])
    return np.stack([y, cx, cz], axis=-1)


def _ycxcz_to_xyz(ycxcz: np.ndarray) -> np.ndarray:
    y = (ycxcz[..., 0] + 16.0) / 116.0
    cx = ycxcz[..., 1] / 500.0
    cz = ycxcz[..., 2] / 200.0
    return np.stack(
        [
            (y + cx) * ILLUMINANT[0],
            y * ILLUMINANT[1],
            (y - cz) * ILLUMINANT[2],
        ],
        axis=-1,
    )


def _xyz_to_cielab(xyz: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    delta_cube = delta**3
    factor = 1.0 / (3.0 * delta * delta)
    term = 4.0 / 29.0
    v = xyz * INV_ILLUMINANT
    v = np.where(v > delta_cube, np.cbrt(np.maximum(v, 0.0)), factor * v + term)
    l_ch = 116.0 * v[..., 1] - 16.0
    a_ch = 500.0 * (v[..., 0] - v[..., 1])
    b_ch = 200.0 * (v[..., 1] - v[..., 2])
    return np.stack([l_ch, a_ch, b_ch], axis=-1)


def _hunt_adjust(lab: np.ndarray) -> np.ndarray:
    l_ch = lab[..., 0]
    return np.stack(
        [
            l_ch,
            0.01 * l_ch * lab[..., 1],
            0.01 * l_ch * lab[..., 2],
        ],
        axis=-1,
    )


def _hyab(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dl = np.abs(a[..., 0] - b[..., 0])
    dab = np.hypot(a[..., 1] - b[..., 1], a[..., 2] - b[..., 2])
    return dl + dab


def _gaussian(x2: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * math.sqrt(math.pi / b) * np.exp(-math.pi * math.pi * x2 / b)


def _gaussian_sqrt(x2: np.ndarray, a: float, b: float) -> np.ndarray:
    return math.sqrt(a * math.sqrt(math.pi / b)) * np.exp(-math.pi * math.pi * x2 / b)


def _spatial_filters(ppd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_b = max(*B1, *B2)
    radius = int(math.ceil(3.0 * math.sqrt(max_b / (2.0 * math.pi * math.pi)) * ppd))
    dx = 1.0 / ppd
    idx = (np.arange(2 * radius + 1, dtype=np.float64) - radius) * dx
    ix2 = idx * idx
    wy = _gaussian(ix2, A1[0], B1[0])
    wcx = _gaussian(ix2, A1[1], B1[1])
    wcz1 = _gaussian_sqrt(ix2, A1[2], B1[2])
    wcz2 = _gaussian_sqrt(ix2, A2[2], B2[2])
    wy /= np.sum(wy)
    wcx /= np.sum(wcx)
    norm_cz = 1.0 / math.hypot(float(np.sum(wcz1)), float(np.sum(wcz2)))
    wcz1 *= norm_cz
    wcz2 *= norm_cz
    return wy, wcx, wcz1, wcz2


def _feature_filter(ppd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stddev = 0.5 * GW * ppd
    radius = int(math.ceil(3.0 * stddev))
    x = np.arange(2 * radius + 1, dtype=np.float64) - radius
    g = np.exp(-x * x / (2.0 * stddev * stddev))
    dg = -x * g
    ddg = (x * x / (stddev * stddev) - 1.0) * g
    g /= np.sum(g)
    dg_pos = float(np.sum(dg[dg > 0.0]))
    dg_neg = float(-np.sum(dg[dg < 0.0]))
    ddg_pos = float(np.sum(ddg[ddg > 0.0]))
    ddg_neg = float(-np.sum(ddg[ddg < 0.0]))
    dg = np.where(dg > 0.0, dg / dg_pos, np.where(dg < 0.0, dg / dg_neg, 0.0))
    ddg = np.where(ddg > 0.0, ddg / ddg_pos, np.where(ddg < 0.0, ddg / ddg_neg, 0.0))
    return g, dg, ddg


def _convolve1d_edge(channel: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """1D correlation with edge-clamped boundary padding along `axis` (0 or 1)."""
    radius = len(kernel) // 2
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (radius, radius)
    padded = np.pad(channel, pad_width, mode="edge")
    out = np.zeros_like(channel, dtype=np.float64)
    length = channel.shape[axis]
    for k, weight in enumerate(kernel):
        if axis == 1:
            out += weight * padded[:, k : k + length]
        else:
            out += weight * padded[k : k + length, :]
    return out


def _compute_cmax() -> float:
    green_xyz = P3_TO_XYZ @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    blue_xyz = P3_TO_XYZ @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    g_hunt = _hunt_adjust(_xyz_to_cielab(green_xyz))
    b_hunt = _hunt_adjust(_xyz_to_cielab(blue_xyz))
    return float(_hyab(g_hunt, b_hunt) ** GQC)


CMAX: float = _compute_cmax()
PCCMAX: float = GPC * CMAX
SPATIAL_WY, SPATIAL_WCX, SPATIAL_WCZ1, SPATIAL_WCZ2 = _spatial_filters(DEFAULT_PPD)
FEAT_G, FEAT_DG, FEAT_DDG = _feature_filter(DEFAULT_PPD)


def _prepare_flip_features(p3_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute `(filtered_hunt, edge_point)` for a `[0, 1]` linear Display P3 RGB image."""
    clamped = np.clip(p3_rgb, 0.0, 1.0)
    xyz = np.einsum("ij,...j->...i", P3_TO_XYZ, clamped)
    ycxcz = _xyz_to_ycxcz(xyz)

    sy_x = _convolve1d_edge(ycxcz[..., 0], SPATIAL_WY, axis=1)
    scx_x = _convolve1d_edge(ycxcz[..., 1], SPATIAL_WCX, axis=1)
    scz1_x = _convolve1d_edge(ycxcz[..., 2], SPATIAL_WCZ1, axis=1)
    scz2_x = _convolve1d_edge(ycxcz[..., 2], SPATIAL_WCZ2, axis=1)

    sy = _convolve1d_edge(sy_x, SPATIAL_WY, axis=0)
    scx = _convolve1d_edge(scx_x, SPATIAL_WCX, axis=0)
    scz = _convolve1d_edge(scz1_x, SPATIAL_WCZ1, axis=0) + _convolve1d_edge(
        scz2_x, SPATIAL_WCZ2, axis=0
    )

    filt_xyz = _ycxcz_to_xyz(np.stack([sy, scx, scz], axis=-1))
    filt_p3 = np.clip(np.einsum("ij,...j->...i", XYZ_TO_P3, filt_xyz), 0.0, 1.0)
    filt_lab = _xyz_to_cielab(np.einsum("ij,...j->...i", P3_TO_XYZ, filt_p3))
    hunt = _hunt_adjust(filt_lab)

    yn = ycxcz[..., 0] / 116.0 + 16.0 / 116.0
    dx_x = _convolve1d_edge(yn, FEAT_DG, axis=1)
    ddx_x = _convolve1d_edge(yn, FEAT_DDG, axis=1)
    g_x = _convolve1d_edge(yn, FEAT_G, axis=1)

    dx = _convolve1d_edge(dx_x, FEAT_G, axis=0)
    ddx = _convolve1d_edge(ddx_x, FEAT_G, axis=0)
    dy = _convolve1d_edge(g_x, FEAT_DG, axis=0)
    ddy = _convolve1d_edge(g_x, FEAT_DDG, axis=0)

    edge_point = np.stack([np.hypot(dx, dy), np.hypot(ddx, ddy)], axis=-1)
    return hunt, edge_point


def _box3x3(premul_rgba: np.ndarray) -> np.ndarray:
    """3x3 box filter over valid in-bounds neighbors (matching `metrics.rs::box3x3`)."""
    h, w, c = premul_rgba.shape
    acc = np.zeros((h, w, c), dtype=np.float64)
    counts = np.zeros((h, w, 1), dtype=np.float64)
    for dy in (-1, 0, 1):
        y_src_start = max(0, dy)
        y_src_end = h + min(0, dy)
        y_dst_start = max(0, -dy)
        y_dst_end = h - max(0, dy)
        for dx in (-1, 0, 1):
            x_src_start = max(0, dx)
            x_src_end = w + min(0, dx)
            x_dst_start = max(0, -dx)
            x_dst_end = w - max(0, dx)
            acc[y_dst_start:y_dst_end, x_dst_start:x_dst_end] += premul_rgba[
                y_src_start:y_src_end, x_src_start:x_src_end
            ]
            counts[y_dst_start:y_dst_end, x_dst_start:x_dst_end] += 1.0
    return acc / counts


def _max_block_mean(error_map: np.ndarray, block_size: int = 8) -> float:
    h, w = error_map.shape
    max_mean = 0.0
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = error_map[y : min(y + block_size, h), x : min(x + block_size, w)]
            if block.size > 0:
                mean_val = float(np.mean(block))
                if mean_val > max_mean:
                    max_mean = mean_val
    return max_mean


def compare_images(
    ref_premul_p3: np.ndarray,
    test_premul_p3: np.ndarray,
    heatmap_out: Path | None = None,
) -> ComparisonMetrics:
    """Compare two premultiplied linear Display P3 RGBA images using LDR-FLIP."""
    if ref_premul_p3.shape != test_premul_p3.shape:
        raise ValueError(
            f"dimension mismatch: reference {ref_premul_p3.shape[1]}x{ref_premul_p3.shape[0]} "
            f"vs test {test_premul_p3.shape[1]}x{test_premul_p3.shape[0]}"
        )

    ref_hunt, ref_feat = _prepare_flip_features(ref_premul_p3[..., :3])
    test_hunt, test_feat = _prepare_flip_features(test_premul_p3[..., :3])

    cd = np.power(_hyab(ref_hunt, test_hunt), GQC)
    cd = np.where(
        cd < PCCMAX,
        cd * (GPT / PCCMAX),
        ((cd - PCCMAX) / (CMAX - PCCMAX)) * (1.0 - GPT) + GPT,
    )
    edge_diff = np.abs(ref_feat[..., 0] - test_feat[..., 0])
    point_diff = np.abs(ref_feat[..., 1] - test_feat[..., 1])
    fd = np.sqrt(np.maximum(edge_diff, point_diff) * (1.0 / math.sqrt(2.0)))
    error_map = np.clip(np.power(np.maximum(cd, 0.0), 1.0 - fd), 0.0, 1.0)
    # Zero out exact floating-point noise when pixels match identically.
    exact_match = np.all(ref_premul_p3 == test_premul_p3, axis=-1)
    if np.all(exact_match):
        error_map.fill(0.0)

    flip_mean = float(np.mean(error_map)) if error_map.size else 0.0
    flip_p95 = float(np.percentile(error_map, 95.0)) if error_map.size else 0.0
    flip_max = float(np.max(error_map)) if error_map.size else 0.0
    max_block = _max_block_mean(error_map, block_size=8) if error_map.size else 0.0

    local_diff = np.abs(_box3x3(ref_premul_p3) - _box3x3(test_premul_p3))
    max_local_err = float(np.max(local_diff)) if local_diff.size else 0.0

    heatmap_rel: str | None = None
    if heatmap_out is not None:
        heatmap_out.parent.mkdir(parents=True, exist_ok=True)
        indices = np.clip(np.rint(error_map * 255.0).astype(np.int32), 0, 255)
        rgb8 = np.rint(MAGMA_LUT[indices] * 255.0).astype(np.uint8)
        Image.fromarray(rgb8, mode="RGB").save(heatmap_out)
        heatmap_rel = heatmap_out.as_posix()

    return ComparisonMetrics(
        flip_mean=flip_mean,
        flip_p95=flip_p95,
        flip_max=flip_max,
        max_block_flip_mean=max_block,
        max_local_error=max_local_err,
        heatmap_path=heatmap_rel,
    )


def discover_snapshots(root: Path | None) -> dict[tuple[str, str, str], Path]:
    """Walk `root` and return `(suite, case, stage) -> path` for all PNG files."""
    if root is None or not root.is_dir():
        return {}
    found: dict[tuple[str, str, str], Path] = {}
    for png in sorted(root.rglob("*.png")):
        rel = png.relative_to(root)
        parts = rel.parts
        if any(p.startswith(".") or p in ("diff-cpu", "diff-baseline") for p in parts):
            continue
        stage = Path(parts[-1]).stem
        if len(parts) >= 3:
            suite = "/".join(parts[:-2])
            case = parts[-2]
        elif len(parts) == 2:
            suite = "default"
            case = parts[0]
        else:
            suite = "default"
            case = "default"
        found[(suite, case, stage)] = png
    return found


def _stage_rel_path(prefix: str, key: tuple[str, str, str]) -> Path:
    suite, case, stage = key
    return Path(prefix) / suite / case / f"{stage}.png"


def _copy_into_output(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)


def format_markdown_summary(report: dict[str, Any]) -> str:
    """Format the `$GITHUB_STEP_SUMMARY` completeness and FLIP Markdown report."""
    summary = report["summary"]
    thresholds = report["thresholds"]
    status_str = "PASS" if report["passed"] else "FAIL"
    lines = [
        f"## Visual Conformance & Showcase Gallery — {status_str}",
        "",
        (
            f"- **Suites**: `{summary['suites']}` | **Cases**: `{summary['cases']}` | "
            f"**Stages**: `{summary['stages']}` | **Passed**: `{summary['passed']}` | "
            f"**Failed**: `{summary['failed']}` | **Blank**: `{summary['blank']}` | "
            f"**Missing**: `{summary['missing']}`"
        ),
        (
            f"- **FLIP Thresholds**: `mean <= {thresholds['flip_mean_max']:.4f}`, "
            f"`p95 <= {thresholds['flip_p95_max']:.4f}`"
        ),
        "",
        "| Suite | Case | Stage | Size | GPU | CPU | GPU↔CPU FLIP (mean / p95 / max) | GPU↔Base FLIP (mean / p95 / max) | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["stages"]:
        size_cell = (
            f"`{item['width']}×{item['height']}`"
            if item["width"] and item["height"]
            else "—"
        )
        gpu_cell = (
            "ok"
            if item["gpu_non_blank"] is True
            else ("BLANK" if item["gpu_non_blank"] is False else "missing")
        )
        if item["cpu_path"] is not None:
            cpu_cell = "ok" if item["cpu_non_blank"] is True else "BLANK"
        else:
            cpu_cell = "—"

        def _fmt_cmp(cmp_dict: dict[str, Any] | None) -> str:
            if not cmp_dict:
                return "—"
            return (
                f"`{cmp_dict['flip_mean']:.4f}` / "
                f"`{cmp_dict['flip_p95']:.4f}` / "
                f"`{cmp_dict['flip_max']:.4f}`"
            )

        cpu_cmp_cell = _fmt_cmp(item["cpu_comparison"])
        base_cmp_cell = _fmt_cmp(item["baseline_comparison"])
        if item["passed"]:
            status_cell = "PASS"
        else:
            reasons = "; ".join(item["failures"])
            status_cell = f"FAIL ({reasons})"
        lines.append(
            f"| `{item['suite']}` | `{item['case']}` | `{item['stage']}` | "
            f"{size_cell} | {gpu_cell} | {cpu_cell} | {cpu_cmp_cell} | {base_cmp_cell} | {status_cell} |"
        )
    lines.append("")
    return "\n".join(lines)


def evaluate_gallery(
    gpu_dir: Path,
    output_dir: Path,
    cpu_dir: Path | None = None,
    baseline_dir: Path | None = None,
    require_baseline: bool = False,
    allow_empty: bool = False,
    flip_mean_max: float = 0.02,
    flip_p95_max: float = 0.05,
    repository: str = "",
    commit_sha: str = "",
    template_path: Path | None = None,
    summary_file: Path | None = None,
) -> dict[str, Any]:
    """Evaluate all discovered snapshots, write `report.json` + `index.html`, and return the report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_map = discover_snapshots(gpu_dir)
    cpu_active = cpu_dir is not None and cpu_dir.exists()
    cpu_map = discover_snapshots(cpu_dir) if cpu_active else {}
    baseline_active = baseline_dir is not None and baseline_dir.exists()
    baseline_map = discover_snapshots(baseline_dir) if baseline_active else {}

    all_keys = sorted(set(gpu_map.keys()) | set(cpu_map.keys()) | set(baseline_map.keys()))

    stages: list[StageRecord] = []
    suites_seen: set[str] = set()
    cases_seen: set[tuple[str, str]] = set()
    blank_count = 0
    missing_count = 0

    for key in all_keys:
        suite, case, stage = key
        suites_seen.add(suite)
        cases_seen.add((suite, case))
        rec = StageRecord(suite=suite, case=case, stage=stage)

        gpu_src = gpu_map.get(key)
        cpu_src = cpu_map.get(key)
        base_src = baseline_map.get(key)

        gpu_premul: np.ndarray | None = None
        cpu_premul: np.ndarray | None = None
        base_premul: np.ndarray | None = None

        if gpu_src is not None:
            gpu_rel = _stage_rel_path("gpu", key)
            gpu_dst = output_dir / gpu_rel
            _copy_into_output(gpu_src, gpu_dst)
            rec.gpu_path = gpu_rel.as_posix()
            rgba8, gpu_premul, non_blank = load_rgba_png(gpu_dst)
            rec.height, rec.width = int(rgba8.shape[0]), int(rgba8.shape[1])
            rec.gpu_non_blank = non_blank
            if not non_blank:
                rec.passed = False
                rec.failures.append("GPU frame is blank or uniform")
                blank_count += 1
        else:
            rec.passed = False
            rec.missing = True
            rec.failures.append("missing GPU capture")
            missing_count += 1

        if cpu_active:
            if cpu_src is not None:
                cpu_rel = _stage_rel_path("cpu", key)
                cpu_dst = output_dir / cpu_rel
                _copy_into_output(cpu_src, cpu_dst)
                rec.cpu_path = cpu_rel.as_posix()
                rgba8_cpu, cpu_premul, cpu_non_blank = load_rgba_png(cpu_dst)
                if rec.width is None:
                    rec.height, rec.width = int(rgba8_cpu.shape[0]), int(rgba8_cpu.shape[1])
                rec.cpu_non_blank = cpu_non_blank
                if not cpu_non_blank:
                    rec.passed = False
                    rec.failures.append("CPU frame is blank or uniform")
                    blank_count += 1
            else:
                rec.passed = False
                rec.missing = True
                rec.failures.append("missing CPU capture")
                missing_count += 1

        if baseline_active:
            if base_src is not None:
                base_rel = _stage_rel_path("baseline", key)
                base_dst = output_dir / base_rel
                _copy_into_output(base_src, base_dst)
                rec.baseline_path = base_rel.as_posix()
                _, base_premul, _ = load_rgba_png(base_dst)
            elif require_baseline:
                rec.passed = False
                rec.missing = True
                rec.failures.append("missing required baseline PNG")
                missing_count += 1

        if gpu_premul is not None and cpu_premul is not None:
            if gpu_premul.shape != cpu_premul.shape:
                rec.passed = False
                rec.failures.append(
                    f"GPU ({gpu_premul.shape[1]}x{gpu_premul.shape[0]}) vs "
                    f"CPU ({cpu_premul.shape[1]}x{cpu_premul.shape[0]}) size mismatch"
                )
            else:
                heat_rel = _stage_rel_path("diff-cpu", key)
                cmp_metrics = compare_images(
                    ref_premul_p3=gpu_premul,
                    test_premul_p3=cpu_premul,
                    heatmap_out=output_dir / heat_rel,
                )
                cmp_metrics.heatmap_path = heat_rel.as_posix()
                rec.cpu_comparison = cmp_metrics
                if cmp_metrics.flip_mean > flip_mean_max:
                    rec.passed = False
                    rec.failures.append(
                        f"GPU↔CPU flip_mean {cmp_metrics.flip_mean:.4f} > {flip_mean_max:.4f}"
                    )
                if cmp_metrics.flip_p95 > flip_p95_max:
                    rec.passed = False
                    rec.failures.append(
                        f"GPU↔CPU flip_p95 {cmp_metrics.flip_p95:.4f} > {flip_p95_max:.4f}"
                    )

        if gpu_premul is not None and base_premul is not None:
            if gpu_premul.shape != base_premul.shape:
                rec.passed = False
                rec.failures.append(
                    f"GPU ({gpu_premul.shape[1]}x{gpu_premul.shape[0]}) vs "
                    f"Baseline ({base_premul.shape[1]}x{base_premul.shape[0]}) size mismatch"
                )
            else:
                heat_rel = _stage_rel_path("diff-baseline", key)
                cmp_metrics = compare_images(
                    ref_premul_p3=base_premul,
                    test_premul_p3=gpu_premul,
                    heatmap_out=output_dir / heat_rel,
                )
                cmp_metrics.heatmap_path = heat_rel.as_posix()
                rec.baseline_comparison = cmp_metrics
                if cmp_metrics.flip_mean > flip_mean_max:
                    rec.passed = False
                    rec.failures.append(
                        f"GPU↔Baseline flip_mean {cmp_metrics.flip_mean:.4f} > {flip_mean_max:.4f}"
                    )
                if cmp_metrics.flip_p95 > flip_p95_max:
                    rec.passed = False
                    rec.failures.append(
                        f"GPU↔Baseline flip_p95 {cmp_metrics.flip_p95:.4f} > {flip_p95_max:.4f}"
                    )

        stages.append(rec)

    passed_count = sum(1 for s in stages if s.passed)
    failed_count = len(stages) - passed_count
    overall_passed = (failed_count == 0) and (len(stages) > 0 or allow_empty)

    report: dict[str, Any] = {
        "repository": repository,
        "commit_sha": commit_sha,
        "passed": overall_passed,
        "thresholds": {
            "flip_mean_max": flip_mean_max,
            "flip_p95_max": flip_p95_max,
        },
        "summary": {
            "suites": len(suites_seen),
            "cases": len(cases_seen),
            "stages": len(stages),
            "passed": passed_count,
            "failed": failed_count,
            "blank": blank_count,
            "missing": missing_count,
        },
        "stages": [asdict(s) for s in stages],
    }

    report_json_path = output_dir / "report.json"
    serialized_json = json.dumps(report, indent=2)
    report_json_path.write_text(serialized_json + "\n", encoding="utf-8")

    resolved_template = (
        template_path
        if template_path is not None
        else Path(__file__).resolve().parent / "gallery_template.html"
    )
    template_html = resolved_template.read_text(encoding="utf-8")
    safe_embedded_json = serialized_json.replace("</", "<\\/")
    index_html = template_html.replace("/*__GALLERY_REPORT_JSON__*/", safe_embedded_json)
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")

    md_summary = format_markdown_summary(report)
    if summary_file is not None:
        with summary_file.open("a", encoding="utf-8") as handle:
            handle.write(md_summary)

    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-dir", type=Path, required=True, help="Directory of GPU snapshots")
    parser.add_argument("--cpu-dir", type=Path, default=None, help="Directory of CPU snapshots")
    parser.add_argument(
        "--baseline-dir", type=Path, default=None, help="Directory of committed baseline snapshots"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for gallery and report"
    )
    parser.add_argument(
        "--require-baseline",
        action="store_true",
        help="Fail when a stage has no matching baseline snapshot",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Succeed even when zero snapshots were captured",
    )
    parser.add_argument(
        "--flip-mean-max", type=float, default=0.02, help="Maximum allowed FLIP mean error"
    )
    parser.add_argument(
        "--flip-p95-max", type=float, default=0.05, help="Maximum allowed FLIP p95 error"
    )
    parser.add_argument(
        "--repository",
        type=str,
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="Repository name for gallery header",
    )
    parser.add_argument(
        "--commit-sha",
        type=str,
        default=os.environ.get("GITHUB_SHA", ""),
        help="Commit SHA for report metadata",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path(os.environ["GITHUB_STEP_SUMMARY"])
        if os.environ.get("GITHUB_STEP_SUMMARY")
        else None,
        help="Markdown step summary output path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_gallery(
        gpu_dir=args.gpu_dir,
        output_dir=args.output_dir,
        cpu_dir=args.cpu_dir,
        baseline_dir=args.baseline_dir,
        require_baseline=args.require_baseline,
        allow_empty=args.allow_empty,
        flip_mean_max=args.flip_mean_max,
        flip_p95_max=args.flip_p95_max,
        repository=args.repository,
        commit_sha=args.commit_sha,
        summary_file=args.summary_file,
    )
    if not report["passed"]:
        if report["summary"]["stages"] == 0 and not args.allow_empty:
            print("::error::Visual gallery captured zero snapshots.", file=sys.stderr)
        else:
            print(
                f"::error::Visual gallery failed: {report['summary']['failed']} of "
                f"{report['summary']['stages']} stage(s) failed conformance checks.",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
