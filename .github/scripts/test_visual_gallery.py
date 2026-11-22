#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "numpy==2.2.4",
#   "pillow==11.1.0",
#   "pyyaml==6.0.2",
# ]
# ///
"""Unit and smoke tests for `visual_gallery.py` and workflow YAML definitions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

import visual_gallery as vg


def _write_test_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGBA").save(path)


def _patterned_rgba(width: int = 48, height: int = 32, shift: int = 0) -> np.ndarray:
    y, x = np.meshgrid(
        np.arange(height, dtype=np.int32),
        np.arange(width, dtype=np.int32),
        indexing="ij",
    )
    r = ((x * 5 + shift) % 256).astype(np.uint8)
    g = ((y * 7 + shift) % 256).astype(np.uint8)
    b = (((x + y) * 3) % 256).astype(np.uint8)
    a = np.full((height, width), 255, dtype=np.uint8)
    return np.stack([r, g, b, a], axis=-1)


class VisualGalleryTests(unittest.TestCase):
    def test_identical_images_have_zero_flip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "frame.png"
            _write_test_png(p, _patterned_rgba())
            _, premul, non_blank = vg.load_rgba_png(p)
            self.assertTrue(non_blank)
            metrics = vg.compare_images(premul, premul)
            self.assertEqual(metrics.flip_mean, 0.0)
            self.assertEqual(metrics.flip_p95, 0.0)
            self.assertEqual(metrics.flip_max, 0.0)
            self.assertEqual(metrics.max_block_flip_mean, 0.0)
            self.assertEqual(metrics.max_local_error, 0.0)

    def test_different_images_produce_positive_flip_and_heatmap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            p_a = tmp_path / "a.png"
            p_b = tmp_path / "b.png"
            heat = tmp_path / "heat.png"
            _write_test_png(p_a, _patterned_rgba(shift=0))
            _write_test_png(p_b, _patterned_rgba(shift=64))
            _, premul_a, _ = vg.load_rgba_png(p_a)
            _, premul_b, _ = vg.load_rgba_png(p_b)
            metrics = vg.compare_images(premul_a, premul_b, heatmap_out=heat)
            self.assertGreater(metrics.flip_mean, 0.05)
            self.assertGreaterEqual(metrics.flip_p95, metrics.flip_mean)
            self.assertGreaterEqual(metrics.flip_max, metrics.flip_p95)
            self.assertGreater(metrics.max_local_error, 0.01)
            self.assertTrue(heat.is_file())
            with Image.open(heat) as hm:
                self.assertEqual(hm.size, (48, 32))
                self.assertEqual(hm.mode, "RGB")

    def test_blank_and_uniform_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            transparent = tmp_path / "transparent.png"
            uniform = tmp_path / "uniform.png"
            _write_test_png(transparent, np.zeros((16, 16, 4), dtype=np.uint8))
            _write_test_png(uniform, np.full((16, 16, 4), 128, dtype=np.uint8))
            _, _, nb_trans = vg.load_rgba_png(transparent)
            _, _, nb_unif = vg.load_rgba_png(uniform)
            self.assertFalse(nb_trans)
            self.assertFalse(nb_unif)

    def test_end_to_end_gallery_report_and_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gpu_dir = root / "in-gpu"
            cpu_dir = root / "in-cpu"
            base_dir = root / "in-base"
            out_dir = root / "gallery-out"
            summary_md = root / "summary.md"

            img = _patterned_rgba(32, 24, shift=0)
            _write_test_png(gpu_dir / "controls/button" / "primary" / "00_initial.png", img)
            _write_test_png(cpu_dir / "controls/button" / "primary" / "00_initial.png", img)
            _write_test_png(base_dir / "controls/button" / "primary" / "00_initial.png", img)

            report = vg.evaluate_gallery(
                gpu_dir=gpu_dir,
                cpu_dir=cpu_dir,
                baseline_dir=base_dir,
                output_dir=out_dir,
                require_baseline=True,
                repository="water-rs/hydrolysis-m3",
                commit_sha="deadbeef",
                summary_file=summary_md,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["summary"]["suites"], 1)
            self.assertEqual(report["summary"]["cases"], 1)
            self.assertEqual(report["summary"]["stages"], 1)
            self.assertEqual(report["summary"]["passed"], 1)
            self.assertEqual(report["summary"]["failed"], 0)

            self.assertTrue((out_dir / "report.json").is_file())
            self.assertTrue((out_dir / "index.html").is_file())
            self.assertTrue(
                (out_dir / "diff-cpu/controls/button/primary/00_initial.png").is_file()
            )
            self.assertTrue(
                (out_dir / "diff-baseline/controls/button/primary/00_initial.png").is_file()
            )
            loaded = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["repository"], "water-rs/hydrolysis-m3")
            html = (out_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("water-rs/hydrolysis-m3", html)
            self.assertNotIn("/*__GALLERY_REPORT_JSON__*/", html)
            md = summary_md.read_text(encoding="utf-8")
            self.assertIn("Visual Conformance & Showcase Gallery — PASS", md)
            self.assertIn("`controls/button`", md)

    def test_workflow_yaml_files_are_valid(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        wf_dir = repo_root / ".github" / "workflows"
        visual_wf = wf_dir / "visual-gallery.yml"
        self.assertTrue(visual_wf.is_file(), f"missing {visual_wf}")
        for yml_path in sorted(wf_dir.glob("*.yml")):
            parsed = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
            self.assertIsInstance(parsed, dict)
            self.assertIn("jobs", parsed)


if __name__ == "__main__":
    unittest.main()
