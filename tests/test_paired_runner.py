import json
import tempfile
import unittest
from pathlib import Path

from cosine_filter_audit.adaptive_attack import PHASE1, PHASE2
from cosine_filter_audit.paired_runner import (
    _suite_signature,
    _validate_completed_run_artifact,
    _validate_resume_checkpoint,
    build_suite_arms,
    parse_csv_values,
)


class PairedRunnerTests(unittest.TestCase):
    def test_builds_selection_and_robust_aggregation_arms_by_attack_phase(self):
        arms = build_suite_arms(
            ["none", "cosine", "multi_signal"],
            ["krum"],
            ["none", PHASE1, PHASE2],
        )

        self.assertEqual(len(arms), 12)
        self.assertEqual(arms[0]["arm_id"], "none__none")
        phase2_krum = next(
            arm for arm in arms if arm["arm_id"] == f"{PHASE2}__krum"
        )
        self.assertEqual(phase2_krum["selection_mode"], "none")
        self.assertEqual(phase2_krum["aggregation_mode"], "krum")

    def test_rejects_unknown_or_duplicate_suite_arms(self):
        with self.assertRaisesRegex(ValueError, "Unknown suite"):
            build_suite_arms(["mystery"], [], ["none"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_csv_values("none,none", name="arms")

    def test_builds_calibration_only_selection_arms_without_aggregator(self):
        arms = build_suite_arms(
            ["multi_signal"],
            [],
            ["none", PHASE1, PHASE2],
        )

        self.assertEqual(len(arms), 3)
        self.assertTrue(all(arm["selection_mode"] == "multi_signal" for arm in arms))
        self.assertTrue(all(arm["aggregation_mode"] == "fedavg" for arm in arms))

    def test_rejects_suite_without_any_arms(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            build_suite_arms([], [], ["none"])

    def test_suite_signature_ignores_output_and_resume_paths(self):
        common = {
            "num-server-rounds": 10,
            "run-output-dir": "C:\\old\\runs",
            "suite-resume-manifest": "C:\\old\\suite_manifest.json",
        }
        changed_paths = {
            **common,
            "run-output-dir": "/mnt/runs",
            "suite-resume-manifest": "/mnt/runs/suite_manifest.json",
        }
        args = {
            "seeds": [1, 2],
            "regimes": ["iid"],
            "arms": [{"arm_id": "none__multi_signal"}],
            "calibration_sha256": None,
        }

        self.assertEqual(
            _suite_signature(common, **args),
            _suite_signature(changed_paths, **args),
        )
        self.assertNotEqual(
            _suite_signature(common, **args),
            _suite_signature({**common, "num-server-rounds": 11}, **args),
        )

    def test_resume_validation_allows_calibration_path_to_move(self):
        calibration = {
            "path": "C:\\old\\calibration.json",
            "sha256": "calibration-hash",
            "threshold": 2.5,
        }
        manifest = {
            "suite_status": "in_progress",
            "suite_signature": "suite-hash",
            "source_fingerprint_sha256": "source-hash",
            "runtime_versions": {
                "python": "3.11.8",
                "torch": "2.5.1",
                "numpy": "2.1.3",
                "flower": "1.36.0",
            },
            "threshold_calibration": calibration,
        }

        _validate_resume_checkpoint(
            manifest,
            suite_signature="suite-hash",
            source_fingerprint="source-hash",
            calibration_provenance={
                **calibration,
                "path": "/mnt/calibration.json",
            },
            runtime_versions=manifest["runtime_versions"],
        )

        with self.assertRaisesRegex(ValueError, "versions changed"):
            _validate_resume_checkpoint(
                manifest,
                suite_signature="suite-hash",
                source_fingerprint="source-hash",
                calibration_provenance=calibration,
                runtime_versions={**manifest["runtime_versions"], "torch": "other"},
            )

    def test_completed_artifact_verification_checks_identity_and_rounds(self):
        item = {"seed": 7, "regime": "iid", "arm_id": "none__multi_signal"}
        record = {
            "suite_id": "suite-1",
            "suite_regime": "iid",
            "suite_arm_id": "none__multi_signal",
            "config": {"seed": 7},
            "rounds": {
                "1": {"round_status": "AGGREGATE_FINITE"},
                "2": {"round_status": "AGGREGATE_FINITE"},
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            _validate_completed_run_artifact(
                path, item, suite_id="suite-1", num_rounds=2
            )

            record["rounds"].pop("2")
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing completed round"):
                _validate_completed_run_artifact(
                    path, item, suite_id="suite-1", num_rounds=2
                )


if __name__ == "__main__":
    unittest.main()
