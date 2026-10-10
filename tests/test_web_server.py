import unittest
import json
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from cosine_filter_audit import web_server
from cosine_filter_audit.web_server import (
    LocalUIHandler,
    flower_run_config,
    validate_run_config,
)


def valid_payload(**overrides):
    payload = {
        "representation_mode": "delta",
        "reference_mode": "current_mean",
        "selection_mode": "detector",
        "threshold": 0.7,
        "attack_enabled": False,
        "attack_scale": -7.0,
        "num_rounds": 1,
        "seed": 42,
    }
    payload.update(overrides)
    return payload


class WebConfigTests(unittest.TestCase):
    def test_builds_bounded_run_config_and_disables_attack_by_default(self):
        config = validate_run_config(valid_payload())
        encoded = flower_run_config(config)
        self.assertIn("num-server-rounds=1", encoded)
        self.assertIn("attack-start-round=999", encoded)
        self.assertIn("selection-mode='detector'", encoded)

    def test_attack_is_explicitly_enabled(self):
        config = validate_run_config(valid_payload(attack_enabled=True))
        self.assertIn("attack-start-round=1", flower_run_config(config))

    def test_legacy_reference_is_valid_only_for_legacy_representation(self):
        payload = valid_payload(
            representation_mode="legacy_full_model",
            reference_mode="previous_global_model",
        )
        self.assertEqual(
            validate_run_config(payload)["reference_mode"], "previous_global_model"
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_run_config(
                valid_payload(reference_mode="previous_global_model")
            )

    def test_oracle_selection_is_delta_only(self):
        with self.assertRaisesRegex(ValueError, "requires delta"):
            validate_run_config(
                valid_payload(
                    representation_mode="legacy_full_model",
                    selection_mode="oracle",
                )
            )

    def test_rejects_unknown_fields_and_modes(self):
        payload = valid_payload(extra_command="echo unsafe")
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_run_config(payload)
        with self.assertRaisesRegex(ValueError, "Unsupported selection"):
            validate_run_config(valid_payload(selection_mode="shell"))

    def test_rejects_unbounded_or_nonfinite_parameters(self):
        for update in (
            {"num_rounds": 11},
            {"num_rounds": True},
            {"threshold": float("nan")},
            {"seed": -1},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_run_config(valid_payload(**update))


class LocalUIHandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        root = Path(cls.tempdir.name)
        cls.runs_root = root / "runs"
        cls.results_root = root / "results"
        run_id = "20261005T130353Z_ddcc4b91d6a5"
        artifact = (
            cls.results_root
            / "experiment_artifacts"
            / "smoke"
            / run_id
            / "run.json"
        )
        artifact.parent.mkdir(parents=True)
        artifact.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "created_at": "2026-10-05T13:05:15+00:00",
                    "config": {
                        "representation_mode": "delta",
                        "selection_mode": "none",
                        "reference_mode": "current_mean",
                        "attack_scale": 1.0,
                        "seed": 42,
                    },
                    "provenance": {"code_revision": None},
                    "server_evaluation_metrics": {
                        "1": {"accuracy": 0.5, "loss": 1.0}
                    },
                    "rounds": {
                        "1": {
                            "round_status": "AGGREGATE_FINITE",
                            "selected_node_ids": [12],
                            "clients": {"12": {"flagged": False}},
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        cls.runs_patch = patch.object(web_server, "RUNS_ROOT", cls.runs_root)
        cls.results_patch = patch.object(web_server, "RESULTS_ROOT", cls.results_root)
        cls.runs_patch.start()
        cls.results_patch.start()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), LocalUIHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.results_patch.stop()
        cls.runs_patch.stop()
        cls.tempdir.cleanup()

    def test_serves_ui_and_lists_curated_run_artifacts(self):
        with urlopen(f"{self.base_url}/") as response:
            page = response.read().decode("utf-8")
            self.assertIn("Cosine Filter Audit", page)
            self.assertEqual(
                response.headers["Content-Security-Policy"],
                "default-src 'self'; style-src 'self'; script-src 'self'; "
                "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            )
        with urlopen(f"{self.base_url}/api/runs") as response:
            payload = json.load(response)
        self.assertEqual(len(payload["runs"]), 1)
        self.assertEqual(payload["artifact_errors"], [])

    def test_run_detail_rejects_traversal_and_serves_known_artifact(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(f"{self.base_url}/api/runs/..%5C..%5Csecret")
        self.assertEqual(error.exception.code, 400)
        run_id = "20261005T130353Z_ddcc4b91d6a5"
        with urlopen(f"{self.base_url}/api/runs/{run_id}") as response:
            record = json.load(response)
        self.assertEqual(record["run_id"], run_id)

    def test_start_requires_same_origin(self):
        request = Request(
            f"{self.base_url}/api/start",
            data=json.dumps(valid_payload()).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": "http://example.invalid",
            },
            method="POST",
        )
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 403)

    def test_start_uses_allowlisted_flower_config(self):
        request = Request(
            f"{self.base_url}/api/start",
            data=json.dumps(valid_payload(attack_enabled=True)).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": self.base_url,
            },
            method="POST",
        )
        with patch.object(
            web_server,
            "_start_run",
            return_value={"job_id": "job", "started_at": "now", "config": {}},
        ) as launch:
            with urlopen(request) as response:
                result = json.load(response)
        self.assertEqual(response.status, 202)
        self.assertEqual(result["job"]["job_id"], "job")
        self.assertEqual(launch.call_args.args[0]["selection_mode"], "detector")


if __name__ == "__main__":
    unittest.main()
