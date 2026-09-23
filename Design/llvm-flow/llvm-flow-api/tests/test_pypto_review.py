import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.services.pypto_review import collect_review_run
from app.services.pypto_run_producer import (
    get_review_run,
    produce_review_run,
    review_run_capabilities,
    start_review_run,
)


class PyPTOReviewRunTest(unittest.TestCase):
    def test_canonical_scenario_binds_verified_artifacts(self):
        review_run = collect_review_run(16)

        self.assertEqual(review_run["status"], "complete")
        self.assertEqual(
            review_run["sourceSnapshot"]["sha256"],
            "7205d9a92a3e1e60b4cc770e74c1717696ab743e49d8d5cf5bfda1845746f9a1",
        )
        self.assertTrue(review_run["sourceSnapshot"]["verified"])
        self.assertTrue(review_run["frontendMaterialization"]["verified"])
        self.assertEqual(review_run["frontendMaterialization"]["dumpCount"], 6)
        self.assertTrue(review_run["loopUnrollEvidence"]["byteIdentical"])

    def test_unbound_scenario_stays_draft(self):
        review_run = collect_review_run(33)

        self.assertEqual(review_run["status"], "draft")
        self.assertEqual(
            review_run["requestedScenario"]["evidenceStatus"],
            "request-only",
        )
        self.assertEqual(review_run["baselineScenario"]["inputT"], 16)

    def test_canonical_artifact_import_writes_immutable_manifest(self):
        with TemporaryDirectory() as directory:
            storage_root = Path(directory)
            queued = start_review_run(16, storage_root=storage_root)
            complete = produce_review_run(queued["id"], storage_root)

            self.assertEqual(complete["status"], "complete")
            self.assertEqual(complete["producer"], "artifact-import")
            self.assertEqual(
                complete["manifest"]["artifactFileCount"], 9
            )
            self.assertEqual(len(complete["manifest"]["sha256"]), 64)
            manifest_path = Path(complete["manifest"]["path"])
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(manifest_path.stat().st_mode & 0o222, 0)
            self.assertEqual(
                get_review_run(queued["id"], storage_root)["status"],
                "complete",
            )

    def test_new_shape_is_blocked_without_controlled_runner(self):
        with TemporaryDirectory() as directory:
            storage_root = Path(directory)
            queued = start_review_run(33, storage_root=storage_root)
            blocked = produce_review_run(queued["id"], storage_root)

            self.assertEqual(blocked["status"], "blocked")
            self.assertEqual(blocked["producer"], "pypto-jit")
            self.assertEqual(
                blocked["reasonCode"],
                "controlled-runner-not-configured",
            )
            self.assertEqual(blocked["result"]["status"], "draft")

    def test_capabilities_do_not_claim_local_jit_is_available(self):
        capabilities = review_run_capabilities()["producers"]
        pypto_jit = next(
            item for item in capabilities if item["kind"] == "pypto-jit"
        )

        self.assertFalse(pypto_jit["available"])
        self.assertTrue(pypto_jit["compilerInvoked"])


if __name__ == "__main__":
    unittest.main()
