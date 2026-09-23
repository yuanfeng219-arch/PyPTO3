"""Read-only adapter for a PyPTO control-flow Review Run.

The adapter never invokes the compiler.  It binds a requested shape to an
existing, normalized Identity Index only when the artifact set can be verified
on disk and the scenario matches the scenario recorded by that index.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_evidence_path(raw_path: str, evidence_base: Path) -> Path:
    candidate = (evidence_base / raw_path).resolve()
    artifact_root = evidence_base.parent.parent.resolve()
    if not candidate.is_relative_to(artifact_root):
        raise ValueError(
            f"Evidence path escapes artifact workspace: {raw_path}"
        )
    return candidate


def _function_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    functions = payload.get("functions", [])
    if len(functions) != 1:
        raise ValueError(
            f"Expected one function in materialization dump: {path}"
        )
    function = functions[0]
    return {
        "artifact": path.name,
        "sha256": _sha256(path),
        "funcMagic": function.get("funcmagic"),
        "functionHash": function.get("hash"),
        "functionName": function.get("func_magicname"),
        "operationCount": len(function.get("operations", [])),
    }


def collect_review_run(input_t: int) -> dict[str, Any]:
    """Validate the canonical Review Run against artifacts on disk."""
    project_root = Path(__file__).resolve().parents[2]
    index_path = (
        project_root.parent
        / "docs/evidence/lightning-indexer-prolog-quant-identity-index.json"
    )
    index = json.loads(index_path.read_text())
    # Paths in the canonical evidence file are authored relative to docs/.
    evidence_base = index_path.parent.parent

    source_spec = index["reviewRun"]["sourceSnapshot"]
    source_path = _resolve_evidence_path(source_spec["path"], evidence_base)
    source_exists = source_path.is_file()
    source_hash = _sha256(source_path) if source_exists else None
    expected_source_hash = source_spec.get("contentHash")
    source_verified = bool(source_hash) and (
        not expected_source_hash or source_hash == expected_source_hash
    )

    pass_spec = index["compilerEvidence"]["loopUnrollPass"]
    before_path = _resolve_evidence_path(
        pass_spec["beforeArtifact"], evidence_base
    )
    after_path = _resolve_evidence_path(
        pass_spec["afterArtifact"], evidence_base
    )
    before_hash = _sha256(before_path) if before_path.is_file() else None
    after_hash = _sha256(after_path) if after_path.is_file() else None
    loop_unroll_verified = (
        before_hash == pass_spec["beforeSha256"]
        and after_hash == pass_spec["afterSha256"]
    )

    materialization_spec = index["compilerEvidence"][
        "firstObservedSpecializedFunctions"
    ]
    materialization_dir = _resolve_evidence_path(
        materialization_spec["artifactDirectory"], evidence_base
    )
    dumps = sorted(
        materialization_dir.glob(
            "Before_000_RemoveRedundantReshape_"
            "TENSOR_IndexerPrologQuantQuantLoop_*.json"
        )
    )
    functions = [_function_summary(path) for path in dumps]
    expected_hashes = {
        item["pathHash"] for item in index.get("generatedPaths", [])
    }
    observed_hashes = {item["functionHash"] for item in functions}
    materialization_verified = bool(expected_hashes) and (
        observed_hashes == expected_hashes
    )

    baseline_t = index["reviewRun"]["runtimeScenario"]["inputT"]["value"]
    artifacts_verified = (
        source_verified and loop_unroll_verified and materialization_verified
    )
    scenario_bound = input_t == baseline_t and artifacts_verified
    status = "complete" if scenario_bound else "draft"
    source_id = source_hash[:12] if source_hash else "missing"
    run_id = f"{index['reviewRun']['id']}:{source_id}:t{input_t}"

    return {
        "schemaVersion": "pypto.control-flow-review-run.v1",
        "id": run_id,
        "status": status,
        "capturedAt": datetime.now(UTC).isoformat(),
        "requestedScenario": {
            "inputT": input_t,
            "evidenceStatus": "bound" if scenario_bound else "request-only",
        },
        "baselineScenario": {
            "inputT": baseline_t,
            "confidence": index["reviewRun"]["runtimeScenario"]["inputT"][
                "confidence"
            ],
            "runId": index["reviewRun"]["id"],
        },
        "sourceSnapshot": {
            "path": str(source_path),
            "exists": source_exists,
            "sha256": source_hash,
            "expectedSha256": expected_source_hash,
            "verified": source_verified,
            "sizeBytes": source_path.stat().st_size if source_exists else None,
            "identityStatus": "captured" if source_exists else "missing",
        },
        "frontendMaterialization": {
            "boundary": materialization_spec["boundary"],
            "artifactDirectory": str(materialization_dir),
            "dumpCount": len(functions),
            "functions": functions,
            "expectedFunctionHashes": sorted(expected_hashes),
            "missingFunctionHashes": sorted(expected_hashes - observed_hashes),
            "verified": materialization_verified,
            "attribution": "unknown",
        },
        "loopUnrollEvidence": {
            "passId": pass_spec["passId"],
            "beforeSha256": before_hash,
            "afterSha256": after_hash,
            "byteIdentical": bool(before_hash and before_hash == after_hash),
            "verified": loop_unroll_verified,
        },
        "evidenceCompleteness": {
            "sourceSnapshot": "complete" if source_verified else "missing",
            "shapeScenario": "derived" if scenario_bound else "missing",
            "frontendMaterialization": (
                "complete" if materialization_verified else "partial"
            ),
        },
        "message": (
            f"Artifact evidence is bound to t={input_t}."
            if scenario_bound
            else f"No verified artifact is bound to requested t={input_t}; "
            f"baseline remains t={baseline_t}."
        ),
    }
