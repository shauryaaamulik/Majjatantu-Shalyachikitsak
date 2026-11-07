from pathlib import Path

from majjatantu.reporting import EditLog
from majjatantu.types import EditLogEntry, EditSpec, PatchDecision


def test_edit_log_roundtrip(tmp_path: Path):
    path = tmp_path / "edits.jsonl"
    log = EditLog(path)
    edit = EditSpec(
        edit_id="abc",
        operator="direction_ablation",
        target_params=["layer.weight"],
        params_touched=64,
        fraction_touched=0.0001,
        description="test edit",
    )
    entry = EditLogEntry(
        edit=edit,
        model_hash_before="aaa",
        model_hash_after="bbb",
        decision=PatchDecision.REJECT,
        metrics={"target": 0.1},
        forward_passes_used=3,
        notes="overfit risk",
    )
    log.append(entry)
    assert len(log) == 1

    reloaded = EditLog(path)
    assert len(reloaded) == 1
    assert reloaded.entries()[0].decision == PatchDecision.REJECT
    assert reloaded.entries()[0].edit.params_touched == 64
