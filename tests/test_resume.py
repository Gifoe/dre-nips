from pathlib import Path

from biodynformer.orchestrator import should_skip_run, write_run_done


def test_resume_skips_completed_fold(tmp_path: Path):
    run_dir = tmp_path / "v1" / "task1" / "5fold" / "fold_1"

    assert not should_skip_run(run_dir, resume=True)
    write_run_done(run_dir, {"metric": 1.0})
    assert should_skip_run(run_dir, resume=True)
    assert not should_skip_run(run_dir, resume=False)
