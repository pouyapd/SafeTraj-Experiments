"""
The published thesis artifacts are unchanged (SHA-256 manifest) and the
numbers in the result tables are consistent with the thesis tables.
"""

from tools import manifest
from tools.verify_results import run_checks


def test_artifacts_match_manifest():
    missing, changed, unlisted = manifest.check()
    assert not missing, f"missing artifacts: {missing}"
    assert not changed, f"modified artifacts: {changed}"
    assert not unlisted, f"artifacts not in the manifest: {unlisted}"


def test_result_tables_reproduce_thesis_numbers():
    failed = [(name, detail) for name, ok, detail in run_checks() if not ok]
    assert not failed, failed


def test_every_check_family_ran():
    names = {name for name, _, _ in run_checks()}
    assert names == {f"C{i}" for i in range(1, 9)}
