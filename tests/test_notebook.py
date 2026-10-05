"""Static checks of the Kaggle launcher: it cannot be executed without a CUDA session."""

import ast
import json
import re

from mgn_vortex.config import CONFIG_DIR, REPO_ROOT, load_config, steps_per_run

NOTEBOOK = REPO_ROOT / "kaggle" / "run_cuda.ipynb"


def _code():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _text():
    nb = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def test_cells_parse_and_have_no_stored_output():
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == []


def test_ref_must_be_a_full_sha_and_head_is_verified():
    code = _code()
    assert 'fullmatch(r"[0-9a-f]{40}", REF)' in code[0]
    text = "\n".join(code)
    assert "assert head == REF" in text and 'assert tree_is_clean(), "the working tree is not clean"' in text


def test_expensive_steps_are_off_by_default_and_guarded():
    code = _code()
    assert re.search(r"^RUN_FULL = False\b", code[0], re.M)
    assert re.search(r"^RUN_OFFICIAL = False\b", code[0], re.M)
    for command, flag in (("--config full --resume", "RUN_FULL"), ("--config official --resume", "RUN_OFFICIAL")):
        cells = [c for c in code if command in c]
        assert len(cells) == 1
        # the step starts only behind `is not True` and a budget comparison, never by reducing the config
        assert f"if {flag} is not True:" in cells[0]
        assert "> hours_left()" in cells[0] and "was not reduced" in cells[0]
        assert cells[0].index(f"if {flag} is not True:") < cells[0].index(command)
        assert "study." not in cells[0] and "epochs=" not in cells[0]  # no overrides of the study


def test_order_of_the_checks():
    text = "\n".join(_code())
    order = ["nvidia-smi", "git\", \"checkout", "pip install", "pytest", "download_data.py", "--config smoke", "probe.py --config full", "--config full --resume"]
    positions = [text.index(item) for item in order]
    assert positions == sorted(positions)


def test_official_runs_only_after_full():
    cell = next(c for c in _code() if "--config official --resume" in c)
    assert "elif not full_done:" in cell


def test_versions_are_printed():
    text = "\n".join(_code())
    for item in ("torch.__version__", "torch.version.cuda", "physicsnemo.__version__", "torch_geometric.__version__", "torch_scatter.__version__", "get_device_name"):
        assert item in text


def test_referenced_scripts_and_configs_exist():
    text = "\n".join(_code())
    for script in set(re.findall(r"scripts/\w+\.py", text)):
        assert (REPO_ROOT / script).exists(), script
    for name in set(re.findall(r"--config (\w+)", text)):
        assert (CONFIG_DIR / f"{name}.yaml").exists(), name


def test_notebook_numbers_match_the_configs():
    full, official = load_config("full"), load_config("official")
    text = _text()
    assert f"--train {full.num_training_samples} --valid {full.validation.num_samples} --test {full.num_test_samples}" in text
    assert f"--train {official.num_training_samples} " in text
    assert f"{steps_per_run(full):,} gradient steps each" in text
    assert f"{steps_per_run(official):,} gradient steps" in text
    assert f"{len(full.study.processor_sizes) * len(full.study.seeds)} runs" in text
    assert f"{full.num_test_time_steps - 1}-step rollout" in text
