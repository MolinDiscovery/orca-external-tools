"""Focused checks for the UMA plus GFN2-xTB ALPB external potential."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from oet.calculator.uma import UmaCalc, _read_xtb_gradient, _run_xtb_correction


def test_solvent_option_is_parsed_explicitly() -> None:
    """The OET CLI accepts the exact option written by FRUST."""
    _, args, remaining = UmaCalc().parse_args(
        ["water.ext", "-t", "omol", "--xtb-alpb", "chloroform", "--xtb-exe", "/opt/xtb"]
    )
    assert args["xtb_alpb"] == "chloroform"
    assert args["xtb_exe"] == "/opt/xtb"
    assert remaining == []


def test_xtb_gradient_rejects_missing_component(tmp_path: Path) -> None:
    """A partial xTB gradient cannot be used in the composite potential."""
    gradient = tmp_path / "xtb.gradient"
    gradient.write_text(
        "$grad\ncycle = 1 SCF energy = -1.0\n0.0 0.0 0.0 H\n0.1 0.2 0.3\n$end\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="shape mismatch"):
        _read_xtb_gradient(gradient, natoms=2)


@pytest.mark.parametrize("solvent", [None, "chloroform"])
def test_xtb_nonzero_exit_fails_external_evaluation(tmp_path: Path, solvent: str | None) -> None:
    """A failed xTB subprocess must not fall back to gas-phase UMA."""
    executable = tmp_path / "xtb-fail"
    executable.write_text("#!/bin/sh\necho deliberate-failure\nexit 7\n", encoding="utf-8")
    executable.chmod(0o755)
    xyz = tmp_path / "water.xyz"
    xyz.write_text("1\nH\nH 0 0 0\n", encoding="utf-8")
    workdir = tmp_path / "scratch"
    workdir.mkdir()
    calc_data = SimpleNamespace(xyzfile=xyz, charge=0, mult=1, ncores=1, dograd=True, natoms=1)
    with pytest.raises(RuntimeError, match="exit 7.*deliberate-failure"):
        _run_xtb_correction(calc_data, xtb_exe=executable, solvent=solvent, workdir=workdir)
