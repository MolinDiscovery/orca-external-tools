#!/usr/bin/env python3
"""
This package provides [fairchem](https://github.com/facebookresearch/fairchem) wrappers for ORCA's ExtTool interface.
Before starting to use this module, please make sure your have access to the [fairchem repository](https://huggingface.co/facebook/UMA) and logged in with your
huggingface account. For details, please see the GitHub repository or the [respective tutorials](https://fair-chem.github.io/).

Provides
--------
class: UmaCalc(CalcServer)
    Class for performing a UMA calculation together with ORCA
main: function
    Main function
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oet import ASSETS_DIR
from oet.core.base_calc import BaseCalc, CalculationData
from oet.core.misc import ENERGY_CONVERSION, LENGTH_CONVERSION, mult_to_nue, xyzfile_to_at_coord

try:
    # Suppress pkg_resources deprecated warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        from fairchem.core.calculate.pretrained_mlip import available_models
        from fairchem.core.units.mlip_unit.api.inference import UMATask
        from huggingface_hub import hf_hub_download
except ImportError as e:
    print(
        f"[MISSING] Required module fairchem-core not found: {e}.\n"
        "Please install the packages in the virtual environment.\n"
        "Therefore, activate the venv, got to the orca-external-tools "
        "main directory and use pip install -r ./requirements/uma.txt\n"
        "Also, make sure you are logged in with your Hugging Face account:\n"
        "https://fair-chem.github.io/core/install.html#access-to-gated-models-on-huggingface"
    )
    sys.exit(1)

try:
    import torch
except ImportError as e:
    print("[MISSING] torch not found:", e)
    sys.exit(1)

try:
    from ase import Atoms
except ImportError as e:
    print("[MISSING] ase not found:", e)
    sys.exit(1)


# Override the default fairchem `CACHE_DIR`, unless the environment variable is set
DEFAULT_CACHE_DIR = str(os.environ.get("FAIRCHEM_CACHE_DIR", ASSETS_DIR / "fairchem"))
_XTB_ENERGY = re.compile(r"TOTAL ENERGY\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s+Eh")


@dataclass(frozen=True)
class XtbResult:
    """GFN2-xTB energy and gradient in ORCA-facing units.

    Attributes
    ----------
    energy : float
        Electronic energy in Hartree.
    gradient : list[float]
        Flattened Cartesian gradient in Hartree/Bohr.
    """

    energy: float
    gradient: list[float]


def _read_xtb_gradient(path: Path, natoms: int) -> list[float]:
    """Read and validate a namespaced xTB gradient in Hartree/Bohr.

    Parameters
    ----------
    path : Path
        Path to the xTB ``.gradient`` file.
    natoms : int
        Expected number of atoms.

    Returns
    -------
    list[float]
        Flattened Cartesian gradient in input atom order.
    """
    if not path.is_file():
        raise RuntimeError(f"GFN2-xTB did not produce gradient file: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not any(line.strip() == "$grad" for line in lines):
        raise RuntimeError(f"GFN2-xTB gradient has no $grad block: {path}")
    gradient: list[float] = []
    coordinate_rows = 0
    in_gradient = False
    for line in lines:
        fields = line.split()
        if fields == ["$grad"]:
            in_gradient = True
            continue
        if not in_gradient:
            continue
        if fields == ["$end"]:
            break
        if len(fields) == 4 and fields[-1].isalpha():
            coordinate_rows += 1
        elif len(fields) == 3:
            try:
                gradient.extend(float(value.replace("D", "E")) for value in fields)
            except ValueError as exc:
                raise RuntimeError(f"Malformed GFN2-xTB gradient: {path}") from exc
    if coordinate_rows != natoms or len(gradient) != 3 * natoms:
        raise RuntimeError(
            f"GFN2-xTB gradient shape mismatch in {path}: "
            f"{coordinate_rows} atoms, {len(gradient)} components; expected {natoms}, {3 * natoms}"
        )
    if not all(math.isfinite(value) for value in gradient):
        raise RuntimeError(f"GFN2-xTB gradient contains nonfinite values: {path}")
    return gradient


def _run_xtb_correction(
    calc_data: CalculationData,
    *,
    xtb_exe: Path,
    solvent: str | None,
    workdir: Path,
) -> XtbResult:
    """Evaluate GFN2-xTB with optional ALPB in a private scratch directory.

    Parameters
    ----------
    calc_data : CalculationData
        ORCA request and molecular geometry.
    xtb_exe : Path
        Normal xTB executable, not the g-xTB fork.
    solvent : str or None
        ALPB solvent, or ``None`` for gas phase.
    workdir : Path
        Private scratch directory for this xTB evaluation.

    Returns
    -------
    XtbResult
        Energy and requested gradient in Hartree and Hartree/Bohr.
    """
    label = f"GFN2-xTB ALPB({solvent})" if solvent else "GFN2-xTB gas"
    args = [
        str(xtb_exe),
        str(calc_data.xyzfile),
        "--gfn",
        "2",
        "--chrg",
        str(calc_data.charge),
        "--uhf",
        str(mult_to_nue(calc_data.mult)),
        "--parallel",
        str(calc_data.ncores),
        "--namespace",
        "xtb",
    ]
    if solvent:
        args.extend(["--alpb", solvent])
    if calc_data.dograd:
        args.append("--grad")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(calc_data.ncores)
    result = subprocess.run(args, cwd=workdir, env=env, text=True, capture_output=True, check=False)
    output = result.stdout + result.stderr
    (workdir / "xtb.out").write_text(output, encoding="utf-8")
    if result.returncode != 0:
        detail = output.strip().splitlines()[-8:]
        raise RuntimeError(f"{label} failed (exit {result.returncode}): {' | '.join(detail)}")
    if "normal termination of xtb" not in output:
        detail = output.strip().splitlines()[-8:]
        raise RuntimeError(f"{label} did not report normal termination: {' | '.join(detail)}")
    energies = _XTB_ENERGY.findall(output)
    if not energies:
        raise RuntimeError(f"{label} did not report a total energy: {workdir / 'xtb.out'}")
    energy = float(energies[-1])
    if not math.isfinite(energy):
        raise RuntimeError(f"{label} returned a nonfinite energy")
    gradient = (
        _read_xtb_gradient(workdir / "xtb.gradient", calc_data.natoms) if calc_data.dograd else []
    )
    return XtbResult(energy=energy, gradient=gradient)


class UmaCalc(BaseCalc):
    # Fairchem calculator used to compute energy and grad
    _calc: FAIRChemCalculator | None = None

    def set_calculator(
        self,
        param: str,
        basemodel: str,
        device: str,
        cache_dir: str,
        inference_settings: str = "batch",
        force: bool = False,
    ) -> None:
        """
        Prepare the `FAIRChemCalculator` object to compute energy and gradient, if not done already.

        Parameters
        ----------
        param: str
            Parameter set used by fairchem
        basemodel: str
            UMA basemodel
        device: str
            Device that should be used, e.g., cpu or cuda
        cache_dir: str
            Cache directory to read/write downloaded model files to
        inference_settings: str, default = "batch"
            FairChem inference mode. ``batch`` avoids CPU compilation.
        force: bool, default = False
            Force re-initialization of the calculator, even if already initialized
        """
        if not self._calc or force:
            # Make sure the cache directory exists
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            # Monkey-patch the Fairchem CACHE_DIR: the provided one is not always respected.
            # In particular, `pretrained_checkpoint_path_from_name` just uses `CACHE_DIR`
            pretrained_mlip.CACHE_DIR = cache_dir
            # Suppress fairchemcore internal warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predictor = pretrained_mlip.get_predict_unit(
                    basemodel,
                    device=device,
                    cache_dir=cache_dir,
                    inference_settings=inference_settings,
                )
                self._calc = FAIRChemCalculator(predictor, task_name=param)

    def get_calculator(self) -> FAIRChemCalculator:
        """
        Returns UMA calculator
        """
        return self._calc

    def check_for_model_files(self, basemodel: str, cache_dir: str) -> bool:
        """
        Check if model files are available in current cache directory.

        Parameters
        ----------
        basemodel: str
            UMA basemodel
        cache_dir: str
            Cache directory

        Returns
        -------
        bool
            True, if model files were found
        """
        try:
            # First the model parameter
            hf_hub_download(
                filename=basemodel + ".pt",
                repo_id="facebook/UMA",
                subfolder="checkpoints",
                cache_dir=cache_dir,
                local_files_only=True,
            )
            # Then the atomic references
            hf_hub_download(
                filename="iso_atom_elem_refs.yaml",
                repo_id="facebook/UMA",
                subfolder="references",
                cache_dir=cache_dir,
                local_files_only=True,
            )
        except Exception:  # noqa: BLE001 - any cache lookup failure means the files are unavailable
            return False
        else:
            return True

    def switch_to_offline_mode(self) -> None:
        """
        Goes into offline mode to prevent HuggingFace from downloading something from the web
        """
        os.environ["HF_HUB_OFFLINE"] = "1"
        # Older versions require a different keyword:
        os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"

    @classmethod
    def extend_parser(cls, parser: ArgumentParser) -> None:
        """Add Uma parsing options.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        parser.add_argument(
            "-t",
            "--task",
            type=UMATask,
            choices=list(UMATask),
            default=(default_task := UMATask.OMOL),
            metavar="TASK",
            dest="param",
            help="The UMA task/parameter set name. "
            "Options: " + ", ".join(UMATask) + ". "
            f"Default: {default_task}. ",
        )
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            default=(default_model := "uma-s-1p1"),
            metavar="MODEL",
            dest="basemodel",
            choices=available_models,
            help="The UMA base model. "
            "Options: " + ", ".join(available_models) + ". "
            f"Default: {default_model}. ",
        )
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            default="cpu",
            metavar="DEVICE",
            dest="device",
            choices=(device_choices := ["cpu", "cuda"]),
            help="Device to perform the calculation on. "
            "Options: " + ", ".join(device_choices) + ". "
            "Default: cpu. ",
        )
        parser.add_argument(
            "-c",
            "--cachedir",
            type=str,
            default=str(DEFAULT_CACHE_DIR),
            metavar="DIR",
            dest="cache_dir",
            help="The cache directory to store downloaded model files. "
            "Can also be set via the environment variable FAIRCHEM_CACHE_DIR. "
            f'Default: "{DEFAULT_CACHE_DIR}".',
        )
        parser.add_argument(
            "-o",
            "--offline",
            type=bool,
            default=False,
            dest="offline_mode",
            help="Force into offline mode. Please note that there will be an error if the model parameters are not found.",
        )
        parser.add_argument(
            "--xtb-alpb",
            choices=["chloroform"],
            metavar="SOLVENT",
            help=(
                "Add E(GFN2-xTB, ALPB solvent) - E(GFN2-xTB, gas) and the matching "
                "gradient to UMA. Currently supported: chloroform."
            ),
        )
        parser.add_argument(
            "--xtb-exe",
            metavar="PATH",
            help="Normal xTB executable for --xtb-alpb; defaults to XTB_EXE or PATH.",
        )
        parser.add_argument(
            "--inference-settings",
            choices=["default", "batch", "turbo"],
            default="batch",
            help="FairChem inference mode; batch avoids CPU compilation. Default: batch.",
        )

    def run_uma(
        self,
        atom_types: list[str],
        coordinates: list[tuple[float, float, float]],
        calc_data: CalculationData,
    ) -> tuple[float, list[float]]:
        """
        Runs an UMA calculation.

        Parameters
        ----------
        atom_types : list[str]
            List of element symbols (e.g., ["O", "H", "H"])
        coordinates : list[tuple[float, float, float]]
            List of (x, y, z) coordinates
        calc_data: CalculationData
            Object with calculation data for the run

        Returns
        -------
        float
            The computed energy (Eh)
        list[float]
            Flattened gradient vector (Eh/Bohr), if computed, otherwise empty
        """

        # set the number of threads
        torch.set_num_threads(calc_data.ncores)

        # make ase atoms object for calculation
        atoms = Atoms(symbols=atom_types, positions=coordinates)
        atoms.info = {"charge": calc_data.charge, "spin": calc_data.mult}
        atoms.calc = self._calc

        # Suppress fairchemcore internal warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            energy = atoms.get_potential_energy() / ENERGY_CONVERSION["eV"]
        gradient = []
        if calc_data.dograd:
            forces = atoms.get_forces()
            # Convert forces to gradient (-1) and unit conversion
            fac = -LENGTH_CONVERSION["Ang"] / ENERGY_CONVERSION["eV"]
            gradient = (fac * forces).flatten().tolist()

        return energy, gradient

    def calc(
        self,
        calc_data: CalculationData,
        args_parsed: dict[str, Any],
        args_not_parsed: list[str],
    ) -> tuple[float, list[float]]:
        """
        Routine for calculating energy and optional gradient.
        Writes ORCA output


        Parameters
        ----------
        calc_data: CalculationData
            Object with calculation data for the run
        args_parsed: dict[str, Any]
            Arguments parsed as defined in extend_parser
        args_not_parsed: list[str]
            Arguments not parsed so far

        Returns
        -------
        float
            The computed energy (Eh)
        list[float]
            Flattened gradient vector (Eh/Bohr), if computed, otherwise empty
        """
        # Get the arguments parsed as defined in extend_parser
        param = args_parsed.get("param")
        basemodel = args_parsed.get("basemodel")
        device = args_parsed.get("device")
        cache_dir = args_parsed.get("cache_dir")
        offline_mode = args_parsed.get("offline_mode")
        xtb_alpb = args_parsed.get("xtb_alpb")
        xtb_exe_arg = args_parsed.get("xtb_exe")
        inference_settings = args_parsed.get("inference_settings", "batch")
        if (
            not isinstance(param, str)
            or not isinstance(basemodel, str)
            or not isinstance(device, str)
            or not isinstance(cache_dir, str)
        ):
            raise TypeError("Problems handling input parameters.")
        if args_not_parsed:
            raise ValueError(f"Unrecognized UMA arguments: {args_not_parsed}")
        if xtb_exe_arg and not xtb_alpb:
            raise ValueError("--xtb-exe requires --xtb-alpb")
        xtb_exe: Path | None = None
        if xtb_alpb:
            if calc_data.pointcharges:
                raise NotImplementedError(
                    "UMA xTB solvent correction does not support point charges"
                )
            requested_xtb = xtb_exe_arg or os.environ.get("XTB_EXE") or "xtb"
            resolved_xtb = shutil.which(requested_xtb)
            if resolved_xtb is None:
                raise FileNotFoundError(f"GFN2-xTB executable not found: {requested_xtb}")
            xtb_exe = Path(resolved_xtb).resolve()
        # Check if the model files are available
        model_files_available = self.check_for_model_files(basemodel=basemodel, cache_dir=cache_dir)
        # If they are available, switch to offline mode.
        if model_files_available:
            self.switch_to_offline_mode()
        # If they are not available, but the user requested the offline mode to prevent online communication,
        # print a warning as this will likely cause subsequent errors.
        elif offline_mode:
            self.switch_to_offline_mode()
            # Check if the model files are locally available. If not, subsequent errors will occur
            # as they cannot be downloaded.
            print(
                "WARNING: Offline mode selected, but no model files were detected. "
                "This will likely cause subsequent errors."
            )

        # setup calculator if not already set
        # this is important as usage on a server would otherwise cause
        # initialization with every call so that nothing is gained
        self.set_calculator(
            param=param,
            basemodel=basemodel,
            device=device,
            cache_dir=cache_dir,
            inference_settings=inference_settings,
        )

        # process the XYZ file
        atom_types, coordinates = xyzfile_to_at_coord(calc_data.xyzfile)

        # run uma
        energy, gradient = self.run_uma(
            atom_types=atom_types, coordinates=coordinates, calc_data=calc_data
        )

        if calc_data.dograd and len(gradient) != 3 * calc_data.natoms:
            raise RuntimeError(
                f"UMA returned {len(gradient)} gradient components; expected {3 * calc_data.natoms}"
            )
        if not math.isfinite(energy) or not all(math.isfinite(value) for value in gradient):
            raise RuntimeError("UMA returned a nonfinite energy or gradient")

        uma_energy = energy
        components: dict[str, float] = {"uma": uma_energy}
        if xtb_alpb:
            assert xtb_exe is not None
            with tempfile.TemporaryDirectory(prefix="uma_xtb_", dir=calc_data.tmp_dir) as tmp:
                root = Path(tmp)
                gas_dir = root / "gas"
                alpb_dir = root / "alpb"
                gas_dir.mkdir()
                alpb_dir.mkdir()
                gas = _run_xtb_correction(calc_data, xtb_exe=xtb_exe, solvent=None, workdir=gas_dir)
                alpb = _run_xtb_correction(
                    calc_data, xtb_exe=xtb_exe, solvent=xtb_alpb, workdir=alpb_dir
                )
            energy += alpb.energy - gas.energy
            if calc_data.dograd:
                gradient = [
                    uma_value + alpb_value - gas_value
                    for uma_value, alpb_value, gas_value in zip(
                        gradient, alpb.gradient, gas.gradient, strict=True
                    )
                ]
            components.update({"xtb_alpb": alpb.energy, "xtb_gas": gas.energy})

        record = {
            "task": param,
            "uma_model": basemodel,
            "inference_settings": inference_settings,
            "xtb_method": "GFN2-xTB" if xtb_alpb else None,
            "solvent_model": "ALPB" if xtb_alpb else None,
            "solvent": xtb_alpb,
            "xtb_executable": str(xtb_exe) if xtb_exe else None,
            "energy_components_Eh": components,
            "total_energy_Eh": energy,
            "gradient_components": len(gradient),
        }
        record_path = calc_data.orca_input_dir / f"{calc_data.basename}.uma.json"
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

        return energy, gradient


def main() -> None:
    """
    Main routine for execution
    """
    calculator = UmaCalc()
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.run(inputfile=inputfile, args_parsed=args, args_not_parsed=args_not_parsed)


# Python entry point
if __name__ == "__main__":
    main()
