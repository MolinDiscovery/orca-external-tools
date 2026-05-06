import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from oet import ROOT_DIR
from oet.calculator.gxtb import GxtbCalc
from oet.core.test_utilities import (
    OH,
    get_filenames,
    read_result_file,
    run_wrapper,
    write_input_file,
    write_xyz_file,
)

gxtb_script_path = ROOT_DIR / "../../bin/oet_gxtb"


def run_gxtb(inputfile: str, output_file: str, exe: str | None = None) -> None:
    arguments = ["--exe", exe] if exe else None
    run_wrapper(
        inputfile=inputfile,
        script_path=gxtb_script_path,
        outfile=output_file,
        args=arguments,
    )


class GxtbV2Tests(unittest.TestCase):
    def test_resolve_gxtb_executable_uses_argument_or_env(self):
        with unittest.mock.patch.dict(os.environ, {"GXTB_EXE": "/env/xtb"}, clear=False):
            self.assertEqual(GxtbCalc.resolve_gxtb_executable(None), "/env/xtb")
            self.assertEqual(GxtbCalc.resolve_gxtb_executable("/arg/xtb"), "/arg/xtb")

    def test_gxtb_command_args_include_v2_flags_and_gradient(self):
        calc_data = Mock()
        calc_data.xyzfile = Path("mol.xyz")
        calc_data.charge = -1
        calc_data.mult = 2
        calc_data.ncores = 4
        calc_data.basename = "mol_EXT"
        calc_data.dograd = True

        args = GxtbCalc.gxtb_command_args(calc_data, extra_args=["--acc", "0.2"])

        self.assertEqual(args[:2], ["mol.xyz", "--gxtb"])
        self.assertIn("--grad", args)
        self.assertIn("--namespace", args)
        self.assertEqual(args[args.index("--chrg") + 1], "-1")
        self.assertEqual(args[args.index("--uhf") + 1], "1")
        self.assertEqual(args[args.index("--parallel") + 1], "4")
        self.assertEqual(args[args.index("--namespace") + 1], "mol_EXT")
        self.assertEqual(args[-2:], ["--acc", "0.2"])

    def test_gxtb_command_args_omit_gradient_for_sp(self):
        calc_data = Mock()
        calc_data.xyzfile = Path("mol.xyz")
        calc_data.charge = 0
        calc_data.mult = 1
        calc_data.ncores = 1
        calc_data.basename = "mol_EXT"
        calc_data.dograd = False

        args = GxtbCalc.gxtb_command_args(calc_data, extra_args=[])

        self.assertNotIn("--grad", args)

    def test_reads_v2_energy_and_gradient_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "mol_EXT.energy").write_text(
                "$energy\n     1    -1.16305886619    -1.16305886619    -1.16305886619\n$end\n"
            )
            (root / "mol_EXT.gradient").write_text(
                "$grad\n"
                "  cycle =      0    SCF energy =    -1.16305886619   |dE/dxyz| =  0.005192\n"
                "    0.00000000000000      0.00000000000000      0.00000000000000      H\n"
                "    0.00000000000000      0.00000000000000      1.39839733221913      H\n"
                "   0.0000000000000E+00   0.0000000000000E+00  -3.6711581268357E-03\n"
                "   0.0000000000000E+00   0.0000000000000E+00   3.6711581268357E-03\n"
                "$end\n"
            )

            cwd = Path.cwd()
            try:
                os.chdir(root)
                energy = GxtbCalc.read_energy("mol_EXT.energy")
                gradient = GxtbCalc.read_gradient("mol_EXT.gradient", natoms=2)
            finally:
                os.chdir(cwd)

        self.assertAlmostEqual(energy, -1.16305886619)
        self.assertEqual(
            gradient,
            [0.0, 0.0, -3.6711581268357e-03, 0.0, 0.0, 3.6711581268357e-03],
        )

    def test_reads_v2_energy_from_stdout_for_sp(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "mol_EXT.out"
            output.write_text(
                "electronic energy             -1.1630588661873E+00 Eh\n"
                "total energy                  -1.1630588661873E+00 Eh\n"
            )

            energy = GxtbCalc.read_energy_from_output(output)

        self.assertAlmostEqual(energy, -1.1630588661873)

    def test_live_h2_engrad_if_gxtb_exe_is_configured(self):
        gxtb_exe = os.getenv("GXTB_EXE")
        if not gxtb_exe:
            self.skipTest("GXTB_EXE is not configured")
        if not gxtb_script_path.exists():
            self.skipTest("oet_gxtb script is not installed")

        xyz_file, input_file, engrad_out, output_file = get_filenames("H2_live")
        write_xyz_file(xyz_file, OH)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=1,
            do_gradient=1,
        )
        run_gxtb(input_file, output_file, exe=gxtb_exe)

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Wrapper output not found. Check {output_file} for details."
            ) from e

        self.assertEqual(num_atoms, 2)
        self.assertLess(energy, 0.0)
        self.assertEqual(len(gradients), 6)

    def test_live_h2_sp_if_gxtb_exe_is_configured(self):
        gxtb_exe = os.getenv("GXTB_EXE")
        if not gxtb_exe:
            self.skipTest("GXTB_EXE is not configured")
        if not gxtb_script_path.exists():
            self.skipTest("oet_gxtb script is not installed")

        xyz_file, input_file, engrad_out, output_file = get_filenames("H2_live_sp")
        write_xyz_file(xyz_file, OH)
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=1,
            do_gradient=0,
        )
        run_gxtb(input_file, output_file, exe=gxtb_exe)

        num_atoms, energy, gradients = read_result_file(engrad_out)

        self.assertEqual(num_atoms, 2)
        self.assertLess(energy, 0.0)
        self.assertEqual(len(gradients), 0)

    def test_installed_entrypoints_help(self):
        if not gxtb_script_path.exists():
            self.skipTest("oet_gxtb script is not installed")
        help_out = subprocess.run(
            [gxtb_script_path, "--help"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertIn("GXTB_EXE", help_out)


if __name__ == "__main__":
    unittest.main()
