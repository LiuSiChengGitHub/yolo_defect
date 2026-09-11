"""Exercise workflow dispatch without running a compiler or inference session."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1] / "tools"


class Stage1WorkflowTest(unittest.TestCase):
    def invoke(self, action, fail_native=False):
        """Run the real dispatch with external work replaced by event recording."""
        if os.name == "nt":
            shell = shutil.which("powershell.exe")
            source = str(TOOLS / "stage1.ps1").replace("'", "''")
            python = sys.executable.replace("'", "''")
            harness = r"""
$ErrorActionPreference = 'Stop'
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    '__SOURCE__', [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
foreach ($function in $ast.FindAll({param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst]
}, $false)) { Invoke-Expression $function.Extent.Text }
function Invoke-EnsureBuild { Write-Output 'EVENT build' }
function Invoke-CleanBuild { Write-Output 'EVENT clean' }
function Invoke-FullTests { Write-Output 'EVENT test' }
function Invoke-Demo { Write-Output 'EVENT demo' }
function Invoke-Consistency { Write-Output 'EVENT consistency' }
function Invoke-Benchmark {
    Write-Output "EVENT benchmark $script:ResolvedWarmup $script:ResolvedRepeat"
}
function Invoke-BatchRun { Write-Host 'EVENT batch' }
$script:RunDir = [IO.Path]::GetTempPath()
$script:BatchManifestPath = 'manifest'
$script:ConfigPath = 'config'
$script:ResolvedWarmup = 10
$script:ResolvedRepeat = 100
$Action = '__ACTION__'
__EXECUTE__
"""
            execute = r"""
$dispatch = @($ast.FindAll({param($node)
    $node -is [System.Management.Automation.Language.SwitchStatementAst] -and
    $node.Condition.Extent.Text -eq '$Action'
}, $true))[-1]
Invoke-Expression $dispatch.Extent.Text
"""
            if fail_native:
                execute = (
                    "Invoke-NativeStep -Name child -FilePath '"
                    + python
                    + "' -Arguments @('-c', 'raise SystemExit(7)')"
                )
            harness = harness.replace("__SOURCE__", source).replace(
                "__ACTION__", action
            ).replace("__EXECUTE__", execute)
            suffix = ".ps1"
            args = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
        else:
            shell = shutil.which("bash")
            harness = (TOOLS / "stage1.sh").read_text().rsplit('main "$@"', 1)[0]
            harness += r"""
preflight() { :; }
stage() { :; }
ensure_build() { printf 'EVENT build\n'; }
clean_build() { printf 'EVENT clean\n'; }
run_tests() { printf 'EVENT test\n'; }
run_demo() { printf 'EVENT demo\n'; }
run_consistency() { printf 'EVENT consistency\n'; }
run_benchmark() { printf 'EVENT benchmark %s %s\n' "$1" "$2"; }
run_batch() { printf 'EVENT batch\n'; }
new_run_dir() { RUN_DIR=/unused; }
"""
            if fail_native:
                harness += 'run child "$1" -c "raise SystemExit(7)"\n'
            else:
                harness += 'main "$1"\n'
            suffix = ".sh"
            args = [shell]

        self.assertIsNotNone(shell, "the platform workflow shell must be available")
        with tempfile.TemporaryDirectory(prefix="stage1_workflow_test_") as tmp:
            path = Path(tmp) / ("harness" + suffix)
            path.write_text(harness, encoding="utf-8")
            if os.name == "nt":
                args.append(str(path))
            else:
                args.extend([str(path), sys.executable if fail_native else action])
            return subprocess.run(args, capture_output=True, text=True)

    def events(self, action):
        result = self.invoke(action)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return [line for line in result.stdout.splitlines() if line.startswith("EVENT ")]

    def test_benchmark_does_not_run_consistency(self):
        self.assertEqual(self.events("benchmark"), ["EVENT build", "EVENT benchmark 10 100"])

    def test_all_builds_once_without_repeating_ctest_coverage(self):
        self.assertEqual(
            self.events("all"),
            ["EVENT build", "EVENT test", "EVENT benchmark 10 100", "EVENT batch"],
        )

    def test_standalone_actions_remain_available(self):
        self.assertEqual(self.events("consistency"), ["EVENT build", "EVENT consistency"])
        self.assertEqual(self.events("clean-build"), ["EVENT clean"])

    def test_native_failure_preserves_exit_code(self):
        result = self.invoke("benchmark", fail_native=True)
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertIn("exit code 7", result.stderr)


if __name__ == "__main__":
    unittest.main()
