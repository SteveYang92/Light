import json
import subprocess
import tempfile
import time
from pathlib import Path

from .cache import ASRCache
from .checker import RegressionChecker
from .history import HistoryManager
from .models import DiffReport, RunRecord, TestCase


class RegressionRunner:
    def __init__(self, snapshots_dir: Path):
        self.history = HistoryManager(snapshots_dir)
        self.checker = RegressionChecker()
        self.cache = ASRCache()

    def run(self, case: TestCase, keep_subtitle: bool = False) -> tuple[RunRecord, DiffReport]:
        run_id = time.strftime("%Y%m%dT%H%M%S")

        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)

            has_cached_asr = self._prepare_asr_cache(case, tmpdir)

            start = time.time()
            subtitle_path = self._generate_subtitle(case, tmpdir, resume_from_correct=has_cached_asr)
            duration = time.time() - start

            # Save ASR transcript to cache for future runs
            if not has_cached_asr:
                transcript_path = tmpdir / "transcript.json"
                if transcript_path.exists():
                    self.cache.save(case.input_audio, transcript_path)

            report = self._run_qc(case, subtitle_path, tmpdir)

            record = RunRecord(
                run_id=run_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                case_name=case.name,
                report=report,
                duration_sec=duration,
                git_commit=self._get_git_commit(),
            )

            self.history.save_run(case.name, record, subtitle_path if keep_subtitle else None)

            # Prefer a fixed golden baseline (set via `rebaseline`) for cross-person
            # comparability; fall back to the immediately previous run when no
            # baseline is set yet (first run / not yet rebaselined).
            prev = self.history.get_baseline(case.name) or self.history.get_previous_run(case.name, run_id)
            if prev is None:
                diff = DiffReport(
                    baseline_run_id=run_id,
                    current_run_id=run_id,
                    errors_delta=0,
                    warnings_delta=0,
                    suggestions_delta=0,
                    rule_changes=[],
                    new_issues=[],
                    fixed_issues=[],
                    degraded=False,
                    reasons=["First run — no baseline and no previous to compare"],
                )
            else:
                diff = self.checker.compare(prev, record, case.thresholds)

            self.history.save_diff(case.name, run_id, diff)
            return record, diff

    def _prepare_asr_cache(self, case: TestCase, output_dir: Path) -> bool:
        """Return True if ASR cache was restored (skipping ASR on next run)."""
        if case.input_asr and case.input_asr.exists():
            import shutil

            shutil.copy2(case.input_asr, output_dir / "transcript.json")
            return True

        cached = self.cache.get(case.input_audio)
        if cached:
            import shutil

            shutil.copy2(cached, output_dir / "transcript.json")
            return True

        return False

    def _generate_subtitle(self, case: TestCase, output_dir: Path, resume_from_correct: bool = False) -> Path:
        import sys

        cmd = [
            "uv",
            "run",
            "light-subtitle",
            "-i",
            str(case.input_audio),
            "-o",
            str(output_dir),
        ]
        if resume_from_correct:
            cmd.extend(["--resume-from", "correct"])
        if case.subtitle_config.get("source_lang"):
            cmd.extend(["-l", case.subtitle_config["source_lang"]])
        if case.subtitle_config.get("target_lang"):
            cmd.extend(["--target-lang", case.subtitle_config["target_lang"]])
        if case.subtitle_config.get("bilingual"):
            cmd.append("--bilingual")

        # Show stderr on failure so users can diagnose API / cache issues.
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            print("  [regression] light-subtitle timed out after 300s", file=sys.stderr)
            raise

        if result.returncode != 0:
            print("  [regression] light-subtitle failed:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            result.check_returncode()

        srt_files = list(output_dir.glob("*.srt"))
        if not srt_files:
            raise FileNotFoundError("No subtitle file generated")
        return srt_files[0]

    def _run_qc(self, case: TestCase, subtitle_path: Path, tmpdir: Path) -> dict:
        qc_output = tmpdir / "qc_report.json"
        cmd = [
            "uv",
            "run",
            "light-qc",
            "-i",
            str(subtitle_path),
            "-f",
            "json",
            "-o",
            str(qc_output),
        ]
        if case.qc_config.get("source_lang"):
            cmd.extend(["--source-lang", case.qc_config["source_lang"]])
        if case.qc_config.get("target_lang"):
            cmd.extend(["--target-lang", case.qc_config["target_lang"]])
        if case.qc_config.get("bilingual"):
            cmd.append("--bilingual")

        transcript_path = tmpdir / "transcript.json"
        if transcript_path.exists():
            cmd.extend(["--transcript", str(transcript_path)])

        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return json.loads(qc_output.read_text(encoding="utf-8"))

    def _get_git_commit(self) -> str | None:
        try:
            result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except Exception:
            return None
