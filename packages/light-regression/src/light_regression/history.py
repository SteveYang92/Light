import json
from pathlib import Path

from .models import DiffReport, RunRecord


class HistoryManager:
    def __init__(self, snapshots_dir: Path):
        self.snapshots_dir = snapshots_dir

    def case_dir(self, case_name: str) -> Path:
        return self.snapshots_dir / case_name

    def run_dir(self, case_name: str, run_id: str) -> Path:
        return self.case_dir(case_name) / "runs" / run_id

    def save_run(self, case_name: str, record: RunRecord, subtitle_path: Path | None = None) -> None:
        rdir = self.run_dir(case_name, record.run_id)
        rdir.mkdir(parents=True, exist_ok=True)

        run_meta = {
            "run_id": record.run_id,
            "timestamp": record.timestamp,
            "case_name": record.case_name,
            "duration_sec": record.duration_sec,
            "git_commit": record.git_commit,
        }
        (rdir / "run.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        (rdir / "report.json").write_text(json.dumps(record.report, indent=2, ensure_ascii=False), encoding="utf-8")

        if subtitle_path and subtitle_path.exists():
            import shutil

            shutil.copy2(subtitle_path, rdir / "subtitle.srt")

        self._update_manifest(case_name, record.run_id)

    def _update_manifest(self, case_name: str, run_id: str) -> None:
        mpath = self.case_dir(case_name) / "manifest.json"
        manifest = (
            json.loads(mpath.read_text(encoding="utf-8"))
            if mpath.exists()
            else {
                "case_name": case_name,
                "baseline_run_id": None,
                "latest_run_id": None,
                "runs": [],
            }
        )
        manifest["latest_run_id"] = run_id
        if run_id not in manifest["runs"]:
            manifest["runs"].append(run_id)
        mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def set_baseline(self, case_name: str, run_id: str) -> None:
        mpath = self.case_dir(case_name) / "manifest.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        manifest["baseline_run_id"] = run_id
        mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        src = self.run_dir(case_name, run_id) / "report.json"
        dst = self.case_dir(case_name) / "baseline.json"
        import shutil

        shutil.copy2(src, dst)

    def get_baseline(self, case_name: str) -> RunRecord | None:
        bpath = self.case_dir(case_name) / "baseline.json"
        if not bpath.exists():
            return None

        report = json.loads(bpath.read_text(encoding="utf-8"))
        mpath = self.case_dir(case_name) / "manifest.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}

        return RunRecord(
            run_id=manifest.get("baseline_run_id", "unknown"),
            timestamp="baseline",
            case_name=case_name,
            report=report,
            duration_sec=0,
            git_commit=None,
        )

    def get_run(self, case_name: str, run_id: str) -> RunRecord:
        rdir = self.run_dir(case_name, run_id)
        run_meta = json.loads((rdir / "run.json").read_text(encoding="utf-8"))
        report = json.loads((rdir / "report.json").read_text(encoding="utf-8"))
        return RunRecord(
            run_id=run_id,
            timestamp=run_meta["timestamp"],
            case_name=case_name,
            report=report,
            duration_sec=run_meta["duration_sec"],
            git_commit=run_meta.get("git_commit"),
        )

    def list_runs(self, case_name: str) -> list[str]:
        rdir = self.case_dir(case_name) / "runs"
        if not rdir.exists():
            return []
        return sorted(d.name for d in rdir.iterdir() if d.is_dir())

    def get_previous_run(self, case_name: str, current_run_id: str) -> RunRecord | None:
        """Get the run immediately before the current one."""
        mpath = self.case_dir(case_name) / "manifest.json"
        if not mpath.exists():
            return None
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        runs = manifest.get("runs", [])
        try:
            idx = runs.index(current_run_id)
        except ValueError:
            return None
        if idx > 0:
            return self.get_run(case_name, runs[idx - 1])
        return None

    def save_diff(self, case_name: str, run_id: str, diff: DiffReport) -> None:
        dpath = self.run_dir(case_name, run_id) / "diff.json"
        data = {
            "baseline_run_id": diff.baseline_run_id,
            "current_run_id": diff.current_run_id,
            "errors_delta": diff.errors_delta,
            "warnings_delta": diff.warnings_delta,
            "suggestions_delta": diff.suggestions_delta,
            "rule_changes": diff.rule_changes,
            "new_issues": diff.new_issues,
            "fixed_issues": diff.fixed_issues,
            "degraded": diff.degraded,
            "reasons": diff.reasons,
        }
        dpath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
