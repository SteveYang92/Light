from .models import DiffReport, RunRecord


class RegressionChecker:
    def compare(self, baseline: RunRecord, current: RunRecord, thresholds: dict) -> DiffReport:
        b = baseline.report
        c = current.report

        diff = DiffReport(
            baseline_run_id=baseline.run_id,
            current_run_id=current.run_id,
            errors_delta=c.get("errors", 0) - b.get("errors", 0),
            warnings_delta=c.get("warnings", 0) - b.get("warnings", 0),
            suggestions_delta=c.get("suggestions", 0) - b.get("suggestions", 0),
            rule_changes=[],
            new_issues=[],
            fixed_issues=[],
            degraded=False,
            reasons=[],
        )

        max_new_errors = thresholds.get("max_new_errors", 0)
        if diff.errors_delta > max_new_errors:
            diff.degraded = True
            diff.reasons.append(f"Errors increased: {b.get('errors', 0)} → {c.get('errors', 0)} (+{diff.errors_delta})")

        b_by_rule = self._group_by_rule(b.get("issues", []))
        c_by_rule = self._group_by_rule(c.get("issues", []))
        strict_rules = thresholds.get("strict_rules", [])

        for rule in set(b_by_rule) | set(c_by_rule):
            before = len(b_by_rule.get(rule, []))
            after = len(c_by_rule.get(rule, []))
            delta = after - before

            if delta != 0:
                diff.rule_changes.append({"rule": rule, "before": before, "after": after, "delta": delta})

            if delta > 0 and rule in strict_rules:
                diff.degraded = True
                diff.reasons.append(f"Strict rule '{rule}' degraded: {before} → {after}")

        diff.new_issues = self._find_new(b.get("issues", []), c.get("issues", []))
        diff.fixed_issues = self._find_fixed(b.get("issues", []), c.get("issues", []))

        if diff.new_issues:
            diff.reasons.append(f"{len(diff.new_issues)} new issues introduced")

        return diff

    def _group_by_rule(self, issues: list[dict]) -> dict:
        result = {}
        for i in issues:
            rule = i.get("rule", "unknown")
            result.setdefault(rule, []).append(i)
        return result

    def _find_new(self, baseline: list[dict], current: list[dict]) -> list[dict]:
        """Issues in current whose rule has more issues than in baseline."""
        b_by_rule = self._group_by_rule(baseline)
        c_by_rule = self._group_by_rule(current)
        new = []
        for rule, c_issues in c_by_rule.items():
            b_count = len(b_by_rule.get(rule, []))
            excess = len(c_issues) - b_count
            if excess > 0:
                new.extend(c_issues[:excess])
        return new

    def _find_fixed(self, baseline: list[dict], current: list[dict]) -> list[dict]:
        """Issues in baseline whose rule has fewer issues in current."""
        b_by_rule = self._group_by_rule(baseline)
        c_by_rule = self._group_by_rule(current)
        fixed = []
        for rule, b_issues in b_by_rule.items():
            c_count = len(c_by_rule.get(rule, []))
            excess = len(b_issues) - c_count
            if excess > 0:
                fixed.extend(b_issues[:excess])
        return fixed
