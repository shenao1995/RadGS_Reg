"""User targets for paired reconstruction/registration evaluation (no model imports)."""
import math


GOAL_SCHEMA = 'paired_volume_ssim_relative_1pct_v1'
MIN_RELATIVE_SSIM_GAIN = .01


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def gate(metrics, baseline=None):
    """Require a 1% relative gain in full-volume 3D SSIM and all six axes.

    Foreground improvement is retained to reject gains caused only by background.
    Missing/nonfinite paired baseline evidence cannot establish improvement.
    """
    baseline = baseline or {}
    value = metrics.get('reconstruction_ssim3d')
    reference = baseline.get('reconstruction_ssim3d')
    valid = finite(value) and finite(reference) and 0 < reference <= 1 and 0 <= value <= 1
    delta = value - reference if valid else None
    relative = delta / reference if valid else None
    foreground = metrics.get('reconstruction_ssim3d_foreground_roi')
    foreground_reference = baseline.get('reconstruction_ssim3d_foreground_roi')
    criteria = {
        'paired_full_volume_ssim_gain_at_least_1percent': bool(valid and value >= reference * (1 + MIN_RELATIVE_SSIM_GAIN)),
        'foreground_ssim_improved': bool(finite(foreground) and finite(foreground_reference) and foreground > foreground_reference),
    }
    for axis in 'xyz':
        for kind, unit, label in [('translation', 'mm', 'below_1mm'), ('rotation', 'deg', 'below_1degree')]:
            error = metrics.get(f'{kind}_{axis}_{unit}')
            criteria[f'{kind}_{axis}_{label}'] = bool(finite(error) and 0 <= error < 1)
    return dict(schema=GOAL_SCHEMA, passed=all(criteria.values()), criteria=criteria,
                full_volume_ssim_baseline=reference, full_volume_ssim_joint=value,
                full_volume_ssim_absolute_gain=delta,
                full_volume_ssim_relative_gain=relative,
                full_volume_ssim_required=reference * (1 + MIN_RELATIVE_SSIM_GAIN) if valid else None,
                full_volume_ssim_absolute_0_01_gain_met=bool(valid and value >= reference + .01))


def verified_test_goal(status, expected_cases=431):
    """Recompute targets; an old success flag alone must never stop experiments."""
    test = status.get('test') or {}
    results = test.get('results') or {}
    joint, baseline = results.get('joint') or {}, results.get('baseline') or {}
    return bool(status.get('status') == 'metrics_verified_on_test'
                and test.get('split') == 'test' and test.get('full_split_evaluated')
                and not test.get('partial_evaluation', False)
                and not test.get('diagnostic_only', False)
                and joint.get('cases') == baseline.get('cases') == expected_cases
                and gate(joint.get('metrics') or {}, baseline.get('metrics') or {})['passed'])
