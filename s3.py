"""THBF S3: pilot and switching accounting, paper (4), equations (27)-(31).

Accounting formulas plus S1/S2 integration and PDF checks. Configuration is in config.py.
Run python s3.py to produce metrics and check results. Importing does not run simulations.
RF/EM pilot flags persist for an RF period; switching is charged per event.
"""
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Sequence
from config import S3Config as Config
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from dynamic_channel import DynamicTHBFConfig, load_or_generate_trajectory
from static_model import run_fixed_configuration
from config import S3_INTEGRATION

def _flag(value):
    if value not in (0, 1):
        raise ValueError('Update flag must be 0 or 1')
    return int(value)


def pilot_cost_bb(cfg: Config) -> float:
    return cfg.tau_bb / cfg.n_sym


def pilot_cost_rf(u_rf_period: bool, cfg: Config) -> float:
    """Use the CHARGEABLE flag at the start of the current RF period."""
    return _flag(u_rf_period) * cfg.tau_rf / (cfg.t_rf * cfg.n_sym)


def pilot_cost_em(u_em_period: bool, cfg: Config) -> float:
    """EM overhead is spread over t_rf slots, NOT t_em slots."""
    return _flag(u_em_period) * cfg.tau_em / (cfg.t_rf * cfg.n_sym)


def switch_cost_em(q_old: int, q_new: int) -> float:
    return float(q_old != q_new)


def switch_cost_rf(beams_old: Sequence[int], beams_new: Sequence[int]) -> float:
    if len(beams_old) == 0 or len(beams_old) != len(beams_new):
        raise ValueError('Beam vectors must have equal nonzero lengths')
    return sum(int(a != b) for a, b in zip(beams_old, beams_new)) / len(beams_old)


def net_rate(gross_rate: float, pilot_fraction: float) -> float:
    if not math.isfinite(gross_rate) or gross_rate < 0:
        raise ValueError('Gross SE must be finite and nonnegative')
    if not math.isfinite(pilot_fraction) or not 0 <= pilot_fraction < 1:
        raise ValueError('Pilot fraction must be in [0, 1)')
    return (1 - pilot_fraction) * gross_rate


def utility(rate_net: float, cost_em: float, cost_rf: float, cfg: Config) -> float:
    # Negative utility is valid. Never clip it to zero.
    return rate_net - cfg.lambda_em * cost_em - cfg.lambda_rf * cost_rf


@dataclass(frozen=True)
class SlotResult:
    t: int
    q: int
    beams: tuple
    initialized: bool
    refresh_rf: bool
    update_em: bool
    pilot_bb: float
    pilot_rf: float
    pilot_em: float
    pilot_total: float
    switch_em: float
    switch_rf: float
    gross_se: float
    net_se: float
    utility: float


class S3Accounting:
    """Call step exactly once for EVERY slot, starting at t=0.

    q and beams are the APPLIED configuration for that slot. refresh_rf says
    whether RF training happens now (not whether it happened earlier).
    At t=0 the supplied configuration is initialization: all pilot and
    switching costs are excluded from accounting, as specified in the paper.
    Create a new instance or call reset() for each trajectory.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self._next_t = 0
        self._q = None
        self._beams = None
        self._period_rf = False
        self._period_em = False

    def step(self, t: int, gross_se: float, q: int,
             beams: Sequence[int], refresh_rf: bool = False) -> SlotResult:
        cfg = self.cfg
        if isinstance(t, bool) or not isinstance(t, Integral) or t != self._next_t:
            raise ValueError(f'Expected consecutive slot {self._next_t}, received {t}')
        if isinstance(q, bool) or not isinstance(q, Integral) or q < 0:
            raise ValueError('q must be a nonnegative integer state ID')
        beams = tuple(beams)  # Snapshot: later mutations by caller cannot change history.
        if len(beams) != cfg.n_rf or len(set(beams)) != cfg.n_rf:
            raise ValueError('Need n_rf ordered, distinct beam IDs')
        if any(isinstance(b, bool) or not isinstance(b, Integral) or b < 0 for b in beams):
            raise ValueError('Beam IDs must be nonnegative integers')
        refresh_rf = bool(_flag(refresh_rf))
        initialized = t == 0
        update_em = not initialized and q != self._q
        c_em = 0.0 if initialized else switch_cost_em(self._q, q)
        c_rf = 0.0 if initialized else switch_cost_rf(self._beams, beams)

        if not initialized:
            if refresh_rf and t % cfg.t_rf:
                raise ValueError('RF refresh is allowed only at RF candidate events')
            if update_em and t % cfg.t_em:
                raise ValueError('EM change is allowed only at EM candidate events')
            if update_em and not refresh_rf:
                raise ValueError('EM change requires RF refresh; set refresh_rf=True')
            if c_rf and not refresh_rf:
                raise ValueError('Beam changes require RF refresh')

        period_rf, period_em = self._period_rf, self._period_em
        if t % cfg.t_rf == 0:
            period_rf = refresh_rf and not initialized
            period_em = update_em
        p_bb = 0.0 if initialized else pilot_cost_bb(cfg)
        p_rf = pilot_cost_rf(period_rf, cfg)
        p_em = pilot_cost_em(period_em, cfg)
        total = p_bb + p_rf + p_em
        r_net = net_rate(gross_se, total)
        result = SlotResult(
            t, int(q), beams, initialized, refresh_rf or initialized, update_em,
            p_bb, p_rf, p_em, total, c_em, c_rf, float(gross_se), r_net,
            utility(r_net, c_em, c_rf, cfg))
        # Commit only after validation: a failed call does not corrupt the state.
        self._next_t = t + 1
        self._q, self._beams = q, beams
        self._period_rf, self._period_em = period_rf, period_em
        return result

def evaluate_trajectory(trajectory, model_cfg, accounting_cfg, events, slots):
    """Yield S1 outputs + S3 cost for consecutive slots of an existing S2 trace.

    IDs follow S1: q=1..N_EM, beam=1..B. Each event contains the applied q,
    ordered beams, and whether RF training occurs. Between events, hold q/b.
    This function never generates another trajectory or chooses a configuration.
    """
    if model_cfg.NRF != accounting_cfg.n_rf:
        raise ValueError('S1 NRF must equal S3 n_rf')
    gains = np.asarray(trajectory['alpha'])
    aods = np.asarray(trajectory['aods'])
    if gains.ndim != 3 or gains.shape != aods.shape:
        raise ValueError('S2 alpha and aods must have matching (T,K,L) shapes')
    if gains.shape[1:] != (model_cfg.K, model_cfg.L):
        raise ValueError('S2 dimensions must agree with model K,L')
    if type(slots) is not int or not 1 <= slots <= len(gains):
        raise ValueError('evaluate_slots must be a positive integer within trajectory length')
    event_map = {}
    for event in events:
        if set(event) != {'t', 'q', 'beams', 'refresh_rf'}:
            raise ValueError('Each event needs exactly t, q, beams, refresh_rf')
        t = event['t']
        if type(t) is not int or not 0 <= t < slots or t in event_map:
            raise ValueError('Event times must be unique integer slots in the evaluated range')
        if type(event['q']) is not int or not 1 <= event['q'] <= model_cfg.N_EM:
            raise ValueError('S1 q IDs must be 1..N_EM')
        if any(type(b) is not int or not 1 <= b <= model_cfg.B for b in event['beams']):
            raise ValueError('S1 beam IDs must be 1..B')
        if type(event['refresh_rf']) is not bool:
            raise ValueError('refresh_rf must be a JSON boolean')
        event_map[t] = event
    if 0 not in event_map:
        raise ValueError('An initialization event at t=0 is required')

    # Validate the complete action schedule cheaply before expensive S1 work.
    check = S3Accounting(accounting_cfg)
    configurations = []
    for t in range(slots):
        event = event_map.get(t)
        if event is not None:
            q, beams = event['q'], tuple(event['beams'])
        refresh = event['refresh_rf'] if event is not None else False
        check.step(t, 0.0, q, beams, refresh)
        configurations.append((q, beams, refresh))

    accounting = S3Accounting(accounting_cfg)
    for t, (q, beams, refresh) in enumerate(configurations):
        # Applied configuration and current physical channel are identical in
        # the S1 gross-rate calculation and S3 cost calculation.
        physical = run_fixed_configuration(model_cfg, gains[t], aods[t], q, beams)
        result = accounting.step(t, physical['gross_se'], q, beams, refresh)
        row = asdict(result)
        row['time_s'] = float(t * model_cfg.Ts)
        row['selected_users'] = [int(k) for k in physical['selected_users']]
        row['sinr'] = [float(x) for x in physical['sinr']]
        yield row


def replay_accounting(rows, cfg, omit_refresh_at=None):
    """Reuse exactly the same S1 rates and applied actions; never rerun the channel."""
    accounting = S3Accounting(cfg)
    return [asdict(accounting.step(
        r['t'], r['gross_se'], r['q'], r['beams'],
        False if r['t'] == omit_refresh_at else r['refresh_rf'])) for r in rows]


def check_s3(rows, cfg, tau_scale=2.0, atol=1e-10):
    """PDF page 11, S3 acceptance checks. Missing event coverage is a failure.

    Initialization exemption covers all pilot acquisition and switching costs.
    Explicit booleans are used so python -O cannot skip checks.
    """
    if not rows or not np.isfinite(tau_scale) or tau_scale <= 1:
        raise ValueError('Need nonempty rows and finite tau_scale > 1')
    if not np.isfinite(atol) or atol < 0:
        raise ValueError('atol must be finite and nonnegative')
    reports = {}
    zero = replay_accounting(rows, replace(cfg, tau_bb=0, tau_rf=0, tau_em=0))
    error = max(abs(r['net_se'] - r['gross_se']) for r in zero)
    reports['zero_pilots'] = {'max_abs_net_minus_gross': error, 'passed': error <= atol}
    for field in ('tau_rf', 'tau_em'):
        original = getattr(cfg, field)
        if original == 0:
            raise ValueError(f'{field} must be positive for the increase test')
        increased = original * tau_scale
        alternative = replay_accounting(rows, replace(cfg, **{field: increased}))
        delta = [b['net_se'] - a['net_se'] for a, b in zip(rows, alternative)]
        reports['larger_' + field] = {
            'original_tau': original, 'increased_tau': increased,
            'min_net_se_change': min(delta), 'max_net_se_change': max(delta),
            'passed': max(delta) <= atol,
        }

    # Defaults must be checked independently of any custom cost settings.
    defaults = Config()
    if (cfg.t_rf, cfg.t_em, cfg.n_rf) != (defaults.t_rf, defaults.t_em, defaults.n_rf):
        raise ValueError('PDF default-fraction replay needs default RF/EM periods and NRF')
    default_rows = replay_accounting(rows, defaults)
    categories = {
        'without_refresh': [r for r in default_rows
                            if not r['initialized'] and r['pilot_rf'] == 0 and r['pilot_em'] == 0],
        'with_rf_refresh': [r for r in default_rows
                            if not r['initialized'] and r['pilot_rf'] > 0 and r['pilot_em'] == 0],
        'with_em_update': [r for r in default_rows
                           if not r['initialized'] and r['pilot_em'] > 0],
    }
    expected = {'without_refresh': 1/14, 'with_rf_refresh': 9/70, 'with_em_update': 17/70}
    fractions = {name: sample[0]['pilot_total'] if sample else None
                 for name, sample in categories.items()}
    reports['default_pilot_fractions'] = {
        **fractions, 'expected_rounded': [0.0714, 0.1286, 0.2429],
        'passed': all(bool(sample) and all(abs(r['pilot_total'] - expected[name]) <= atol
                      for r in sample) for name, sample in categories.items()),
    }
    unchanged = [r for previous, r in zip(rows, rows[1:])
                 if r['refresh_rf'] and r['beams'] == previous['beams']]
    reports['refresh_without_beam_switch'] = {
        'slots': [r['t'] for r in unchanged],
        'switch_rf': [r['switch_rf'] for r in unchanged],
        'passed': bool(unchanged) and all(abs(r['switch_rf']) <= atol for r in unchanged),
    }
    em = [r for previous, r in zip(rows, rows[1:]) if r['q'] != previous['q']]
    rejected = False
    if em:
        try:
            replay_accounting(rows[:em[0]['t'] + 1], cfg, omit_refresh_at=em[0]['t'])
        except ValueError as exc:
            rejected = 'EM change requires RF refresh' in str(exc)
    reports['em_requires_rf_refresh'] = {
        'em_slots': [r['t'] for r in em], 'em_without_rf_rejected': rejected,
        'passed': bool(em) and all(r['refresh_rf'] for r in em) and rejected,
    }
    initial = rows[:cfg.t_rf]
    reports['initialization'] = {
        'slots_checked': len(initial),
        'initialization_pilot_total': rows[0]['pilot_total'],
        'passed': len(initial) == cfg.t_rf and
        abs(rows[0]['pilot_total']) <= atol and
        all(
            all(abs(r[key]) <= atol for key in ('pilot_rf', 'pilot_em', 'switch_rf', 'switch_em'))
            and (abs(r['pilot_bb']) <= atol if r['initialized'] else
                 abs(r['pilot_bb'] - cfg.tau_bb/cfg.n_sym) <= atol)
            for r in initial),
    }
    return reports


def main():
    spec = S3_INTEGRATION
    root = Path(__file__).resolve().parent
    model_cfg = DynamicTHBFConfig(**spec['model'])
    accounting_cfg = Config()
    settings = spec['trajectory']
    trajectory, _, _ = load_or_generate_trajectory(
        model_cfg, seed=settings['seed'], speeds_kmh=settings['speeds_kmh'],
        cache_dir=root / settings['cache_dir'])
    rows = list(evaluate_trajectory(trajectory, model_cfg, accounting_cfg,
                                    spec['events'], spec['evaluate_slots']))
    checks = check_s3(rows, accounting_cfg, **spec['checks'])
    out = root / spec['output_dir']
    out.mkdir(parents=True, exist_ok=True)
    report_metrics = {
        'slots_evaluated': len(rows),
        'seed': int(settings['seed']),
        'speeds_kmh': np.asarray(trajectory['speeds_kmh']).tolist(),
        'mean_gross_se': float(np.mean([r['gross_se'] for r in rows])),
        'mean_net_se': float(np.mean([r['net_se'] for r in rows])),
        'mean_utility': float(np.mean([r['utility'] for r in rows])),
        'mean_pilot_fraction': float(np.mean([r['pilot_total'] for r in rows])),
        'rf_refreshes_excluding_initialization': sum(r['refresh_rf'] and not r['initialized'] for r in rows),
        'em_switches': sum(r['update_em'] for r in rows),
        'changed_rf_chain_count': sum(int(round(r['switch_rf'] * accounting_cfg.n_rf)) for r in rows),
    }
    groups = {
        "simulation": ["slots_evaluated", "seed", "speeds_kmh"],
        "spectral_efficiency": ["mean_gross_se", "mean_net_se"],
        "pilot": ["mean_pilot_fraction"],
        "switching": ["rf_refreshes_excluding_initialization", "em_switches",
                      "changed_rf_chain_count"],
        "utility": ["mean_utility"],
    }
    sections = [(title, {key: report_metrics[key] for key in keys})
                for title, keys in groups.items()]
    sections.extend(checks.items())
    report = '\n\n'.join(f'[{title}]\n{values}' for title, values in sections) + '\n'
    (out / 'output.txt').write_text(report, encoding='utf-8')
    print(report)

if __name__ == '__main__':
    main()
