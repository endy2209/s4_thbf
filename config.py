from dataclasses import dataclass
import math
from numbers import Integral
import numpy as np


@dataclass(frozen=True)
class StaticTHBFConfig:
    """Static S1 parameters. Defaults follow Table II where applicable."""

    # System dimensions
    Nt: int = 64
    NRF: int = 4
    Ns: int = 4
    K: int = 8
    L: int = 3

    # EM layer: Eq. (3)-(5)
    N_EM: int = 8
    psi_max_deg: float = 45.0
    phi_3db_deg: float = 65.0
    omega_max_db: float = 30.0
    M: int = 360

    # RF layer: Eq. (8)
    B: int = 64
    beta: int = 3

    # Eq. (11), (20), (22)
    snr_db: float = 10.0
    # Implementation normalization only: the paper specifies Pmax/sigma^2.
    sigma2: float = 1.0

    @property
    def Pmax(self) -> float:
        return self.sigma2 * 10.0 ** (self.snr_db / 10.0)

    @property
    def psi_max(self) -> float:
        return float(np.deg2rad(self.psi_max_deg))

    @property
    def phi_3db(self) -> float:
        return float(np.deg2rad(self.phi_3db_deg))


@dataclass(frozen=True)
class S3Config:
    """S3 cost parameters; defaults follow manuscript Table II."""
    n_sym: int = 14
    t_rf: int = 40
    t_em: int = 160
    n_rf: int = 4
    tau_bb: float = 1.0
    tau_rf: float = 32.0
    tau_em: float = 64.0
    lambda_rf: float = 20.0
    lambda_em: float = 100.0

    def __post_init__(self):
        for name in ('n_sym', 't_rf', 't_em', 'n_rf'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.t_em % self.t_rf:
            raise ValueError('t_em must be a multiple of t_rf')
        for name in ('tau_bb', 'tau_rf', 'tau_em', 'lambda_rf', 'lambda_em'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        maximum = self.tau_bb / self.n_sym + (self.tau_rf + self.tau_em) / (self.t_rf * self.n_sym)
        if maximum >= 1:
            raise ValueError('Maximum pilot fraction must be less than 1 (paper constraint)')


# Diagnostic action schedule; not an optimized policy.
S3_INTEGRATION = {'purpose': 'Integration diagnostic only; scripted actions, not a baseline or learned policy.',
 'model': {},
 'trajectory': {'seed': 2299, 'speeds_kmh': 30.0, 'cache_dir': 'data/trajectories'},
 'evaluate_slots': 200,
 'events': [{'t': 0, 'q': 5, 'beams': [34, 22, 40, 35], 'refresh_rf': True},
            {'t': 40, 'q': 5, 'beams': [34, 22, 40, 35], 'refresh_rf': True},
            {'t': 80, 'q': 5, 'beams': [34, 23, 40, 36], 'refresh_rf': True},
            {'t': 160, 'q': 6, 'beams': [34, 23, 40, 36], 'refresh_rf': True}],
 'output_dir': 'outputs/s3_integration',
 'checks': {'tau_scale': 2.0, 'atol': 1e-10}}
