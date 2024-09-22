from typing import Callable

import optuna


class HyperparameterOptimizer:
    def __init__(self, objective: Callable, n_trials: int = 100):
        self.objective = objective
        self.n_trials = n_trials

    def optimize(self) -> dict:
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=self.n_trials)
        return study.best_params