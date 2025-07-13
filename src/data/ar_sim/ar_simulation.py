import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import jit
from sklearn.datasets import make_blobs
from tqdm import tqdm

from src.data.ar_sim.utils import *


@jit(nopython=True)
def get_xt(X_full, W_full, d_vitals, p, cov_matrix, t):
    xt = (
        np.sum(X_full[:, t - p : t] * coef_x(d_vitals, p), axis=-1).reshape(-1) / (p * d_vitals)
        + np.sum(W_full[t - p : t] * coef_x_w(p)) / p
        + eps_x_sample(1, cov_matrix, d_vitals).reshape(-1)
    )

    return xt


@jit(nopython=True)
def get_log_odds_t(X_full, W_full, Y_full, d_vitals, p, treat_imb, t):
    # current xt should affect current treatment wt
    log_odds_t = (
        np.sum(W_full[t - p : t] * coef_w(treat_imb, t, p)) / p
        + np.sum(X_full[:, t - p + 1 : t + 1] * coef_w_x(treat_imb, t, d_vitals, p))
        / (p * d_vitals)
        + np.sum(Y_full[t - p : t] * coef_w_y(treat_imb, t, p)) / p
    )

    return log_odds_t


@jit(nopython=True)
def get_po(X_full, W_full, Y_full, RE, d_vitals, d_random_effects, p, patient_idx, coeff, t, w):

    # current xt should affect current POs
    yt = (
        np.sum(X_full[:, t - p + 1 : t + 1] * coef_y_x(w, p)) / (p * d_vitals)
        + np.sum(RE[patient_idx] * coef_y_re(w, d_random_effects, coeff))
        / d_random_effects
        * np.sum(W_full[t - p : t])
        + np.sum(Y_full[t - p : t] * coef_y(w, p)) / p
        + np.random.normal(0, 0.01)
    )

    return yt


@jit(nopython=True)
def simulate_patient_factuals(
    coeff, t_final, p, d_vitals, d_random_effects, cov_matrix, treat_imb, RE, X, W, Y, patient_idx
):
    # Pre-allocate arrays for a single patient
    X_full = np.zeros((d_vitals, t_final + 1))
    W_full = np.zeros(t_final)
    Y_full = np.zeros(t_final)
    ites = np.zeros(t_final)

    # Initialize with initial values
    X_full[:, :p] = X[patient_idx]
    W_full[:p] = W[patient_idx]
    Y_full[:p] = Y[patient_idx]

    t = p + 1
    while t < t_final:

        xt = get_xt(X_full, W_full, d_vitals, p, cov_matrix, t)
        X_full[:, t] = xt

        # current xt should affect current treatment wt
        log_odds_t = get_log_odds_t(X_full, W_full, Y_full, d_vitals, p, treat_imb, t)
        wt = np.random.binomial(1, p=sigmoid(log_odds_t))
        W_full[t] = wt

        # current xt should affect current POs
        yt_0 = get_po(
            X_full, W_full, Y_full, RE, d_vitals, d_random_effects, p, patient_idx, coeff, t, w=0
        )

        yt_1 = get_po(
            X_full, W_full, Y_full, RE, d_vitals, d_random_effects, p, patient_idx, coeff, t, w=1
        )

        yt = yt_0 * (wt == 0) + yt_1 * (wt == 1)

        ites[t] = yt_1 - yt_0

        Y_full[t] = yt
        t += 1

    # additional sim of xt to get define array of next vitals
    xt = get_xt(X_full, W_full, d_vitals, p, cov_matrix, t_final)
    X_full[:, t_final] = xt

    return X_full, W_full, Y_full, ites


@dataclass
class FactualDataSimulator:
    """_summary_

    Returns:
        _type_: _description_
    """

    num_patients: int = 100
    d_vitals: int = 10
    d_random_effects: int = 10
    seq_length: int = 35
    treat_imb: float = 1.0
    p: int = 8
    cov_rho: float = 0.3
    cov_var: float = 0.2
    re_centers: int = 3
    re_cluster_std: float = 0.4
    coeff: float = 1.2

    def __post_init__(self):
        self.t_final = self.seq_length * 2

        self.sequence_lengths = self.seq_length * np.ones(
            self.num_patients
        )  # No Censorship modeled
        self.cov_matrix = (
            np.ones([self.d_vitals, self.d_vitals]) * self.cov_rho
            + np.identity(self.d_vitals) * (1 - self.cov_rho)
        ) * self.cov_var

        self.X_init, self.W_init, self.Y_init = initialize_data(
            n=self.num_patients, d_x=self.d_vitals, p=self.p, cov_matrix=self.cov_matrix
        )
        centers_list = [np.ones((1, self.d_random_effects)) * i for i in range(self.re_centers)]
        centers = np.concatenate(centers_list, axis=0)
        self.RE, self.re_labels = make_blobs(
            n_samples=self.num_patients,
            n_features=self.d_random_effects,
            centers=centers,
            cluster_std=self.re_cluster_std,
            shuffle=False,
            center_box=(0, 10),
        )

        self.X, self.W, self.Y, self.ites = self.simulate(self.X_init, self.W_init, self.Y_init)
        print("self.X", self.X.shape)

    def simulate(self, X, W, Y):
        results = Parallel(n_jobs=-1)(
            delayed(simulate_patient_factuals)(
                self.coeff,
                self.t_final,
                self.p,
                self.d_vitals,
                self.d_random_effects,
                self.cov_matrix,
                self.treat_imb,
                self.RE,
                X,
                W,
                Y,
                i,
            )
            for i in tqdm(range(self.num_patients), desc="Simulating patients")
        )

        # Extract results
        X_full = np.array([res[0] for res in results])
        W_full = np.array([res[1] for res in results])
        Y_full = np.array([res[2] for res in results])
        ites = np.array([res[3] for res in results])

        print("X_full", X_full.shape)

        return X_full, W_full, Y_full, ites

    def get_simulation_output(self):
        print("self.X in get", self.X.shape)
        outputs = {
            "outcome": self.Y[:, -self.seq_length :],
            "treatment": self.W[:, -self.seq_length :],
            "vitals": self.X[:, :, -self.seq_length - 1 : -1],
            "next_vitals": self.X[:, :, -self.seq_length :],
            "sequence_lengths": self.sequence_lengths,
            "random_effects": self.RE,
            "re_labels": self.re_labels,
            "ITE": self.ites[:, -self.seq_length :],
        }
        return outputs


@jit(nopython=True)
def simulate_patient_numba(
    coeff, t_final, p, d_vitals, d_random_effects, cov_matrix, treat_imb, RE, X, W, Y, patient_idx
):
    # Pre-allocate arrays for a single patient
    X_full = np.zeros((d_vitals, t_final + 1))
    W_full = np.zeros(t_final)
    Y_full = np.zeros(t_final)
    ites = np.zeros(t_final)

    # Initialize with initial values
    for i in range(p):
        X_full[:, i] = X[patient_idx, :, i]
        W_full[i] = W[patient_idx, i]
        Y_full[i] = Y[patient_idx, i]

    t = p + 1
    while t < t_final:
        # Computing xt
        xt = get_xt(X_full, W_full, d_vitals, p, cov_matrix, t)
        X_full[:, t] = xt

        # Computing log odds
        log_odds_t = get_log_odds_t(X_full, W_full, Y_full, d_vitals, p, treat_imb, t)
        wt = np.random.binomial(1, p=sigmoid(log_odds_t))
        W_full[t] = wt

        # Computing yt_0 and yt_1
        yt_0 = get_po(
            X_full, W_full, Y_full, RE, d_vitals, d_random_effects, p, patient_idx, coeff, t, w=0
        )

        yt_1 = get_po(
            X_full, W_full, Y_full, RE, d_vitals, d_random_effects, p, patient_idx, coeff, t, w=1
        )

        yt = yt_0 * (wt == 0) + yt_1 * (wt == 1)
        Y_full[t] = yt

        ites[t] = yt_1 - yt_0
        ycf_t = yt_0 * (wt == 1) + yt_1 * (wt == 0)

        t += 1

    # additional sim of xt to get define array of next vitals
    xt = get_xt(X_full, W_full, d_vitals, p, cov_matrix, t_final)
    X_full[:, t_final] = xt

    # Convention: Last value is cf
    Y_full[t_final - 1] = ycf_t
    W_full[t_final - 1] = 1 - wt

    return X_full, W_full, Y_full, ites


@dataclass
class CounterFactualDataSimulator:
    """_summary_

    Returns:
        _type_: _description_
    """

    num_patients: int = 100
    d_vitals: int = 10
    d_random_effects: int = 10
    seq_length: int = 35
    treat_imb: float = 1.0
    p: int = 8
    cov_rho: float = 0.3
    cov_var: float = 0.2
    re_centers: int = 3
    re_cluster_std: float = 0.4
    num_treatments: int = 1
    coeff: float = 1.2

    def __post_init__(self):

        self.num_test_points = self.num_patients * self.seq_length * self.num_treatments

        self.t_final = self.seq_length * 2
        self.sequence_lengths = self.seq_length * np.ones(
            self.num_test_points
        )  # No Censorship modeled

        self.cov_matrix = (
            np.ones([self.d_vitals, self.d_vitals]) * self.cov_rho
            + np.identity(self.d_vitals) * (1 - self.cov_rho)
        ) * self.cov_var
        self.X_init, self.W_init, self.Y_init = initialize_data(
            n=self.num_test_points, d_x=self.d_vitals, p=self.p, cov_matrix=self.cov_matrix
        )
        self.ites = np.zeros((self.num_test_points, self.t_final))
        # Make sure the original arrays are writable
        self.ites.setflags(write=True)

        centers_list = [np.ones((1, self.d_random_effects)) * i for i in range(self.re_centers)]
        centers = np.concatenate(centers_list, axis=0)
        self.RE, self.re_labels = make_blobs(
            n_samples=self.num_test_points,
            n_features=self.d_random_effects,
            centers=centers,
            cluster_std=self.re_cluster_std,
            shuffle=False,
            center_box=(0, 10),
        )

        self.X, self.W, self.Y, self.ites = self.simulate(self.X_init, self.W_init, self.Y_init)

    def simulate(self, X, W, Y):
        results = Parallel(n_jobs=-1)(
            delayed(simulate_patient_numba)(
                self.coeff,
                self.t_final,
                self.p,
                self.d_vitals,
                self.d_random_effects,
                self.cov_matrix,
                self.treat_imb,
                self.RE,
                X,
                W,
                Y,
                i,
            )
            for i in tqdm(
                range(self.num_test_points), desc="Simulating exploded patients trajectories"
            )
        )

        # Extract results
        X_full = np.array([res[0] for res in results])
        W_full = np.array([res[1] for res in results])
        Y_full = np.array([res[2] for res in results])
        ites = np.array([res[3] for res in results])

        return X_full, W_full, Y_full, ites

    def get_simulation_output(self):
        outputs = {
            "outcome": self.Y[:, -self.seq_length :],
            "treatment": self.W[:, -self.seq_length :],
            "vitals": self.X[:, :, -self.seq_length - 1 : -1],
            "next_vitals": self.X[:, :, -self.seq_length :],
            "sequence_lengths": self.sequence_lengths,
            "random_effects": self.RE,
            "re_labels": self.re_labels,
            "ITE": self.ites[:, -self.seq_length :],
        }
        return outputs


@dataclass
class CounterFactualDataSimulator_seq:
    """
    Counterfactual Data Simulator for sequential data.

    Attributes:
        num_patients: Number of patients to simulate.
        d_vitals: Number of vital signs.
        d_random_effects: Number of random effects.
        seq_length: Length of the sequence.
        treat_imb: Treatment imbalance factor.
        p: Window size for projections.
        projection_horizon: Number of steps to project into the future.
        cov_rho: Covariance parameter.
        cov_var: Variance parameter.
        re_centers: Number of centers for random effect clusters.
        re_cluster_std: Standard deviation for random effect clusters.
        cf_seq_mode: Mode for generating treatment options.
    """

    num_patients: int = 100
    d_vitals: int = 10
    d_random_effects: int = 10
    seq_length: int = 35
    treat_imb: float = 1.0
    p: int = 8
    projection_horizon: int = 3
    cov_rho: float = 0.3
    cov_var: float = 0.2
    re_centers: int = 3
    re_cluster_std: float = 0.4
    coeff: float = 1.2
    cf_seq_mode: str = "sliding_treatment"

    def __post_init__(self):
        self.generate_treatment_options()
        self.num_test_points = self.num_patients * self.seq_length * len(self.treatment_options)
        self.sequence_lengths = self.seq_length * np.ones(self.num_test_points)
        self.t_final = self.seq_length * 2 + 1
        self.cov_matrix = (
            np.ones([self.d_vitals, self.d_vitals]) * self.cov_rho
            + np.identity(self.d_vitals) * (1 - self.cov_rho)
        ) * self.cov_var
        self.X_init, self.W_init, self.Y_init = initialize_data(
            n=self.num_test_points, d_x=self.d_vitals, p=self.p, cov_matrix=self.cov_matrix
        )
        self.outcomes = np.zeros((self.num_test_points, self.t_final))
        self.vitals = np.zeros((self.num_test_points, self.d_vitals, self.t_final))
        self.treatments = np.zeros((self.num_test_points, self.t_final))
        centers_list = [np.ones((1, self.d_random_effects)) * i for i in range(self.re_centers)]
        centers = np.concatenate(centers_list, axis=0)
        self.RE, self.re_labels = make_blobs(
            n_samples=self.num_patients,
            n_features=self.d_random_effects,
            centers=centers,
            cluster_std=self.re_cluster_std,
            shuffle=False,
            center_box=(0, 10),
        )
        self.RE = np.expand_dims(self.RE, axis=1)
        self.RE = np.repeat(self.RE, self.seq_length * len(self.treatment_options), axis=1)
        self.RE = self.RE.reshape(-1, self.d_random_effects, 1)

    def simulate(self):
        results = Parallel(n_jobs=-1)(delayed(self.simulate_step)(t) for t in range(self.t_final))
        for X, W, Y in results:
            self.X_init, self.W_init, self.Y_init = X, W, Y

    def simulate_step(self, t):
        if t == self.t_final or t + 1 + self.projection_horizon + 1 > self.t_final:
            return self.X_init, self.W_init, self.Y_init

        xt = (
            np.expand_dims(
                np.mean(self.X_init[:, :, -self.p :] * coef_x(self.d_vitals, self.p), axis=-1), -1
            )
            + np.matmul(self.W_init[:, -self.p :], coef_x_w(self.p)).reshape(-1, 1, 1) / self.p
            + eps_x_sample(self.num_test_points, self.cov_matrix, self.d_vitals)
        )
        X = np.concatenate((self.X_init, xt), axis=2)

        log_odds_t = (
            np.matmul(self.W_init[:, -self.p :], coef_w(self.treat_imb, t, self.p)).reshape(-1, 1)
            / self.p
            + np.expand_dims(
                np.mean(
                    X[:, :, -self.p :] * coef_w_x(self.treat_imb, t, self.d_vitals, self.p),
                    axis=(1, 2),
                ),
                axis=-1,
            )
            + np.matmul(self.Y_init[:, -self.p :], coef_w_y(self.treat_imb, t, self.p)).reshape(
                -1, 1
            )
            / self.p
        )
        wt = np.random.binomial(1, p=sigmoid(log_odds_t)).reshape(-1, 1)
        W = np.concatenate((self.W_init, wt), axis=1)

        yt_0 = (
            np.matmul(
                X[:, :, -self.p :] * self.RE, coef_y_re(0, self.p, self.coeff).reshape(-1, 1)
            ).mean(axis=-2)
            + np.matmul(X[:, :, -self.p :], coef_y_x(0, self.p).reshape(-1, 1)).mean(axis=-2)
            + np.matmul(W[:, -self.p : -1], coef_y_w(0, self.p - 1)).reshape(-1, 1) / self.p
            + np.matmul(self.Y_init[:, -self.p :], coef_y(0, self.p)).reshape(-1, 1) / self.p
            + np.random.normal(0, 0.01, size=(self.num_test_points, 1))
        )

        yt_1 = (
            np.matmul(
                X[:, :, -self.p :] * self.RE, coef_y_re(1, self.p, self.coeff).reshape(-1, 1)
            ).mean(axis=-2)
            + np.matmul(X[:, :, -self.p :], coef_y_x(1, self.p).reshape(-1, 1)).mean(axis=-2)
            + np.matmul(W[:, -self.p : -1], coef_y_w(1, self.p - 1)).reshape(-1, 1) / self.p
            + np.matmul(self.Y_init[:, -self.p :], coef_y(1, self.p)).reshape(-1, 1) / self.p
            + np.random.normal(0, 0.01, size=(self.num_test_points, 1))
        )
        yt = yt_0 * (wt == 0).astype(int) + yt_1 * (wt == 1).astype(int)
        Y = np.concatenate((self.Y_init, yt), axis=1)

        return X, W, Y

    def get_simulation_output(self):
        outputs = {
            "outcome": self.outcomes[:, :-1],
            "treatment": self.treatments[:, :-1],
            "vitals": self.vitals[:, :, :-1],
            "next_vitals": self.vitals[:, :, 1:],
            "sequence_lengths": self.sequence_lengths,
            "random_effects": self.RE,
            "re_labels": self.re_labels,
        }
        return outputs

    def generate_treatment_options(self):
        if self.cf_seq_mode == "sliding_treatment":
            self.treatment_options = np.stack(
                [
                    np.eye(self.projection_horizon, dtype=int),
                    np.zeros((self.projection_horizon, self.projection_horizon), dtype=int),
                ],
                axis=-1,
            )
        elif self.cf_seq_mode == "random_trajectories":
            self.treatment_options = np.random.randint(
                0, 2, (self.projection_horizon * 2, self.projection_horizon, 1)
            )
        else:
            raise NotImplementedError(f"Mode '{self.cf_seq_mode}' is not implemented.")
