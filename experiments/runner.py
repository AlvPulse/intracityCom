import numpy as np
import time
import os
import json
from arrays.models import UniformLinearArray
from beams.synthesis import BeamSynthesizer
from sensing.measurements import MeasurementModel
from inference.belief import GridBelief, BayesianUpdater
from planners.adaptive import ExhaustiveSweepPlanner, GreedyEntropyPlanner

class ExperimentRunner:
    def __init__(self, config: dict):
        self.config = config
        self.results = {}

    def run_single_discovery(self):
        start_time = time.time()

        # Setup
        ula = UniformLinearArray(num_elements=self.config.get('num_elements', 16))
        synth = BeamSynthesizer(ula)
        angles = np.linspace(-90, 90, self.config.get('num_grid_points', 181))

        noise_dbm = self.config.get('noise_dbm', -100)
        noise_lin = 10**(noise_dbm/10)

        tx_power = self.config.get('tx_power_dbm', 10)
        path_loss = self.config.get('path_loss_db', 70)
        tx_lin = 10**(tx_power/10)
        pl_lin = 10**(path_loss/10)

        model = MeasurementModel(ula, noise_power_dbm=noise_dbm)
        updater = BayesianUpdater(ula, noise_lin)
        belief = GridBelief(angles)

        true_angle = self.config.get('true_angle', 20.0)
        strategy = self.config.get('planner', 'greedy')

        if strategy == 'exhaustive':
            candidates = np.linspace(-90, 90, 31)
            planner = ExhaustiveSweepPlanner(candidates)
        else:
            candidates = np.linspace(-90, 90, 31)
            planner = GreedyEntropyPlanner(candidates, updater, ula, tx_lin, pl_lin, angles)

        max_probes = self.config.get('max_probes', 50)
        entropy_thresh = self.config.get('entropy_threshold', 1.0)

        probes_used = 0
        entropies = [belief.get_entropy()]
        locked = False

        for _ in range(max_probes):
            if belief.get_entropy() < entropy_thresh:
                locked = True
                break

            b_type, ang, _ = planner.get_next_beam(belief.probs)
            w = synth.synthesize_pencil_beam(ang)

            meas = model.measure(true_angle, w, tx_power, path_loss, 'los')
            likelihoods = updater.compute_likelihood(meas["measured_power"], w, angles, tx_lin, pl_lin)
            belief.update(likelihoods)

            entropies.append(belief.get_entropy())
            probes_used += 1

        end_time = time.time()

        # Determine success
        final_estimate = angles[np.argmax(belief.probs)]
        error = np.abs(final_estimate - true_angle)
        false_convergence = locked and (error > 5.0)

        self.results = {
            "locked": locked,
            "probes_used": probes_used,
            "final_entropy": float(belief.get_entropy()),
            "time_to_lock_sec": end_time - start_time,
            "false_convergence": bool(false_convergence),
            "estimation_error": float(error)
        }

        return self.results

    def save_results(self, filename="reports/experiment_results.json"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)
