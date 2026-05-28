import numpy as np
import time
import os
import json
from arrays.models import UniformLinearArray
from beams.synthesis import BeamSynthesizer
from sensing.measurements import MeasurementModel
from inference.belief import GridBelief, BayesianUpdater
from planners.adaptive import ExhaustiveSweepPlanner, GreedyEntropyPlanner, RandomProbingPlanner, HierarchicalNarrowingPlanner
from utils.propagation import friis_path_loss_linear, friis_path_loss_db, dbm_to_linear

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

        noise_dbm = self.config.get('noise_dbm', -90)
        noise_watts = dbm_to_linear(noise_dbm)

        tx_power_dbm = self.config.get('tx_power_dbm', 10)
        tx_watts = dbm_to_linear(tx_power_dbm)
        distance_m = self.config.get('distance_m', 100)

        model = MeasurementModel(ula, noise_power_dbm=noise_dbm)
        updater = BayesianUpdater(ula, noise_watts)
        belief = GridBelief(angles)

        true_angle = self.config.get('true_angle', 20.0)
        strategy = self.config.get('planner', 'greedy')

        # For planners we need to pass attenuation logic, but we can wrap it
        attenuation = friis_path_loss_linear(distance_m, ula.wavelength)

        if strategy == 'exhaustive':
            candidates = np.linspace(-90, 90, 31)
            planner = ExhaustiveSweepPlanner(candidates)
        elif strategy == 'random':
            planner = RandomProbingPlanner((-90, 90))
        elif strategy == 'hierarchical':
            planner = HierarchicalNarrowingPlanner((-90, 90))
        else:
            candidates = np.linspace(-90, 90, 31)
            # Greedy planner originally needed tx_lin, pl_lin. We pass tx_watts and 1/attenuation as pl_lin
            # to match legacy signature, or we can just pass them directly.
            planner = GreedyEntropyPlanner(candidates, updater, ula, tx_watts, 1.0/attenuation, angles)

        max_probes = self.config.get('max_probes', 50)
        entropy_thresh = self.config.get('entropy_threshold', 1.0)
        quantization_bits = self.config.get('quantization_bits', 0)

        probes_used = 0
        entropies = [belief.get_entropy()]
        locked = False

        sharpness_list = []
        signal_noise_ratios = []

        for _ in range(max_probes):
            if belief.get_entropy() < entropy_thresh:
                locked = True
                break

            b_type, center, width = planner.get_next_beam(belief.probs, angles)

            if b_type == 'pencil':
                w = synth.synthesize_pencil_beam(center)
            elif b_type == 'sector':
                w = synth.synthesize_sector_beam_ls(center - width/2, center + width/2)
            else:
                w = synth.synthesize_pencil_beam(center)

            if quantization_bits > 0:
                w = ula.apply_phase_quantization(w, bits=quantization_bits)

            meas = model.measure(true_angle, w, tx_power_dbm, distance_m, fading_type=self.config.get('fading_type', 'los'))

            # Record diagnostics
            signal_noise_ratios.append(meas["snr_db"])

            likelihoods = updater.compute_likelihood(meas["measured_power_watts"], w, angles, tx_watts, distance_m)

            # Sharpness metric
            mean_lh = np.mean(likelihoods)
            sharpness = np.max(likelihoods) / mean_lh if mean_lh > 0 else 1.0
            sharpness_list.append(sharpness)

            belief.update(likelihoods)
            entropies.append(belief.get_entropy())
            probes_used += 1

        end_time = time.time()

        final_estimate = angles[np.argmax(belief.probs)]
        error = np.abs(final_estimate - true_angle)
        false_convergence = locked and (error > 5.0)

        self.results = {
            "locked": locked,
            "probes_used": probes_used,
            "final_entropy": float(belief.get_entropy()),
            "time_to_lock_sec": end_time - start_time,
            "false_convergence": bool(false_convergence),
            "estimation_error": float(error),
            "avg_snr_db": float(np.mean(signal_noise_ratios)),
            "avg_likelihood_sharpness": float(np.mean(sharpness_list))
        }

        return self.results

    def save_results(self, filename="reports/experiment_results.json"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)
