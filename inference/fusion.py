import numpy as np

class CooperativeFusionEngine:
    def __init__(self, num_panels: int, grid_angles: np.ndarray):
        self.num_panels = num_panels
        self.grid_angles = grid_angles
        self.num_states = len(grid_angles)

    def centralized_fusion(self, local_likelihoods: list) -> np.ndarray:
        """
        Perform naive centralized fusion assuming conditional independence.
        local_likelihoods: list of likelihood arrays from each panel.
        Returns the joint likelihood update.
        """
        assert len(local_likelihoods) == self.num_panels

        joint_likelihood = np.ones(self.num_states)
        for lh in local_likelihoods:
            joint_likelihood *= lh

        # Normalize to avoid underflow/overflow issues in relative scale
        max_lh = np.max(joint_likelihood)
        if max_lh > 0:
            joint_likelihood /= max_lh

        return joint_likelihood

    def distributed_consensus(self, local_beliefs: list, consensus_steps: int = 1) -> list:
        """
        Perform distributed consensus via linear pooling.
        local_beliefs: list of probability arrays from each panel.
        Returns the updated list of beliefs.
        """
        assert len(local_beliefs) == self.num_panels

        beliefs = np.array(local_beliefs) # shape (num_panels, num_states)

        # Simple all-to-all topology for consensus
        adjacency = np.ones((self.num_panels, self.num_panels)) / self.num_panels

        for _ in range(consensus_steps):
            beliefs = adjacency @ beliefs

        return [b for b in beliefs]
