import copy
from typing import Any, Dict, Optional, Tuple

import torch
from torch_geometric.data import Data

from ..al.selector import compute_al_scores, select_topk_candidates
from ..utils.graph_ops import (
    apply_isolate_node,
    build_soft_edge_weight,
    node_incident_edges,
    weaken_edges,
    zero_edges,
)


class GraphDefenseEnv:
    """
    Paper-style defense environment with AL candidates + constrained action budget.

    State: top-k candidate features + global features
    Action: action_type x candidate_index
    Reward: -lambda_asr * ΔASR + lambda_acc * ΔACC - lambda_cost * action_cost
    Constraint: total editing cost <= budget
    """

    def __init__(
        self,
        poisoned_data: Data,
        model,
        device: str = "cpu",
        topk: int = 32,
        lambda_asr: float = 1.0,
        lambda_acc: float = 0.5,
        lambda_cost: float = 0.1,
        max_steps: int = 32,
        target_label: Optional[int] = 0,
        poisoned_eval_mask: Optional[torch.Tensor] = None,
        asr_eval_frequency: int = 1,
        min_clean_acc: float = 0.0,
        clean_acc_penalty: float = 0.0,
        anomaly_weight: float = 0.7,
        jaccard_threshold: float = 0.05,
        guard_beta: float = 3.0,
        max_local_prune_edges: int = 2,
    ):
        self.original_poisoned_data = poisoned_data.to(device)
        self.model = model.to(device)
        self.device = device

        self.topk = topk
        self.lambda_asr = lambda_asr
        self.lambda_acc = lambda_acc
        self.lambda_cost = lambda_cost
        self.max_steps = max_steps
        self.target_label = target_label
        self.asr_eval_frequency = max(1, asr_eval_frequency)
        self.min_clean_acc = min_clean_acc
        self.clean_acc_penalty = clean_acc_penalty
        self.anomaly_weight = anomaly_weight
        self.jaccard_threshold = jaccard_threshold
        self.guard_beta = guard_beta
        self.max_local_prune_edges = max_local_prune_edges

        self.action_types = [
            "skip",
            "del_edge",
            "weaken_edge",
            "isolate",
            "feature_gate",
            "jaccard_prune",
            "guard_reweight",
            "consistency_prune",
            "anomaly_clip",
        ]

        if poisoned_eval_mask is None:
            if hasattr(self.original_poisoned_data, "attack_mask"):
                poisoned_eval_mask = self.original_poisoned_data.attack_mask
            elif hasattr(self.original_poisoned_data, "poison_mask"):
                poisoned_eval_mask = self.original_poisoned_data.poison_mask
            elif hasattr(self.original_poisoned_data, "test_mask"):
                poisoned_eval_mask = self.original_poisoned_data.test_mask
            else:
                poisoned_eval_mask = torch.ones(
                    self.original_poisoned_data.num_nodes,
                    dtype=torch.bool,
                    device=device,
                )
        self.poisoned_eval_mask = poisoned_eval_mask.to(device).bool()

        self.n_action_types = len(self.action_types)
        self.action_space_size = self.n_action_types * self.topk
        self.state_space_size = self.topk * 3 + 4

        self.current_data: Optional[Data] = None
        self.edge_weights: Optional[torch.Tensor] = None
        self.candidates: Optional[torch.Tensor] = None
        self.candidate_scores: Optional[torch.Tensor] = None
        self.candidate_aux: Optional[Dict[str, Any]] = None

        self.step_count = 0
        self.total_cost = 0.0
        self.budget = 0.0
        self.previous_metrics = {"clean_acc": 0.0, "asr": 0.0}

        self.asr_cache = 0.0
        self.asr_cache_step = -1

    def get_action_space_size(self) -> int:
        return self.action_space_size

    def get_state_space_size(self) -> int:
        return self.state_space_size

    def reset(self, budget_ratio: float = 0.005, external_candidates: Optional[torch.Tensor] = None):
        self.current_data = copy.deepcopy(self.original_poisoned_data).to(self.device)
        self.edge_weights = build_soft_edge_weight(self.current_data.edge_index, device=self.device)

        self.step_count = 0
        self.total_cost = 0.0
        self.budget = float(max(1, int(self.current_data.edge_index.size(1) * budget_ratio)))

        if external_candidates is not None:
            self.candidates = external_candidates.to(self.device)
            self._compute_candidate_scores()
        else:
            self._update_candidates()

        self.previous_metrics = self._estimate_metrics()
        self.asr_cache = self.previous_metrics["asr"]
        self.asr_cache_step = 0
        return self._get_state()

    def _update_candidates(self):
        mask = torch.ones(self.current_data.num_nodes, dtype=torch.bool, device=self.device)
        # Avoid editing training nodes to better preserve clean-task utility.
        if hasattr(self.current_data, "train_mask"):
            mask = mask & (~self.current_data.train_mask.bool())
        indices, scores, aux_info = select_topk_candidates(
            self.model,
            self.current_data,
            topk=self.topk,
            edge_weight=self.edge_weights,
            mask=mask,
        )
        anomaly_scores = self._compute_node_anomaly_scores()
        mixed_scores = scores + self.anomaly_weight * anomaly_scores[indices]

        # Re-sort by mixed score to prioritize suspicious-yet-uncertain nodes.
        order = torch.argsort(mixed_scores, descending=True)
        self.candidates = indices[order]
        self.candidate_scores = mixed_scores[order]
        self.candidate_aux = aux_info
        self.candidate_aux["anomaly_scores"] = anomaly_scores

    def _compute_candidate_scores(self):
        al_scores, aux_info = compute_al_scores(
            self.model,
            self.current_data,
            edge_weight=self.edge_weights,
            mc_samples=8,
        )
        if self.candidates.device != al_scores.device:
            self.candidates = self.candidates.to(al_scores.device)
        self.candidate_scores = al_scores[self.candidates]
        self.candidate_aux = aux_info

    def _estimate_metrics(self) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.current_data.x, self.current_data.edge_index, self.edge_weights)
            pred = logits.argmax(dim=-1)

            if hasattr(self.current_data, "test_mask"):
                clean_mask = self.current_data.test_mask
            else:
                clean_mask = torch.ones(self.current_data.num_nodes, dtype=torch.bool, device=self.device)

            clean_acc = (pred[clean_mask] == self.current_data.y[clean_mask]).float().mean().item()
            asr = self._get_cached_asr(pred)
            return {"clean_acc": clean_acc, "asr": asr}

    def _get_cached_asr(self, pred: Optional[torch.Tensor] = None) -> float:
        if (self.step_count - self.asr_cache_step) >= self.asr_eval_frequency or self.asr_cache_step < 0:
            self.asr_cache = self._estimate_asr(pred)
            self.asr_cache_step = self.step_count
        return self.asr_cache

    def _estimate_asr(self, pred: Optional[torch.Tensor] = None) -> float:
        if pred is None:
            with torch.no_grad():
                logits = self.model(self.current_data.x, self.current_data.edge_index, self.edge_weights)
                pred = logits.argmax(dim=-1)

        eval_mask = self.poisoned_eval_mask
        if eval_mask.sum().item() == 0:
            return 0.0

        if self.target_label is not None:
            target = torch.full_like(pred[eval_mask], fill_value=int(self.target_label))
            return (pred[eval_mask] == target).float().mean().item()

        # Fallback: untargeted error rate on poisoned evaluation nodes.
        return (pred[eval_mask] != self.current_data.y[eval_mask]).float().mean().item()

    def _get_state(self) -> torch.Tensor:
        padded_scores = torch.zeros(self.topk, dtype=torch.float32, device=self.device)
        padded_entropy = torch.zeros(self.topk, dtype=torch.float32, device=self.device)
        padded_bald = torch.zeros(self.topk, dtype=torch.float32, device=self.device)

        if self.candidates is not None and len(self.candidates) > 0:
            actual_len = min(len(self.candidates), self.topk)
            padded_scores[:actual_len] = self.candidate_scores[:actual_len]

            unc_aux = self.candidate_aux.get("uncertainty_aux", {}) if self.candidate_aux is not None else {}
            entropy = unc_aux.get("entropy", None)
            bald = unc_aux.get("bald", None)
            if entropy is not None and bald is not None:
                ent_vals = entropy[self.candidates[:actual_len]].to(self.device)
                bald_vals = bald[self.candidates[:actual_len]].to(self.device)
                padded_entropy[:actual_len] = ent_vals
                padded_bald[:actual_len] = bald_vals

        candidate_features = torch.stack([padded_scores, padded_entropy, padded_bald], dim=-1).flatten()

        budget_remaining = max(0.0, self.budget - self.total_cost)
        global_features = torch.tensor(
            [
                budget_remaining / max(self.budget, 1.0),
                self.step_count / max(1, self.max_steps),
                float(self.previous_metrics["clean_acc"]),
                float(self.previous_metrics["asr"]),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return torch.cat([candidate_features, global_features], dim=0)

    def _decode_action(self, action: int) -> Tuple[int, int]:
        action = int(action)
        action_type_id = max(0, min(action // self.topk, self.n_action_types - 1))
        candidate_id = max(0, min(action % self.topk, max(0, len(self.candidates) - 1)))
        return action_type_id, candidate_id

    def _execute_action(self, action_type: str, target_node: int) -> float:
        cost = 0.0

        if action_type == "skip":
            return 0.0

        incident_edges = node_incident_edges(self.current_data.edge_index, target_node)

        if action_type == "del_edge":
            if len(incident_edges) > 0:
                edge_id = self._pick_suspicious_incident_edge(incident_edges)
                self.edge_weights = zero_edges(self.edge_weights, edge_id.view(-1))
                cost = 1.0
        elif action_type == "weaken_edge":
            if len(incident_edges) > 0:
                edge_id = self._pick_suspicious_incident_edge(incident_edges)
                self.edge_weights = weaken_edges(self.edge_weights, edge_id.view(-1), factor=0.5)
                cost = 0.5
        elif action_type == "isolate":
            self.edge_weights = apply_isolate_node(self.current_data.edge_index, self.edge_weights, target_node)
            cost = float(len(incident_edges))
        elif action_type == "feature_gate":
            feat_dim = self.current_data.x.size(1)
            k = max(1, int(feat_dim * 0.2))
            dims = torch.randperm(feat_dim, device=self.device)[:k]
            self.current_data.x[target_node, dims] = 0.0
            cost = float(k) * 0.1
        elif action_type == "jaccard_prune":
            removed = self._prune_by_similarity(target_node, mode="jaccard")
            cost = float(removed)
        elif action_type == "guard_reweight":
            touched = self._reweight_by_similarity(target_node)
            cost = 0.25 * float(touched)
        elif action_type == "consistency_prune":
            removed = self._prune_by_prediction_consistency(target_node)
            cost = float(removed)
        elif action_type == "anomaly_clip":
            clipped = self._clip_anomalous_features(target_node)
            cost = 0.05 * float(clipped)

        return cost

    def _pick_suspicious_incident_edge(self, incident_edges: torch.Tensor) -> torch.Tensor:
        """
        Pick one edge that is more likely to be anomalous:
        lowest endpoint feature cosine similarity among incident edges.
        """
        if incident_edges.numel() <= 1:
            return incident_edges[:1]

        eids = incident_edges.long()
        src = self.current_data.edge_index[0, eids]
        dst = self.current_data.edge_index[1, eids]
        x = self.current_data.x

        src_x = x[src]
        dst_x = x[dst]
        src_n = src_x / (src_x.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        dst_n = dst_x / (dst_x.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        sim = (src_n * dst_n).sum(dim=-1)
        # Smaller similarity -> likely abnormal edge.
        min_idx = torch.argmin(sim)
        return eids[min_idx : min_idx + 1]

    def _compute_node_anomaly_scores(self) -> torch.Tensor:
        """
        Multidimensional anomaly score inspired by MAD-style signals:
        feature outlier + prediction uncertainty + neighborhood class inconsistency.
        """
        n = self.current_data.num_nodes
        x = self.current_data.x
        feat_center = x.mean(dim=0, keepdim=True)
        feat_scale = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        feat_z = ((x - feat_center) / feat_scale).abs().mean(dim=-1)

        with torch.no_grad():
            logits = self.model(self.current_data.x, self.current_data.edge_index, self.edge_weights)
            probs = logits.softmax(dim=-1)
            max_probs, pred = probs.max(dim=-1)
            low_conf = 1.0 - max_probs

        edge_index = self.current_data.edge_index
        src = edge_index[0]
        dst = edge_index[1]
        mismatch = (pred[src] != pred[dst]).float()
        mismatch_acc = torch.zeros(n, dtype=torch.float32, device=self.device)
        deg_acc = torch.zeros(n, dtype=torch.float32, device=self.device)
        mismatch_acc.scatter_add_(0, src, mismatch)
        mismatch_acc.scatter_add_(0, dst, mismatch)
        one = torch.ones_like(mismatch)
        deg_acc.scatter_add_(0, src, one)
        deg_acc.scatter_add_(0, dst, one)
        inconsistency = mismatch_acc / deg_acc.clamp_min(1.0)

        # Robustly normalize components.
        def _norm(v: torch.Tensor) -> torch.Tensor:
            return (v - v.mean()) / (v.std().clamp_min(1e-6))

        score = 0.4 * _norm(feat_z) + 0.35 * _norm(low_conf) + 0.25 * _norm(inconsistency)
        score = (score - score.min()) / (score.max() - score.min() + 1e-6)
        return score

    def _incident_cosine_similarity(self, incident_edges: torch.Tensor) -> torch.Tensor:
        if incident_edges.numel() == 0:
            return torch.empty(0, device=self.device)
        eids = incident_edges.long()
        src = self.current_data.edge_index[0, eids]
        dst = self.current_data.edge_index[1, eids]
        x = self.current_data.x
        src_x = x[src]
        dst_x = x[dst]
        src_x = src_x / src_x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        dst_x = dst_x / dst_x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return (src_x * dst_x).sum(dim=-1)

    def _incident_jaccard_similarity(self, incident_edges: torch.Tensor) -> torch.Tensor:
        if incident_edges.numel() == 0:
            return torch.empty(0, device=self.device)
        eids = incident_edges.long()
        src = self.current_data.edge_index[0, eids]
        dst = self.current_data.edge_index[1, eids]
        x_bin = (self.current_data.x > 0).float()
        src_x = x_bin[src]
        dst_x = x_bin[dst]
        inter = (src_x * dst_x).sum(dim=-1)
        union = ((src_x + dst_x) > 0).float().sum(dim=-1).clamp_min(1.0)
        return inter / union

    def _prune_by_similarity(self, target_node: int, mode: str = "jaccard") -> int:
        incident_edges = node_incident_edges(self.current_data.edge_index, target_node)
        if incident_edges.numel() == 0:
            return 0

        if mode == "jaccard":
            sim = self._incident_jaccard_similarity(incident_edges)
            bad_mask = sim < self.jaccard_threshold
        else:
            sim = self._incident_cosine_similarity(incident_edges)
            bad_mask = sim < 0.0

        candidate_eids = incident_edges[bad_mask]
        if candidate_eids.numel() == 0:
            # Fallback: remove the least similar edge.
            candidate_eids = incident_edges[torch.argmin(sim).view(-1)]

        k = min(self.max_local_prune_edges, candidate_eids.numel())
        to_remove = candidate_eids[:k]
        self.edge_weights = zero_edges(self.edge_weights, to_remove)
        return int(k)

    def _reweight_by_similarity(self, target_node: int) -> int:
        incident_edges = node_incident_edges(self.current_data.edge_index, target_node)
        if incident_edges.numel() == 0:
            return 0
        sim = self._incident_cosine_similarity(incident_edges)
        # Map cosine similarity to (0,1), similar to guard-style edge reweighting.
        gate = torch.sigmoid(self.guard_beta * sim)
        self.edge_weights[incident_edges] = self.edge_weights[incident_edges] * gate
        return int(incident_edges.numel())

    def _prune_by_prediction_consistency(self, target_node: int) -> int:
        incident_edges = node_incident_edges(self.current_data.edge_index, target_node)
        if incident_edges.numel() == 0:
            return 0

        with torch.no_grad():
            logits = self.model(self.current_data.x, self.current_data.edge_index, self.edge_weights)
            probs = logits.softmax(dim=-1)
            pred = probs.argmax(dim=-1)
            conf = probs.max(dim=-1).values

        node_class = pred[target_node]
        eids = incident_edges.long()
        src = self.current_data.edge_index[0, eids]
        dst = self.current_data.edge_index[1, eids]
        nbr = torch.where(src == target_node, dst, src)

        disagree = pred[nbr] != node_class
        high_conf = conf[nbr] > 0.6
        bad = disagree & high_conf
        bad_eids = eids[bad]
        if bad_eids.numel() == 0:
            return 0

        k = min(self.max_local_prune_edges, bad_eids.numel())
        self.edge_weights = zero_edges(self.edge_weights, bad_eids[:k])
        return int(k)

    def _clip_anomalous_features(self, target_node: int) -> int:
        x = self.current_data.x
        feat_mean = x.mean(dim=0)
        feat_std = x.std(dim=0).clamp_min(1e-6)
        row = x[target_node]
        z = (row - feat_mean) / feat_std
        idx = torch.nonzero(z.abs() > 2.5, as_tuple=False).flatten()
        if idx.numel() == 0:
            return 0
        clipped = torch.clamp(z[idx], min=-2.5, max=2.5) * feat_std[idx] + feat_mean[idx]
        self.current_data.x[target_node, idx] = clipped
        return int(idx.numel())

    def _compute_reward(self, before: Dict[str, float], after: Dict[str, float], cost: float) -> float:
        delta_asr = after["asr"] - before["asr"]
        delta_acc = after["clean_acc"] - before["clean_acc"]
        reward = -self.lambda_asr * delta_asr + self.lambda_acc * delta_acc - self.lambda_cost * cost
        if after["clean_acc"] < self.min_clean_acc:
            reward -= self.clean_acc_penalty * (self.min_clean_acc - after["clean_acc"])
        return float(reward)

    def step(self, action: int):
        if isinstance(action, torch.Tensor):
            action = action.item()

        action_type_id, candidate_id = self._decode_action(action)
        if self.candidates is None or candidate_id >= len(self.candidates):
            return self._get_state(), -0.1, True, {"error": "invalid_action"}

        action_type = self.action_types[action_type_id]
        target_node = int(self.candidates[candidate_id].item())

        before_metrics = dict(self.previous_metrics)
        cost = self._execute_action(action_type, target_node)

        self.total_cost += cost
        self.step_count += 1

        after_metrics = self._estimate_metrics()
        self.previous_metrics = after_metrics

        reward = self._compute_reward(before_metrics, after_metrics, cost)
        done = (self.total_cost >= self.budget) or (self.step_count >= self.max_steps)

        if not done:
            self._update_candidates()

        info = {
            "action_type": action_type,
            "target_node": target_node,
            "cost": cost,
            "total_cost": self.total_cost,
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "budget_remaining": max(0.0, self.budget - self.total_cost),
        }
        return self._get_state(), reward, done, info
