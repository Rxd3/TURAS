"""Heterogeneous mean message passing with learned typed bilinear edge scoring."""
from __future__ import annotations

import math
import random

import torch
from torch import nn


class HeterogeneousRelationModel(nn.Module):
    def __init__(self, snapshot, hidden_dim: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if hidden_dim < 1 or layers < 1 or not 0 <= dropout < 1:
            raise ValueError("invalid graph model shape")
        self.snapshot = snapshot
        self.dim = hidden_dim
        self.relation_ids = sorted(snapshot.relations)
        self.relation_index = {r: i for i, r in enumerate(self.relation_ids)}
        self.embeddings = nn.ModuleDict({t: nn.Embedding(len(ids), hidden_dim) for t, ids in snapshot.nodes.items()})
        self.projections = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in snapshot.nodes})
        self.self_layers = nn.ModuleList([
            nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in snapshot.nodes}) for _ in range(layers)])
        # Every direction has separate learned message parameters.
        self.messages = nn.ModuleList([nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(2 * len(self.relation_ids))]) for _ in range(layers)])
        self.decoder = nn.Parameter(torch.empty(len(self.relation_ids), hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.decoder)
        self.dropout = nn.Dropout(dropout)

    def encode(self):
        device = self.decoder.device
        states = {t: self.projections[t](embedding.weight) for t, embedding in self.embeddings.items()}
        for own, messages in zip(self.self_layers, self.messages):
            sums = {t: torch.zeros_like(h) for t, h in states.items()}
            counts = {t: h.new_zeros((len(h), 1)) for t, h in states.items()}
            for rid, relation in enumerate(self.relation_ids):
                edges = [e for e in self.snapshot.edges if e[2] == relation]
                if not edges:
                    continue
                head_type, tail_type, _ = self.snapshot.relations[relation]
                heads = torch.tensor([e[1] for e in edges], device=device)
                tails = torch.tensor([e[4] for e in edges], device=device)
                sums[tail_type].index_add_(0, tails, messages[2 * rid](states[head_type][heads]))
                counts[tail_type].index_add_(0, tails, states[tail_type].new_ones((len(edges), 1)))
                sums[head_type].index_add_(0, heads, messages[2 * rid + 1](states[tail_type][tails]))
                counts[head_type].index_add_(0, heads, states[head_type].new_ones((len(edges), 1)))
            states = {t: self.dropout(torch.relu(own[t](h) + sums[t] / counts[t].clamp_min(1)))
                      for t, h in states.items()}
        return states

    def score_vectors(self, head, tail, relation: str):
        matrix = self.decoder[self.relation_index[relation]]
        score = ((head @ matrix) * tail).sum(-1)
        if not self.snapshot.relations[relation][2]:
            score = (score + ((tail @ matrix) * head).sum(-1)) / 2
        return score / math.sqrt(self.dim)

    def score_edges(self, states, edges):
        output = self.decoder.new_zeros((len(edges),))
        for relation in self.relation_ids:
            selected = [(i, e) for i, e in enumerate(edges) if e[2] == relation]
            if not selected:
                continue
            ht, tt, _ = self.snapshot.relations[relation]
            positions = torch.tensor([i for i, _ in selected], device=self.decoder.device)
            heads = torch.tensor([e[1] for _, e in selected], device=self.decoder.device)
            tails = torch.tensor([e[4] for _, e in selected], device=self.decoder.device)
            values = self.score_vectors(states[ht][heads], states[tt][tails], relation)
            output = output.index_copy(0, positions, values)
        return output


def canonical(edge, snapshot):
    htype, head, relation, ttype, tail = edge
    if not snapshot.relations[relation][2] and htype == ttype and head > tail:
        return ttype, tail, relation, htype, head
    return tuple(edge)


def sample_negatives(edges, snapshot, *, seed: int):
    """Closed-world typed corruption, excluding every known train/internal-dev positive."""
    positive = {canonical(e, snapshot) for e in snapshot.edges + snapshot.validation_edges}
    rng = random.Random(seed)
    negative, saturated = [], 0
    for edge in edges:
        ht, h, r, tt, t = edge
        options = [(ht, h, r, tt, j) for j in range(len(snapshot.nodes[tt]))]
        options = [e for e in options if canonical(e, snapshot) not in positive
                   and not (ht == tt and e[1] == e[4])]
        if not options:
            options = [(ht, j, r, tt, t) for j in range(len(snapshot.nodes[ht]))]
            options = [e for e in options if canonical(e, snapshot) not in positive
                       and not (ht == tt and e[1] == e[4])]
        if options:
            negative.append(rng.choice(options))
        else:
            saturated += 1
    return negative, saturated


def train_relation_model(model, *, epochs: int, learning_rate: float, seed: int = 17):
    """Fit a learned edge decoder; never add held-out edges to encode() adjacency."""
    if epochs < 1 or learning_rate <= 0 or not model.snapshot.edges:
        raise ValueError("invalid graph training budget or empty training graph")
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        model.train()
        negative, saturated = sample_negatives(model.snapshot.edges, model.snapshot, seed=seed + epoch)
        if not negative:
            raise ValueError("no typed negative edges available; graph is saturated")
        optimizer.zero_grad(set_to_none=True)
        states = model.encode()
        positive = model.score_edges(states, model.snapshot.edges)
        negative_scores = model.score_edges(states, negative)
        logits = torch.cat((positive, negative_scores))
        labels = torch.cat((torch.ones_like(positive), torch.zeros_like(negative_scores)))
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("non-finite graph loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append({"epoch": epoch + 1, "loss": float(loss.detach()),
                        "negatives": len(negative), "saturated_positive_edges": saturated})
    return history


def evaluate_relations(model, *, seed: int = 2026):
    """Internal train-group holdout AP and filtered tail-ranking MRR/Hits@10."""
    snapshot = model.snapshot
    positives = snapshot.validation_edges
    if not positives:
        return {"status": "unavailable", "reason": "no internal held-out co-mention relations"}
    negatives, saturated = sample_negatives(positives, snapshot, seed=seed)
    if not negatives:
        return {"status": "unavailable", "reason": "no typed evaluation negatives"}
    model.eval()
    with torch.inference_mode():
        states = model.encode()
        scores = model.score_edges(states, positives + negatives).tolist()
        labels = [1] * len(positives) + [0] * len(negatives)
        ranked = sorted(zip(scores, labels), key=lambda pair: -pair[0])
        # Group equal scores so all-tied predictions return prevalence, not an
        # optimistic result dependent on positive-first input ordering.
        tp = fp = 0
        ap = 0.0
        for value in sorted(set(scores), reverse=True):
            group = [label for score, label in ranked if score == value]
            added = sum(group)
            tp += added
            fp += len(group) - added
            ap += added / len(positives) * tp / (tp + fp)
        known = {canonical(e, snapshot) for e in snapshot.edges + positives}
        ranks = []
        for edge in positives:
            ht, h, relation, tt, tail = edge
            alternatives = [(ht, h, relation, tt, j) for j in range(len(snapshot.nodes[tt]))
                            if (ht != tt or h != j) and ((ht, h, relation, tt, j) == edge
                            or canonical((ht, h, relation, tt, j), snapshot) not in known)]
            values = model.score_edges(states, alternatives).tolist()
            target_score = values[alternatives.index(edge)]
            # Pessimistic ties; never reward an arbitrary candidate order.
            ranks.append(sum(v >= target_score for v in values))
    return {"status": "measured_internal_training_group_holdout", "average_precision": ap,
            "mrr": sum(1 / r for r in ranks) / len(ranks),
            "hits_at_10": sum(r <= 10 for r in ranks) / len(ranks),
            "positive_count": len(positives), "negative_count": len(negatives),
            "saturated_positive_edges": saturated,
            "scope": "internal split of training annotations only; not corpus validation/test"}
