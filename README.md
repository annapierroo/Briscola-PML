# Briscola-PML

We study Bayesian opponent modelling in two-player Briscola.

The idea is simple: we observe how an opponent plays, we assume that their
style can be represented by a latent parameter vector `theta`, and we try to
infer a posterior distribution over that vector. Once we have this posterior,
we can use it to predict future opponent moves.

We are not trying to build a complete Briscola-playing agent. Our focus is the
inference problem itself: the opponent has a hidden hand, we only see public
information and played cards, and we want to understand whether we can recover
an interpretable playing style from partial observations.

## What We Have Built

At the moment, the project includes:

- a two-player Briscola simulator with scoring, trick resolution, and draw
  order;
- a public game state and player-specific views;
- synthetic opponents that choose cards with a softmax model based on `theta`;
- interpretable card features, such as whether a card is a trump, a carico, or
  wins the current trick;
- synthetic data collection from simulated games;
- an absolute marginal likelihood that reasons over the opponent's hidden hand;
- a sequential hand belief that is updated across the moves of the same game;
- mean-field Gaussian variational inference for `theta`;
- validation scripts for theta recovery and held-out posterior prediction;
- comparison scripts to test different feature sets, opponent profiles, and
  random seeds;
- unit tests for the simulator, opponents, inference code, validation code, and
  scripts.


## Model

For each observed opponent move, we write:

- `H_t` for the opponent's hidden hand;
- `I_t` for the public information available at that point;
- `c_t` for the card the opponent actually plays.

Given a candidate hand and the public state, the opponent chooses a card with a
softmax policy:

```text
p(c_t = x | H_t, I_t, theta)
    = exp(theta^T phi(x, H_t, I_t))
      / sum_{x' in H_t} exp(theta^T phi(x', H_t, I_t))
```

Here `phi` is the feature vector of a candidate card, and `theta` tells us how
strongly the opponent cares about each feature. For example, a high positive
weight on `is_carico` means that the opponent tends to play high-value cards.

During inference we do not know the opponent's hand, so we marginalize it:

```text
p(c_t | I_t, theta) = sum_H b_t(H) p(c_t | H, I_t, theta)
```

The important part is `b_t(H)`: our belief over which hands the opponent could
have. We currently use a sequential hand filter. This means that, within the
same game, we do not treat every move as independent. After each observed move,
we update the belief over possible opponent hands and use that updated belief
for the next move.


## Variational Inference

We approximate the posterior over `theta` with a diagonal Gaussian:

```text
q(theta) = N(mu, diag(sigma^2))
```

The code optimizes an ELBO with reparameterization gradients. In practice, this
means that we learn:

- a posterior mean for every feature weight;
- a posterior standard deviation for every feature weight;
- the final posterior after the VI optimization.

The prior over `theta` is a zero-mean Gaussian.

For held-out evaluation, we use the whole variational posterior rather than
only its mean. We sample several theta vectors from `q(theta)`, evaluate the
sequential likelihood for each one, and average those likelihoods in log space.
This gives us a posterior predictive score for the test games.

## Feature Sets

The default feature contains:

```text
is_carico
is_low_trump
is_high_trump
is_low_points
wins_current_trick
```

We use this set because it is easier to interpret than one broad point-value
feature. It separates several Briscola behaviours:

- playing high-value cards;
- using low trumps;
- using high trumps;
- discarding zero-point cards;
- trying to win the current trick.

The synthetic opponent profiles are:

- `aggressive`
- `conservative`
- `greedy_points`

These profiles are hand-written theta vectors. We use them to generate data
where the true parameters are known, so we can check whether inference recovers
the intended style.

## Repository Layout

```text
game/
  cards.py              Cards, suits, ranks, points, and Briscola strength.
  simulator.py          Two-player Briscola engine and public state.

opponents/
  features.py           Feature extraction for candidate moves.
  models.py             Random and theta-softmax synthetic opponents.

experiments/
  episode_collection.py Synthetic game collection.
  validation.py         Recovery, prediction, and validation helpers.

inference/
  beliefs.py            Compatible hidden-hand enumeration.
  likelihood.py         Local and marginal card probabilities.
  vi.py                 Sequential likelihood and variational inference.

scripts/
  run_experiment.py     Single validation runs and comparison grids.

tests/
  test_*.py             Unit tests.
```

## Run Tests

```bash
python3 -m unittest
```

## Run One Validation

This command runs one synthetic experiment with the default feature set:

```bash
python3 scripts/run_experiment.py single \
  --num-games 100 \
  --feature-set style \
  --profile greedy_points \
  --opponent-temperature 1.0 \
  --vi-steps 300 \
  --elbo-samples 2 \
  --posterior-samples 50 \
  --output artifacts/validation_report.json
```

The report tells us:

- the true synthetic `theta`;
- the posterior mean and standard deviation learned by VI;
- the per-feature error and L2 recovery error;
- the ELBO trajectory;
- held-out posterior predictive log-likelihood against a zero-theta baseline.

Useful flags:

- `--opponent-temperature` controls how deterministic the synthetic opponent's
  softmax policy is. Lower values make the opponent follow its preferences more
  sharply.
- `--prior-std` changes the width of the Gaussian prior over `theta`.
- `--vi-steps` controls how long we optimize the variational posterior.
- `--posterior-samples` controls how many theta samples from `q(theta)` we use
  for held-out posterior prediction.

Validation uses a fixed game-level 75/25 train/test split, which keeps all
moves from one game in the same partition and avoids leakage between train and
held-out data. The Adam learning rate is fixed at `0.03`.

## Run A Comparison

This repeats the same validation pipeline across feature sets, profiles, and
seeds:

```bash
python3 scripts/run_experiment.py compare \
  --feature-sets style core compact interactive trump_count extended \
  --profiles aggressive conservative greedy_points \
  --seeds 0 1 2 \
  --num-games 20 \
  --opponent-temperature 1.0 \
  --vi-steps 300 \
  --posterior-samples 20 \
  --jobs 4
```

The `compare` command does not use a different inference method. Each run
follows the same sequential VI pipeline as `single`; the command only repeats
that pipeline over several configurations and summarizes the results.

Outputs are written under `artifacts/comparison/`. The run CSV is written
incrementally, so we can inspect partial results while a larger comparison is
still running.

## Dependencies

Install the project dependencies with:

```bash
pip install -r requirements.txt
```
