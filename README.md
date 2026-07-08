# Briscola-PML

Bayesian opponent modelling for two-player Briscola.

The project studies a focused probabilistic ML problem: infer an interpretable
latent style vector `theta` for an opponent from observed Briscola moves, then
use the posterior over `theta` to predict future opponent actions.

The goal is not to build a complete Briscola-playing agent. The goal is to
isolate and validate the inference problem: hidden opponent style, hidden
opponent hand, partial observations, synthetic-data recovery, and predictive
performance.

## Current Status

Implemented:

- Two-player Briscola simulator.
- Card, suit, rank, points, trick strength, scoring, and draw order.
- Public game state and player views.
- Synthetic softmax opponent models parameterized by `theta`.
- Interpretable move features `phi(card, hand, state)`.
- Synthetic episode collection from simulated games.
- Marginal likelihood over hidden hands, with conditional and absolute modes.
- Sequential hand-filter likelihood over opponent hands.
- Mean-field Gaussian variational inference for `theta` with the sequential
  likelihood as the only training objective.
- Synthetic validation workflow with theta recovery, held-out prediction,
  and baseline comparison.
- Multi-run comparison script for feature sets, profiles, and seeds.
- Unit tests for simulator, opponents, inference, VI, validation, and scripts.

Not implemented yet:

- Exact sequential filtering over the full residual-deck state space.
- Hierarchical priors across opponents.
- Amortized or online inference.
- Decision-value evaluation with a best-response player.

## Model

At each observed opponent move, the opponent has a private hand `H_t`, the
public information is `I_t`, and the observed card is `c_t`.

The action model is a softmax:

```text
p(c_t = x | H_t, I_t, theta)
    = exp(theta^T phi(x, H_t, I_t))
      / sum_{x' in H_t} exp(theta^T phi(x', H_t, I_t))
```

where:

- `theta` is the latent opponent-style vector.
- `phi` is an interpretable feature vector for a candidate card.
- `H_t` is hidden during inference and observed only in synthetic generation.

The inference likelihood marginalizes the hidden hand:

```text
p(c_t | I_t, theta) = sum_H b_t(H) p(c_t | H, I_t, theta)
```

The sequential model keeps a belief over hidden opponent hands across the moves
of the same game. This makes the likelihood depend on previous observed actions
instead of treating each move as independent. The current implementation filters
over hands, not over the full residual stock order.

## Repository Layout

```text
game/
  cards.py              Card, Suit, Rank, points, strength, full deck.
  simulator.py          Two-player Briscola engine and public state.

opponents/
  features.py           Interpretable feature extraction for moves.
  models.py             RandomOpponent and ThetaSoftmaxOpponent.

experiments/
  episode_collection.py Synthetic game loop and observation collection.
  validation.py         Recovery, prediction, and validation helpers.

inference/
  beliefs.py            Compatible hidden-hand enumeration.
  likelihood.py         Marginal card probabilities and log likelihood.
  vi.py                 Sequential likelihood and variational inference.

scripts/
  run_validation.py     Single synthetic validation run.
  run_comparison.py     Grid comparison over features, profiles, and seeds.

tests/
  test_*.py             Unit tests for the project modules.
```

## Feature Sets

The default validation feature set is `style`:

```text
is_carico
is_low_trump
is_high_trump
is_low_points
wins_current_trick
```

This set replaces one broad point-value feature with five small tactical
signals: high-value cards, low trumps, high trumps, zero-point cards, and
whether the move wins the current trick. It is meant to make theta recovery more
interpretable and less dominated by a single large coefficient.

The original `core` feature set is still available:

```text
is_trump
points_normalized
wins_current_trick
lowest_card_in_suit
```

Additional feature sets are available for comparison:

- `compact`
- `trump_count`
- `extended`

Synthetic theta profiles include:

- `aggressive`
- `conservative`
- `greedy_points`

These profiles are hand-chosen data-generation settings, not learned
parameters.

## Run Tests

```bash
python3 -m unittest
```

## Run A Validation Experiment

```bash
python3 scripts/run_validation.py \
  --num-games 100 \
  --feature-set style \
  --theta-scale 1.0 \
  --split-unit game \
  --vi-steps 300 \
  --elbo-samples 2 \
  --progress-interval 50 \
  --output artifacts/validation_report.json
```

The report includes:

- posterior mean and standard deviation for each theta component;
- theta recovery error;
- ELBO history and best ELBO step;
- sequential held-out predictive log likelihood versus baseline;

Useful validation flags:

- `--theta-scale`: multiplies the synthetic theta before generating games.
  Values greater than `1.0` make the synthetic opponent style sharper; values
  below `1.0` make it noisier.
- `--split-unit game|observation`: controls train/test splitting. The default
  `game` keeps all moves from the same game in the same partition, which avoids
  leakage between train and held-out data. `observation` splits individual
  moves and is useful for small smoke tests.
- `--prior-std`: controls the width of the zero-mean Gaussian prior over theta.
  Larger values shrink fitted theta less aggressively toward zero.

## Run A Comparison Grid

```bash
python3 scripts/run_comparison.py \
  --feature-sets style core compact trump_count extended \
  --profiles aggressive conservative greedy_points \
  --seeds 0 1 2 \
  --num-games 20 \
  --theta-scale 1.0 \
  --split-unit game \
  --vi-steps 300 \
  --jobs 4
```

By default, outputs are written under `artifacts/comparison/`. Comparison runs
can execute independent configurations in parallel with `--jobs`. Each fitted
posterior is evaluated on held-out games with the same sequential likelihood
used during training.

For faster exploratory comparisons, use `--max-train-observations`,
`--max-test-observations`, and fewer seeds. The run CSV is written
incrementally as each configuration finishes, so progress is visible before the
full grid completes.

`run_validation.py` executes one complete experiment and prints a detailed
report for that run. `run_comparison.py` repeats the same validation pipeline
over several feature sets, profiles, and seeds, then writes aggregate CSV/JSON
tables so different modelling choices can be compared.

## Development Notes

Install dependencies from:

```bash
pip install -r requirements.txt
```
