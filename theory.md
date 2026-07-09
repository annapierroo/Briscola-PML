# Theory Notes: Bayesian Opponent Modelling in Two-Player Briscola

This document describes the probabilistic model implemented in the
project.

The project is inspired by Southey et al.'s *Bayes' Bluff: Opponent
Modelling in Poker* paper, which studies Bayesian opponent modelling in a card game
with hidden private information. We adopt the same high-level idea of separating
uncertainty about hidden cards from uncertainty about the opponent's playing
style, and of updating beliefs using observed actions. 

Our project differs from that setting because Briscola has different rules, draw dynamics,
and strategic structure from poker. In addition, our goal is narrower: we focus
on recovering and evaluating an interpretable opponent style model, rather than
using the inferred model to compute an exploitative playing policy.

The current implementation uses:

- a feature-based softmax model for synthetic opponent moves;
- a sequential hidden-hand belief within each game;
- a sequential likelihood for training and test evaluation;
- a mean-field Gaussian variational posterior over `theta`;
- posterior predictive test scoring by sampling from the fitted posterior.

---

## 1. Project Scope

The project focuses on Bayesian opponent modelling in two-player Briscola. The
goal is not to learn an optimal Briscola-playing policy, but to infer an
interpretable latent playing style from partially observed play.

We represent the opponent's style with a parameter vector

$$
\theta \in \mathbb{R}^d
$$

Given observations from one or more games, we infer a posterior distribution
over this vector:

$$
p(\theta \mid D)
$$

This posterior represents which playing styles remain plausible after observing
the opponent's actions.

---

## 2. Observations and Latent Variables

For each observed opponent move:

| Symbol | Role | Meaning |
|---|---|---|
| $I_{g,t}$ | observed | observer information before the move: public state and observer hand |
| $c_{g,t}$ | observed | card played by the opponent |
| $H_{g,t}$ | latent local variable | opponent's hidden hand before the move |
| $\theta$ | latent global variable | opponent's playing style |

$I_{g,t}$ corresponds to the pair composed by the public
game state and the observer's private hand.

Here we write the game index $g$ because the hidden-hand belief is reset at game
boundaries. Within the same game, moves are not treated as independent: earlier
observed cards affect the belief over the opponent's possible later hands.

For readability, we often write $I_t$, $c_t$, and $H_t$ when the game index is
clear from context.

---

## 3. Bayesian Formulation

The inferential target is the opponent style parameter $\theta$. Since $\theta$
is not observed directly, we evaluate candidate values through the probability
they assign to the observed move sequence.

For each candidate value of $\theta$, the model defines a likelihood for the
observed opponent actions. Bayes' rule combines this likelihood with a prior:

$$
p(\theta \mid D)
\propto
p(D \mid \theta)p(\theta)
$$


---

## 4. Feature-Based Softmax Policy

For a candidate card $x$, candidate hidden hand $H_t$, and observer information
$I_t$, we compute a feature vector

$$
\phi(x,H_t,I_t) \in \mathbb{R}^d
$$

The card score is linear in the feature vector:

$$
s_\theta(x;H_t,I_t)
=
\theta^\top \phi(x,H_t,I_t)
$$

If the opponent's hand were known, the probability of playing card $x$ would be
a softmax over the legal cards in that hand:

$$
p(c_t = x \mid H_t,I_t,\theta)
=
\frac{
\exp(s_\theta(x;H_t,I_t)/\tau)
}{
\sum_{x' \in H_t}\exp(s_\theta(x';H_t,I_t)/\tau)
}
$$

The temperature $\tau>0$ controls how sharply the synthetic opponent follows
the scores. In the default validation runs, $\tau=1$.

This is a modelling assumption. We are not claiming that real players compute a
linear score. We use this form because it is interpretable, differentiable, and
small enough to fit from limited data.

---

## 5. Current Feature Sets

The project uses two feature sets:

### `core`

```text
is_trump
points_normalized
wins_current_trick
lowest_card_in_suit
```


### `interaction`

```text
trump_progress
points_progress
trump_on_table_points
greedy_take
```

`interaction` keeps only context-dependent terms involving game progress and
the current trick:

- `trump_progress`: trump card played later in the game;
- `points_progress`: point value played later in the game;
- `trump_on_table_points`: trump use when points are already on the table;
- `greedy_take`: winning the current trick when points are on the table.

---

## 6. Synthetic Style Profiles

For validation, we generate synthetic opponents from hand-written
vectors. The profiles we use are:

- `aggressive`
- `conservative`
- `greedy_points`

For `core`, the feature order is:

$$
(\texttt{is\_trump},
\texttt{points\_normalized},
\texttt{wins\_current\_trick},
\texttt{lowest\_card\_in\_suit})
$$

The current `core` vectors are:

| Profile | $\theta_{\mathrm{true}}$ | Intended behaviour |
|---|---|---|
| `aggressive` | $(0.4,\ 1.4,\ 2.2,\ -0.2)$ | values points and winning the current trick |
| `conservative` | $(-1.8,\ -1.1,\ 0.7,\ 1.2)$ | avoids trumps and points, prefers safer low-suit discards |
| `greedy_points` | $(0.1,\ 3.0,\ 0.4,\ -0.2)$ | strongly values high-point cards |

For `interaction`, the feature order is:

$$
(\texttt{trump\_progress},
\texttt{points\_progress},
\texttt{trump\_on\_table\_points},
\texttt{greedy\_take})
$$

The current `interaction` vectors are:

| Profile | $\theta_{\mathrm{true}}$ |
|---|---|
| `aggressive` | $(-1.0,\ -1.5,\ 2.0,\ 3.0)$ |
| `conservative` | $(1.5,\ 1.5,\ -0.5,\ -1.0)$ |
| `greedy_points` | $(-0.1,\ -2.0,\ 0.5,\ 1.0)$ |

These profiles are only used to generate controlled synthetic data. During
inference, the code sees the observations, not the true vector.

---

## 7. Hidden-Hand Marginalization

The observer does not know the opponent's actual hand. For each move, the model
therefore considers all candidate hands compatible with the observer information
used by the belief.

Let $\mathcal{H}_t$ be the set of compatible hands before move $t$. A local
hidden-hand likelihood has the form:

$$
p(c_t \mid I_t,\theta)
=
\sum_{H \in \mathcal{H}_t}
b_t(H)\,
p(c_t \mid H,I_t,\theta)
$$

Here $b_t(H)$ is the belief weight assigned to candidate hand $H$. Candidate
hands are built from cards that could still belong to the opponent. The helper
removes cards that are known not to be in the opponent hand, including the
observer's hand, cards already played, cards currently in the trick, and the
trump card if it is still publicly in the stock.

---

## 8. Sequential Hand Belief

The sequential model carries a belief over possible opponent
hands through the observed moves of the same game.

At the first observed move of a game, the belief is uniform over compatible
hands:

$$
b_1(H)
=
\frac{1}{|\mathcal{H}_1|},
\qquad
H \in \mathcal{H}_1
$$

For a later move, the belief already contains information from previous
observed actions. The likelihood contribution of the observed card is:

$$
\ell_t(\theta)
=
p(c_t \mid I_{\le t},c_{<t},\theta)
=
\sum_{H \in \mathcal{H}_t}
b_t(H;\theta)\,
p(c_t \mid H,I_t,\theta)
$$

The notation $b_t(H;\theta)$ is intentional. After the first move, the belief
depends on $\theta$: a candidate hand receives more posterior mass if it makes
the previously observed actions more likely under that style.

After observing $c_t$, the belief over hands that could have produced the move
is updated by Bayes' rule:

$$
\bar b_t(H;\theta)
=
\frac{
b_t(H;\theta)\,
p(c_t \mid H,I_t,\theta)
}{
\ell_t(\theta)
}
$$

Hands that do not contain the observed card contribute zero probability.

The model then transitions to the next decision point by removing the played
card and filling any missing cards with uniformly weighted compatible
completions:

$$
b_{t+1}(H';\theta)
=
\sum_H
\bar b_t(H;\theta)T_t(H' \mid H)
$$

After an observed move, we remove the played card from each candidate hand.
If the next public hand size implies that the opponent must have drawn new
unknown cards, we enumerate all compatible ways to fill the hand and assign them
equal weight. This way, the filter is "exact" over the candidate hands it represents.

---

## 9. Sequential Likelihood

We group observations by game. Conditional on $\theta$,
different games are treated as independent. Within a game, observations are
connected by the sequential hand belief.

For one game:

$$
p_{\mathrm{seq}}(D_g \mid \theta)
=
\prod_{t=1}^{T_g}
\ell_{g,t}(\theta)
$$

For several games:

$$
p_{\mathrm{seq}}(D \mid \theta)
=
\prod_g
p_{\mathrm{seq}}(D_g \mid \theta)
$$

In log form:

$$
\log p_{\mathrm{seq}}(D \mid \theta)
=
\sum_g
\sum_{t=1}^{T_g}
\log \ell_{g,t}(\theta)
$$

This is the likelihood used for training and for test evaluation.

---

## 10. Prior and Posterior

The prior over playing style is a zero-mean diagonal Gaussian:

$$
\theta \sim \mathcal{N}(0,\sigma_0^2 I)
$$

The default is $\sigma_0=1$, and `run_experiment.py` exposes this as
`--prior-std`.

Bayes' rule gives:

$$
p(\theta \mid D)
=
\frac{
p_{\mathrm{seq}}(D \mid \theta)p(\theta)
}{
p(D)
}
$$

Since the evidence $p(D)$ is not available in closed form, we work with
the unnormalized posterior:

$$
p(\theta \mid D)
\propto
p_{\mathrm{seq}}(D \mid \theta)p(\theta)
$$

---

## 11. Variational Inference

The posterior is approximated with a diagonal Gaussian:

$$
q_\lambda(\theta)
=
\mathcal{N}(\mu,\operatorname{diag}(\sigma^2))
$$

The learned quantities are:

- posterior mean $\mu$;
- posterior standard deviation $\sigma$;
- an ELBO trajectory over optimization steps.

In the implementation, the optimized parameters are the mean vector and the
log-standard-deviation vector:

$$
\lambda = (\mu,\log\sigma)
$$

We optimize $\log\sigma$ rather than $\sigma$ directly so that the recovered
standard deviations are always positive after exponentiation.

The ELBO optimized by the code is:

$$
\mathcal{L}(\lambda)
=
\mathbb{E}_{q_\lambda(\theta)}
\left[
\log p_{\mathrm{seq}}(D_{\mathrm{train}} \mid \theta)
\right]
-
D_{\mathrm{KL}}
\left(
q_\lambda(\theta)
\|p(\theta)
\right)
$$

The expectation is estimated with reparameterized Monte Carlo samples:

$$
\epsilon \sim \mathcal{N}(0,I),
\qquad
\theta = \mu + \sigma\odot\epsilon
$$

For one VI step with $S$ ELBO samples, the
code estimates the expected log-likelihood as:

$$
\frac{1}{S}
\sum_{s=1}^{S}
\log p_{\mathrm{seq}}
\left(
D_{\mathrm{train}} \mid
\mu+\sigma\odot\epsilon^{(s)}
\right),
\qquad
\epsilon^{(s)} \sim \mathcal{N}(0,I)
$$

The KL term is computed analytically for the diagonal Gaussian posterior and
Gaussian prior. The stochastic part of the objective is therefore only the
expected sequential log-likelihood term.

The flag `--elbo-samples` controls how many Monte Carlo samples are used per VI
step to estimate the expected log-likelihood. 

The variational parameters are optimized with Adam. Since the code minimizes
losses, it performs gradient descent on the negative ELBO:

$$
\operatorname{loss}(\lambda)
=
-\mathcal{L}(\lambda)
$$

At each VI step, Adam uses the current stochastic gradient estimate to update
$\mu$ and $\log\sigma$.

---

## 12. Posterior Predictive Test Scoring

After fitting $q(\theta)$ on the training games, the code evaluates prediction
on test games with posterior predictive averaging. It samples:

$$
\theta^{(s)} \sim q_\lambda(\theta),
\qquad
s=1,\dots,S
$$

and estimates:

$$
\log p(D_{\mathrm{test}} \mid D_{\mathrm{train}})
\approx
\log
\left(
\frac{1}{S}
\sum_{s=1}^{S}
p_{\mathrm{seq}}(D_{\mathrm{test}} \mid \theta^{(s)})
\right)
$$

We compute this with a numerically stable log-mean-exp over
the sequential test log-likelihoods.

The flag `--posterior-samples` controls $S$.

The baseline for test comparison is the zero-theta model:

$$
\theta = 0
$$

With all scores equal, this baseline plays uniformly over the candidate hand in
the local softmax. It still uses the same sequential hidden-hand likelihood for
the hand belief.

---

## 13. Train/Test Split

Synthetic observations are split at the game level. Complete games go either
to training or to test. This avoids leakage where earlier and later moves of
the same game would appear in different partitions while sharing hidden-hand
state.

The default train fraction is 75%.

The test set is used only after inference to
measure predictive performance.

---

## 14. Validation Metrics

The current validation reports track both recovery and prediction.

| Quantity | Meaning | Direction |
|---|---|---|
| `theta_true` | synthetic vector used to generate the opponent | fixed |
| `theta_posterior_mean` | learned mean $\mu$ of $q(\theta)$ | close to true theta |
| `theta_posterior_std` | learned uncertainty per feature | interpretation depends on data |
| `theta_l2_error` | $\|\mu-\theta_{\mathrm{true}}\|_2$ | lower is better |
| `test_loglik_delta` | posterior predictive test log-likelihood minus zero-theta baseline | higher is better |
| `test_mean_logp_delta` | per-move version of `test_loglik_delta` | higher is better |
| `final_elbo` | final variational objective value | useful mainly within comparable runs |


In the notebook we also use a probability multiplier:

$$
\exp(\texttt{test\_mean\_logp\_delta})
$$

This is the probability ratio of the posterior predictive
model relative to the zero-theta baseline. For example, a value of `1.06` means
about a 6% per-move probability improvement on the test set.


---

## References

- Southey, F., Bowling, M. P., Larson, B., Piccione, C., Burch, N., Billings,
  D., and Rayner, C. (2005). *Bayes' Bluff: Opponent Modelling in Poker*.
  <https://arxiv.org/abs/1207.1411>
