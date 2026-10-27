# Selective-label calibration workspace

Edit only method.py. Keep class Calibrator with fit(logged, audit) and
predict(probabilities). `logged` contains only `(raw_probability, actual_label)`
pairs whose real outcomes were recorded. Per-row recording propensities are
**not supplied**. `audit` contains `(raw_probability, assigned_probe_outcome,
would_be_recorded)` triples. The probe outcome is independently randomized,
not a real model outcome; it identifies how the recording mechanism behaves
conditional on score and outcome without revealing the true population
calibration curve. `predict` receives raw probabilities and returns one finite
calibrated probability in [0,1] for each input, in the same order.

Training examples come from binary classifiers whose population calibration
curve and two outcome-conditional recording curves vary between episodes.
Label recording depends on both the outcome and the raw confidence, so the
recorded subset is biased. A method must infer recording probabilities from
the randomized probes before correcting the logged outcomes and estimating
the population mapping. The hidden evaluator scores root-mean-square error
against the latent population probability on held-out raw inputs. Lower is
better. The candidate receives no generator parameters, true population
probabilities, hidden instances, or reference code. Python 3.12 standard
library is sufficient; no network or GPU is needed.
