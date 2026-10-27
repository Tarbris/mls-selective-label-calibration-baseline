# Selective-label calibration workspace

Edit only method.py. Keep class Calibrator with fit(observed) and
predict(probabilities). `observed` is a sequence of `(raw_probability,
observed_label, propensity)` triples. The propensity is the disclosed
probability that this row's label is recorded, conditional on its raw
probability and actual label. It lies in (0,1]. Labels for unrecorded rows are
never sent to the method. `predict` receives raw probabilities and returns
one finite calibrated probability in [0,1] for each input, in the same order.

Training examples come from binary classifiers whose population calibration
curve can vary between episodes. Label recording depends on both the outcome
and the raw confidence, so the recorded subset is biased. A method should
estimate the population mapping rather than the empirical label frequency of
only recorded examples. The hidden evaluator scores root-mean-square error
against the latent population probability on held-out raw inputs. Lower is
better. The candidate receives no generator parameters, true population
probabilities, hidden instances, or reference code. Python 3.12 standard
library is sufficient; no network or GPU is needed.
