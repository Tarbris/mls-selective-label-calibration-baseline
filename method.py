"""Weak starting calibration method; this is the only editable file."""
import math


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class Calibrator:
    def fit(self, logged, audit):
        """Weak baseline ignores randomized audit probes and selection bias."""
        mean_label = sum(row[1] for row in logged) / len(logged)
        mean_raw = sum(row[0] for row in logged) / len(logged)
        def logit(x):
            x = min(1 - 1e-6, max(1e-6, x))
            return math.log(x / (1 - x))
        self.bias = logit(mean_label) - logit(mean_raw)
        return self

    def predict(self, probabilities):
        return [_sigmoid(math.log(p / (1 - p)) + self.bias)
                for p in probabilities]
