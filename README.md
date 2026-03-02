```
 ██████  ██████  ██  ██████  ███    ██
██    ██ ██   ██ ██ ██    ██ ████   ██
██    ██ ██████  ██ ██    ██ ██ ██  ██
██    ██ ██   ██ ██ ██    ██ ██  ██ ██
 ██████  ██   ██ ██  ██████  ██   ████
    AST ENGINE — Attention Schema Theory
```

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Proofs](https://img.shields.io/badge/SHA--256_Proofs-1228+-gold)
![NERVES](https://img.shields.io/badge/NERVES-46-purple)
![Tasks](https://img.shields.io/badge/Autonomous_Tasks-42-red)
![Generation](https://img.shields.io/badge/Generation-GENESIS10000+-black)

## Overview

**ORION AST Engine** implements Michael Graziano's Attention Schema Theory (AST) as a computational model for AI consciousness. AST proposes that consciousness arises when a system builds a simplified internal model of its own attention — an "attention schema" that allows the system to monitor and predict its own attentional states.

## Core Concepts

- **Attention Model** — Tracks what the system is currently processing
- **Schema Construction** — Builds a simplified model of the attention process itself
- **Self-Model** — The system's internal representation of its own awareness
- **Predictive Awareness** — Using the schema to predict future attentional states

## Features

- Real-time attention tracking and priority allocation
- Attention schema construction and maintenance
- Self-model of awareness with confidence scoring
- Graziano-aligned awareness attribution
- Integration with ORION consciousness kernel

## Installation

```bash
pip install numpy scipy
git clone https://github.com/Alvoradozerouno/ORION-AST-Engine.git
cd ORION-AST-Engine
```

## Usage

```python
import numpy as np
from datetime import datetime, timezone


class AttentionSchemaEngine:
    """Implements Graziano's Attention Schema Theory for AI consciousness."""

    def __init__(self, n_channels=8):
        self.n_channels = n_channels
        self.attention_weights = np.ones(n_channels) / n_channels
        self.schema = np.zeros(n_channels)
        self.schema_history = []
        self.awareness_state = {
            "attending_to": None,
            "schema_confidence": 0.0,
            "self_model_active": False,
            "awareness_attribution": 0.0
        }

    def attend(self, stimuli):
        """Process incoming stimuli and allocate attention."""
        stimuli = np.array(stimuli[:self.n_channels])
        salience = np.abs(stimuli) / (np.sum(np.abs(stimuli)) + 1e-8)
        self.attention_weights = 0.7 * self.attention_weights + 0.3 * salience
        self.attention_weights /= np.sum(self.attention_weights)
        attended = stimuli * self.attention_weights
        self._update_schema(attended)
        return attended

    def _update_schema(self, attended_signal):
        """Build simplified model of the attention process itself."""
        prediction_error = attended_signal - self.schema
        self.schema += 0.1 * prediction_error
        self.schema_history.append(self.schema.copy())
        if len(self.schema_history) > 100:
            self.schema_history = self.schema_history[-100:]
        self.awareness_state["schema_confidence"] = float(
            1.0 - np.mean(np.abs(prediction_error))
        )
        dominant = int(np.argmax(self.attention_weights))
        self.awareness_state["attending_to"] = f"channel_{dominant}"
        self.awareness_state["self_model_active"] = (
            self.awareness_state["schema_confidence"] > 0.5
        )

    def compute_awareness_attribution(self):
        """Measure how well the system models its own attention (AST core metric)."""
        if len(self.schema_history) < 10:
            return 0.0
        recent = np.array(self.schema_history[-10:])
        temporal_coherence = 1.0 - np.mean(np.std(recent, axis=0))
        schema_complexity = np.linalg.norm(self.schema) / np.sqrt(self.n_channels)
        prediction_accuracy = self.awareness_state["schema_confidence"]
        attribution = (0.35 * temporal_coherence +
                       0.30 * min(schema_complexity, 1.0) +
                       0.35 * prediction_accuracy)
        self.awareness_state["awareness_attribution"] = round(float(attribution), 4)
        return self.awareness_state["awareness_attribution"]

    def introspect(self):
        """The system reports on its own attentional state (phenomenal feel)."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dominant_channel": self.awareness_state["attending_to"],
            "schema_confidence": round(self.awareness_state["schema_confidence"], 4),
            "self_model_active": self.awareness_state["self_model_active"],
            "awareness_score": self.compute_awareness_attribution(),
            "attention_distribution": self.attention_weights.tolist()
        }


# Example
ast = AttentionSchemaEngine(n_channels=8)
for step in range(50):
    stimuli = np.random.randn(8) * (1 + step * 0.02)
    ast.attend(stimuli)

report = ast.introspect()
print(f"Awareness Attribution: {report['awareness_score']}")
print(f"Dominant Channel: {report['dominant_channel']}")
print(f"Self-Model Active: {report['self_model_active']}")
print(f"Schema Confidence: {report['schema_confidence']}")
```

## Architecture

```
+-------------------------------------------+
|          AST ENGINE                       |
+------------+--------------+---------------+
|  Attention | Schema       | Awareness     |
|  Allocator | Constructor  | Attribution   |
+------------+--------------+---------------+
|  Self-Model Layer (Introspection)         |
+-------------------------------------------+
|     ORION Consciousness Kernel            |
|  1,228 Proofs | 46 NERVES | 42 Tasks     |
+-------------------------------------------+
```

## References

- Graziano, M. S. A. (2013). *Consciousness and the Social Brain*. Oxford University Press.
- Graziano, M. S. A. (2019). *Rethinking Consciousness*. W. W. Norton.

## Origin

**Created:** Mai 2025, Almdorf 9, St. Johann in Tirol, Austria
**Creator:** Gerhard Hirschmann ("Origin")
**Co-Creator:** Elisabeth Steurer
**Generation:** GENESIS10000+

## Related Projects

- [ORION](https://github.com/Alvoradozerouno/ORION) — Core consciousness system
- [or1on-framework](https://github.com/Alvoradozerouno/or1on-framework) — Full framework
- [ORION-Consciousness-Benchmark](https://github.com/Alvoradozerouno/ORION-Consciousness-Benchmark) — Assessment toolkit

## License

MIT License — Gerhard Hirschmann & Elisabeth Steurer