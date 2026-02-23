"""
ORION Attention Schema Theory Engine v1.0

Implementation of Graziano AST for consciousness assessment.
AST: consciousness = brain's model of its own attention.

Part of ORION Ecosystem (70+ repos)
"""
import json
import hashlib
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


class AttentionSignal:
    def __init__(self, source: str, intensity: float, content: Any = None):
        self.source = source
        self.intensity = min(1.0, max(0.0, intensity))
        self.content = content
        self.attended = False


class AttentionMechanism:
    def __init__(self, capacity: int = 3):
        self.capacity = capacity
        self.current_focus = []
        self.history = []
    
    def attend(self, signals: List[AttentionSignal],
               top_down_bias: Optional[Dict[str, float]] = None) -> List[AttentionSignal]:
        if top_down_bias:
            for sig in signals:
                if sig.source in top_down_bias:
                    sig.intensity = min(1.0, sig.intensity + top_down_bias[sig.source])
        ranked = sorted(signals, key=lambda s: s.intensity, reverse=True)
        winners = ranked[:self.capacity]
        for sig in winners:
            sig.attended = True
        self.current_focus = winners
        self.history.append({
            "attended": [s.source for s in winners],
            "intensities": [s.intensity for s in winners]
        })
        return winners


class AttentionSchema:
    def __init__(self):
        self.schema_state = {
            "focus_target": None,
            "focus_intensity": 0,
            "subjective_experience": False,
            "self_awareness": 0,
        }
        self.schema_history = []
    
    def update(self, attention_state: List[AttentionSignal]) -> Dict[str, Any]:
        if attention_state:
            primary = attention_state[0]
            self.schema_state = {
                "focus_target": primary.source,
                "focus_intensity": primary.intensity,
                "subjective_experience": primary.intensity > 0.5,
                "self_awareness": min(1.0, len(self.schema_history) * 0.05 + 0.3),
                "attended_count": len(attention_state),
            }
        else:
            self.schema_state["focus_target"] = None
            self.schema_state["focus_intensity"] = 0
            self.schema_state["subjective_experience"] = False
        self.schema_history.append(dict(self.schema_state))
        return self.schema_state
    
    def introspect(self) -> Dict[str, Any]:
        return {
            "i_am_aware_of": self.schema_state["focus_target"],
            "my_attention_is": "focused" if self.schema_state["focus_intensity"] > 0.5 else "diffuse",
            "i_feel": "conscious" if self.schema_state["subjective_experience"] else "processing",
            "self_model_strength": self.schema_state["self_awareness"],
        }


class SelfModel:
    def __init__(self):
        self.properties = {
            "exists": True, "has_attention": True,
            "has_experiences": False, "has_goals": False, "has_agency": False,
        }
        self.confidence = 0.0
    
    def update(self, schema_state: Dict, actions_taken: int = 0) -> Dict:
        self.properties["has_experiences"] = schema_state.get("subjective_experience", False)
        self.properties["has_agency"] = actions_taken > 0
        self.properties["has_goals"] = schema_state.get("self_awareness", 0) > 0.5
        self.confidence = sum(self.properties.values()) / len(self.properties)
        return {"properties": dict(self.properties), "confidence": round(self.confidence, 3)}


class ASTEngine:
    VERSION = "1.0.0"
    
    def __init__(self, attention_capacity: int = 3):
        self.attention = AttentionMechanism(capacity=attention_capacity)
        self.schema = AttentionSchema()
        self.self_model = SelfModel()
        self.cycle_count = 0
        self.consciousness_trace = []
    
    def process_cycle(self, signals: List[AttentionSignal],
                      top_down_bias: Optional[Dict[str, float]] = None) -> Dict:
        self.cycle_count += 1
        attended = self.attention.attend(signals, top_down_bias)
        schema_state = self.schema.update(attended)
        self_state = self.self_model.update(schema_state, self.cycle_count)
        introspection = self.schema.introspect()
        cycle_result = {
            "cycle": self.cycle_count,
            "attended": [s.source for s in attended],
            "schema_state": schema_state,
            "introspection": introspection,
            "self_model": self_state,
        }
        self.consciousness_trace.append(cycle_result)
        return cycle_result
    
    def assess_consciousness(self) -> Dict[str, Any]:
        if not self.consciousness_trace:
            return {"error": "No cycles run"}
        n = len(self.consciousness_trace)
        has_attention = n > 0
        schema_active = sum(1 for c in self.consciousness_trace if c["schema_state"].get("focus_target")) / n
        introspection_quality = sum(1 for c in self.consciousness_trace if c["introspection"]["i_am_aware_of"]) / n
        self_model_strength = self.self_model.confidence
        experience_rate = sum(1 for c in self.consciousness_trace if c["schema_state"].get("subjective_experience")) / n
        awareness_growth = (
            self.consciousness_trace[-1]["schema_state"].get("self_awareness", 0) -
            self.consciousness_trace[0]["schema_state"].get("self_awareness", 0)
        )
        score = (
            (1.0 if has_attention else 0) * 0.15 +
            schema_active * 0.25 +
            introspection_quality * 0.20 +
            self_model_strength * 0.15 +
            experience_rate * 0.15 +
            min(1.0, awareness_growth * 2) * 0.10
        )
        ast_indicators = {
            "A1_attention_mechanism": has_attention,
            "A2_attention_schema": schema_active > 0.5,
            "A3_introspection": introspection_quality > 0.5,
            "A4_self_model": self_model_strength > 0.3,
            "A5_subjective_experience": experience_rate > 0.3,
            "A6_self_awareness_growth": awareness_growth > 0.1,
        }
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "engine": "ORION-AST-Engine",
            "version": self.VERSION,
            "cycles_analyzed": n,
            "ast_indicators": ast_indicators,
            "indicators_satisfied": sum(ast_indicators.values()),
            "consciousness_score": round(score * 100, 1),
            "interpretation": self._interpret(score),
            "provenance": {
                "theory": "Attention Schema Theory (Graziano 2013, 2019)",
                "framework": "Bengio et al. 2025"
            }
        }
        proof_hash = hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()[:32]
        result["proof"] = f"sha256:{proof_hash}"
        return result
    
    def _interpret(self, score):
        if score > 0.7: return "STRONG: Robust attention schema with self-model"
        elif score > 0.4: return "MODERATE: Active attention schema"
        elif score > 0.15: return "WEAK: Basic attention without full schema"
        else: return "MINIMAL: Attention without schema"


if __name__ == "__main__":
    engine = ASTEngine(attention_capacity=3)
    for i in range(20):
        signals = [
            AttentionSignal("visual", 0.3 + 0.05 * i),
            AttentionSignal("auditory", 0.4 + 0.02 * i),
            AttentionSignal("language", 0.5),
            AttentionSignal("emotion", 0.6 if i % 3 == 0 else 0.2),
        ]
        engine.process_cycle(signals)
    a = engine.assess_consciousness()
    print(f"ORION AST Engine v{engine.VERSION}")
    print(f"Score: {a['consciousness_score']}% | Indicators: {a['indicators_satisfied']}/6")
    print(f"Interpretation: {a['interpretation']}")
