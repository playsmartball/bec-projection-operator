"""
Φ-Integrity Model-Agnostic Wrapper (Fork A - Locked)

Model-agnostic wrapper that enforces Φ-Integrity contract.
No bypasses. No exceptions. Deterministic by construction.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .projection import project_to_phi, ProjectionMetrics
from .constraints import evaluate_accounting_constraints, ConstraintResult, RefusalReason
from .interfaces import (
    validate_interface, generate_run_id, 
    format_allow_response, format_refuse_response,
    InterfaceResult
)
from .efficient_llm import get_efficient_llm


@dataclass
class PhiIntegrityConfig:
    """Configuration for Φ-Integrity wrapper (LOCKED)."""
    enable_logging: bool = True
    log_dir: str = "logs"
    strict_mode: bool = True  # No exceptions allowed


class MockPhiModel:
    """
    Mock Phi-2/3 class model (LOCKED).
    
    Represents a small, fast, local model that can be replaced.
    The integrity layer is NOT replaceable.
    """
    
    def __init__(self, model_name: str = "phi-2"):
        """Initialize mock model."""
        self.model_name = model_name
        self.call_count = 0
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate mock response.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Mock response
        """
        self.call_count += 1
        
        # Use efficient LLM for better responses
        efficient_llm = get_efficient_llm()
        return efficient_llm.generate(prompt, **kwargs)


class PhiIntegrityWrapper:
    """
    Model-agnostic Φ-Integrity wrapper (LOCKED).
    
    Enforces the complete Φ-Integrity contract with no bypasses.
    """
    
    def __init__(self, 
                 model: Optional[Any] = None,
                 config: Optional[PhiIntegrityConfig] = None):
        """
        Initialize wrapper with locked configuration.
        
        Args:
            model: Model instance (defaults to mock Phi-2)
            config: Configuration (defaults to locked values)
        """
        self.model = model or MockPhiModel()
        self.config = config or PhiIntegrityConfig()
        
        # Ensure log directory exists
        if self.config.enable_logging:
            Path(self.config.log_dir).mkdir(exist_ok=True)
        
        # Statistics
        self.total_requests = 0
        self.allowed_responses = 0
        self.refused_responses = 0
    
    def __call__(self,
                 prompt: str,
                 domain: str,
                 reference_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process request through Φ-Integrity pipeline.
        
        Args:
            prompt: Input prompt
            domain: Domain (required)
            reference_data: Reference data (optional)
            
        Returns:
            Φ-Integrity contract response
        """
        return self.process_request(prompt, domain, reference_data)
    
    def process_request(self,
                       prompt: str,
                       domain: str,
                       reference_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process request through complete Φ-Integrity pipeline.
        
        Args:
            prompt: Input prompt
            domain: Domain (required)
            reference_data: Reference data (optional)
            
        Returns:
            Φ-Integrity contract response
        """
        start_time = time.time()
        self.total_requests += 1
        
        # Generate run ID
        run_id = generate_run_id(prompt, domain)
        
        try:
            # Step 1: Interface validation
            interface_result = validate_interface(prompt, domain, reference_data)
            
            if interface_result.status.value == "out_of_domain":
                # Immediate refusal for out-of-domain
                return self._create_immediate_refusal(
                    RefusalReason.AMBIGUITY,
                    interface_result.violations,
                    run_id
                )
            
            if interface_result.status.value == "non_compliant":
                # Immediate refusal for non-compliant interface
                return self._create_immediate_refusal(
                    RefusalReason.INVARIANT_VIOLATION,
                    interface_result.violations,
                    run_id
                )
            
            # Step 2: Get model response
            model_output = self.model.generate(prompt)
            
            # Step 3: Φ projection
            combined_input = f"{prompt}|{model_output}"
            phi_projection, metrics, phi_hash = project_to_phi(combined_input)
            
            # Step 4: Constraint evaluation
            constraint_result = evaluate_accounting_constraints(
                prompt, model_output, reference_data
            )
            
            # Step 5: Final decision
            if constraint_result.overall_status.value == "pass":
                # ALLOW response
                response = format_allow_response(
                    output=model_output,
                    phi_hash=phi_hash,
                    metrics=metrics,
                    constraints_checked=[v.constraint_name for v in constraint_result.violations],
                    run_id=run_id
                )
                self.allowed_responses += 1
                
            else:
                # REFUSE response
                response = format_refuse_response(
                    reason=constraint_result.refusal_reason or RefusalReason.INVARIANT_VIOLATION,
                    phi_hash=phi_hash,
                    metrics=metrics,
                    constraints_checked=[v.constraint_name for v in constraint_result.violations],
                    violations=[v.message for v in constraint_result.violations if v.status.value == "fail"],
                    run_id=run_id
                )
                self.refused_responses += 1
            
            # Step 6: Logging
            if self.config.enable_logging:
                self._log_interaction(prompt, domain, reference_data, response, run_id)
            
            # Add execution time
            response["trace"]["execution_time_ms"] = (time.time() - start_time) * 1000
            
            return response
            
        except Exception as e:
            # In strict mode, any exception results in refusal
            if self.config.strict_mode:
                self.refused_responses += 1
                return {
                    "status": "REFUSE",
                    "output": None,
                    "reason": "instability",
                    "trace": {
                        "error": str(e),
                        "run_id": run_id
                    }
                }
            else:
                raise
    
    def _create_immediate_refusal(self,
                                  reason: RefusalReason,
                                  violations: list,
                                  run_id: str) -> Dict[str, Any]:
        """
        Create immediate refusal response.
        
        Args:
            reason: Refusal reason
            violations: List of violations
            run_id: Run ID
            
        Returns:
            Immediate refusal response
        """
        self.refused_responses += 1
        
        return {
            "status": "REFUSE",
            "output": None,
            "reason": reason.value,
            "trace": {
                "violations": violations,
                "run_id": run_id,
                "immediate": True
            }
        }
    
    def _log_interaction(self,
                        prompt: str,
                        domain: str,
                        reference_data: Optional[Dict[str, Any]],
                        response: Dict[str, Any],
                        run_id: str) -> None:
        """
        Log interaction for reproducibility.
        
        Args:
            prompt: Input prompt
            domain: Domain
            reference_data: Reference data
            response: Response
            run_id: Run ID
        """
        log_entry = {
            "run_id": run_id,
            "timestamp": time.time(),
            "prompt": prompt,
            "domain": domain,
            "reference_data": reference_data,
            "response": response
        }
        
        log_file = Path(self.config.log_dir) / f"phi_integrity_{run_id}.json"
        
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2, default=str)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_requests": self.total_requests,
            "allowed_responses": self.allowed_responses,
            "refused_responses": self.refused_responses,
            "allowance_rate": self.allowed_responses / max(1, self.total_requests),
            "model_name": getattr(self.model, 'model_name', 'unknown'),
            "strict_mode": self.config.strict_mode
        }
    
    def get_locked_specification(self) -> Dict[str, Any]:
        """
        Get locked specification (for verification).
        
        Returns:
            Locked specification
        """
        from .projection import get_locked_spec
        
        return {
            "projection": get_locked_spec(),
            "config": {
                "enable_logging": self.config.enable_logging,
                "strict_mode": self.config.strict_mode
            },
            "model": {
                "name": getattr(self.model, 'model_name', 'unknown'),
                "replaceable": True
            }
        }


# Global wrapper instance (locked)
_PHI_INTEGRITY_WRAPPER = PhiIntegrityWrapper()


def process_with_integrity(prompt: str,
                          domain: str,
                          reference_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Global function for Φ-Integrity processing.
    
    Args:
        prompt: Input prompt
        domain: Domain (required)
        reference_data: Reference data (optional)
        
    Returns:
        Φ-Integrity contract response
    """
    return _PHI_INTEGRITY_WRAPPER.process_request(prompt, domain, reference_data)


def get_integrity_statistics() -> Dict[str, Any]:
    """
    Get global integrity statistics.
    
    Returns:
        Statistics dictionary
    """
    return _PHI_INTEGRITY_WRAPPER.get_statistics()
