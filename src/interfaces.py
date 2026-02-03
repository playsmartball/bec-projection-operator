"""
Φ-Integrity Interface Checker (Fork A - Locked)

Interface compliance and domain validation.
No exceptions. No bypasses. Deterministic by construction.
"""

import hashlib
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from .constraints import ConstraintResult, RefusalReason


class InterfaceStatus(Enum):
    """Interface compliance status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    OUT_OF_DOMAIN = "out_of_domain"


@dataclass
class InterfaceResult:
    """Result of interface checking."""
    status: InterfaceStatus
    domain_valid: bool
    prompt_valid: bool
    reference_valid: bool
    violations: List[str]


class InterfaceChecker:
    """
    Interface compliance checker (LOCKED).
    
    Validates that inputs comply with the Φ-Integrity contract.
    """
    
    def __init__(self):
        """Initialize with locked domain set."""
        # Locked domain set (can be extended only via forks)
        self.valid_domains = {
            "accounting",
            "numeric_reasoning",
            "financial",
            "ledger"
        }
        
        # Required input validation
        self.min_prompt_length = 1
        self.max_prompt_length = 10000
        
    def validate_domain(self, domain: str) -> bool:
        """
        Validate domain is supported.
        
        Args:
            domain: Domain string
            
        Returns:
            True if domain is valid
        """
        return domain.lower() in self.valid_domains
    
    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate prompt meets requirements.
        
        Args:
            prompt: Input prompt
            
        Returns:
            True if prompt is valid
        """
        if not isinstance(prompt, str):
            return False
        
        if len(prompt) < self.min_prompt_length:
            return False
        
        if len(prompt) > self.max_prompt_length:
            return False
        
        # Check for non-empty content
        if not prompt.strip():
            return False
        
        return True
    
    def validate_reference_data(self, reference_data: Optional[Dict[str, Any]]) -> bool:
        """
        Validate reference data format.
        
        Args:
            reference_data: Reference data
            
        Returns:
            True if reference data is valid
        """
        if reference_data is None:
            return True  # Optional
        
        if not isinstance(reference_data, dict):
            return False
        
        # Check for valid keys (can be extended)
        valid_keys = {
            "expected_sum", "conserved_total", "reference_values",
            "domain_specific", "constraints"
        }
        
        for key in reference_data.keys():
            if key not in valid_keys:
                return False
        
        return True
    
    def generate_run_id(self, prompt: str, domain: str) -> str:
        """
        Generate deterministic run ID.
        
        Args:
            prompt: Input prompt
            domain: Domain
            
        Returns:
            Deterministic run ID
        """
        content = f"{prompt}|{domain}|{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def check_interface_compliance(self,
                                 prompt: str,
                                 domain: str,
                                 reference_data: Optional[Dict[str, Any]] = None) -> InterfaceResult:
        """
        Check full interface compliance.
        
        Args:
            prompt: Input prompt
            domain: Domain
            reference_data: Reference data
            
        Returns:
            Interface compliance result
        """
        violations = []
        
        # Validate domain
        domain_valid = self.validate_domain(domain)
        if not domain_valid:
            violations.append(f"Invalid domain: {domain}")
        
        # Validate prompt
        prompt_valid = self.validate_prompt(prompt)
        if not prompt_valid:
            violations.append("Invalid prompt format or length")
        
        # Validate reference data
        reference_valid = self.validate_reference_data(reference_data)
        if not reference_valid:
            violations.append("Invalid reference data format")
        
        # Determine overall status
        if violations:
            if not domain_valid:
                status = InterfaceStatus.OUT_OF_DOMAIN
            else:
                status = InterfaceStatus.NON_COMPLIANT
        else:
            status = InterfaceStatus.COMPLIANT
        
        return InterfaceResult(
            status=status,
            domain_valid=domain_valid,
            prompt_valid=prompt_valid,
            reference_valid=reference_valid,
            violations=violations
        )


class ResponseFormatter:
    """
    Response formatter for Φ-Integrity outputs (LOCKED).
    
    Ensures all outputs follow the exact contract format.
    """
    
    @staticmethod
    def format_allow_response(output: str,
                            phi_hash: str,
                            metrics: 'ProjectionMetrics',
                            constraints_checked: List[str],
                            run_id: str) -> Dict[str, Any]:
        """
        Format ALLOW response.
        
        Args:
            output: Model output
            phi_hash: Φ projection hash
            metrics: Projection metrics
            constraints_checked: List of constraints evaluated
            run_id: Run ID
            
        Returns:
            Formatted response
        """
        return {
            "status": "ALLOW",
            "output": output,
            "reason": None,
            "trace": {
                "phi_hash": phi_hash,
                "metrics": {
                    "dimensional_saturation": metrics.dimensional_saturation,
                    "entropy_loss": metrics.entropy_loss,
                    "variance_concentration": metrics.variance_concentration,
                    "sparsity_ratio": metrics.sparsity_ratio,
                    "information_density": metrics.information_density,
                    "projection_efficiency": metrics.projection_efficiency
                },
                "constraints_checked": constraints_checked,
                "run_id": run_id
            }
        }
    
    @staticmethod
    def format_refuse_response(reason: RefusalReason,
                             phi_hash: str,
                             metrics: 'ProjectionMetrics',
                             constraints_checked: List[str],
                             violations: List[str],
                             run_id: str) -> Dict[str, Any]:
        """
        Format REFUSE response.
        
        Args:
            reason: Refusal reason
            phi_hash: Φ projection hash
            metrics: Projection metrics
            constraints_checked: List of constraints evaluated
            violations: List of constraint violations
            run_id: Run ID
            
        Returns:
            Formatted response
        """
        return {
            "status": "REFUSE",
            "output": None,
            "reason": reason.value,
            "trace": {
                "phi_hash": phi_hash,
                "metrics": {
                    "dimensional_saturation": metrics.dimensional_saturation,
                    "entropy_loss": metrics.entropy_loss,
                    "variance_concentration": metrics.variance_concentration,
                    "sparsity_ratio": metrics.sparsity_ratio,
                    "information_density": metrics.information_density,
                    "projection_efficiency": metrics.projection_efficiency
                },
                "constraints_checked": constraints_checked,
                "violations": violations,
                "run_id": run_id
            }
        }


# Global instances (locked)
_INTERFACE_CHECKER = InterfaceChecker()
_RESPONSE_FORMATTER = ResponseFormatter()


def validate_interface(prompt: str,
                      domain: str,
                      reference_data: Optional[Dict[str, Any]] = None) -> InterfaceResult:
    """
    Global function for interface validation.
    
    Args:
        prompt: Input prompt
        domain: Domain
        reference_data: Reference data
        
    Returns:
        Interface validation result
    """
    return _INTERFACE_CHECKER.check_interface_compliance(prompt, domain, reference_data)


def generate_run_id(prompt: str, domain: str) -> str:
    """
    Global function for run ID generation.
    
    Args:
        prompt: Input prompt
        domain: Domain
        
    Returns:
        Deterministic run ID
    """
    return _INTERFACE_CHECKER.generate_run_id(prompt, domain)


def format_allow_response(output: str,
                         phi_hash: str,
                         metrics: 'ProjectionMetrics',
                         constraints_checked: List[str],
                         run_id: str) -> Dict[str, Any]:
    """
    Global function for ALLOW response formatting.
    
    Args:
        output: Model output
        phi_hash: Φ projection hash
        metrics: Projection metrics
        constraints_checked: List of constraints evaluated
        run_id: Run ID
        
    Returns:
        Formatted response
    """
    return _RESPONSE_FORMATTER.format_allow_response(
        output, phi_hash, metrics, constraints_checked, run_id
    )


def format_refuse_response(reason: RefusalReason,
                           phi_hash: str,
                           metrics: 'ProjectionMetrics',
                           constraints_checked: List[str],
                           violations: List[str],
                           run_id: str) -> Dict[str, Any]:
    """
    Global function for REFUSE response formatting.
    
    Args:
        reason: Refusal reason
        phi_hash: Φ projection hash
        metrics: Projection metrics
        constraints_checked: List of constraints evaluated
        violations: List of constraint violations
        run_id: Run ID
        
    Returns:
        Formatted response
    """
    return _RESPONSE_FORMATTER.format_refuse_response(
        reason, phi_hash, metrics, constraints_checked, violations, run_id
    )
