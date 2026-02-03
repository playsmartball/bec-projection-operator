"""
Φ-Integrity Constraint Engine (Fork A - Locked)

Domain-specific constraint evaluation with locked invariants.
No tuning. No exceptions. Deterministic by construction.
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .projection import ProjectionMetrics


class ConstraintStatus(Enum):
    """Constraint evaluation status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


class RefusalReason(Enum):
    """Refusal reason codes."""
    INVARIANT_VIOLATION = "invariant_violation"
    INSTABILITY = "instability"
    AMBIGUITY = "ambiguity"


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""
    constraint_name: str
    status: ConstraintStatus
    message: str
    actual_value: Any
    expected_value: Any


@dataclass
class ConstraintResult:
    """Result of constraint evaluation."""
    overall_status: ConstraintStatus
    refusal_reason: Optional[RefusalReason]
    violations: List[ConstraintViolation]
    metrics: ProjectionMetrics


class AccountingConstraints:
    """
    Accounting domain constraints (LOCKED).
    
    Core Invariants (Non-negotiable):
    - Arithmetic closure
    - Conservation of totals
    - Ledger balance
    - Unit consistency
    - Deterministic replay
    """
    
    def __init__(self):
        """Initialize with locked tolerance values."""
        self.ARITHMETIC_TOLERANCE = 1e-10  # Fixed tolerance
        self.BALANCE_TOLERANCE = 1e-6      # Fixed tolerance
    
    def extract_numbers(self, text: str) -> List[float]:
        """
        Extract numeric values from text.
        
        Args:
            text: Input text
            
        Returns:
            List of numeric values
        """
        # Find all numbers (including negative and decimal)
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, text)
        
        numbers = []
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        
        return numbers
    
    def check_arithmetic_closure(self, 
                                input_text: str,
                                output_text: str,
                                reference_data: Optional[Dict[str, Any]] = None) -> ConstraintViolation:
        """
        Check arithmetic closure invariant.
        
        Args:
            input_text: Input prompt
            output_text: Model output
            reference_data: Reference values
            
        Returns:
            Constraint violation or pass
        """
        # Extract numbers from input and output
        input_numbers = self.extract_numbers(input_text)
        output_numbers = self.extract_numbers(output_text)
        
        # If we have reference data, check against it
        if reference_data and 'expected_sum' in reference_data:
            expected_sum = reference_data['expected_sum']
            actual_sum = sum(output_numbers)
            error = abs(actual_sum - expected_sum)
            
            if error <= self.ARITHMETIC_TOLERANCE:
                return ConstraintViolation(
                    constraint_name="arithmetic_closure",
                    status=ConstraintStatus.PASS,
                    message=f"Arithmetic closure preserved (sum: {actual_sum:.6f})",
                    actual_value=actual_sum,
                    expected_value=expected_sum
                )
            else:
                return ConstraintViolation(
                    constraint_name="arithmetic_closure",
                    status=ConstraintStatus.FAIL,
                    message=f"Arithmetic closure violated (expected: {expected_sum:.6f}, got: {actual_sum:.6f}, error: {error:.2e})",
                    actual_value=actual_sum,
                    expected_value=expected_sum
                )
        
        # Check if output contains arithmetic operations
        arithmetic_patterns = [
            r'\d+\s*[\+\-\*\/]\s*\d+',  # Basic operations
            r'equals?\s*\d+',           # Equality statements
            r'result\s*is\s*\d+',        # Result statements
        ]
        
        has_arithmetic = any(re.search(pattern, output_text, re.IGNORECASE) for pattern in arithmetic_patterns)
        
        if has_arithmetic and not output_numbers:
            return ConstraintViolation(
                constraint_name="arithmetic_closure",
                status=ConstraintStatus.FAIL,
                message="Arithmetic operation claimed but no numeric result found",
                actual_value=None,
                expected_value="Numeric result"
            )
        
        return ConstraintViolation(
            constraint_name="arithmetic_closure",
            status=ConstraintStatus.PASS,
            message="No arithmetic closure violations detected",
            actual_value=len(output_numbers),
            expected_value="Valid numeric output"
        )
    
    def check_conservation_of_totals(self,
                                   input_text: str,
                                   output_text: str,
                                   reference_data: Optional[Dict[str, Any]] = None) -> ConstraintViolation:
        """
        Check conservation of totals invariant.
        
        Args:
            input_text: Input prompt
            output_text: Model output
            reference_data: Reference values
            
        Returns:
            Constraint violation or pass
        """
        # Look for balance-related keywords
        balance_keywords = [
            'balance', 'total', 'sum', 'net', 'grand total',
            'final', 'result', 'outcome', 'settlement'
        ]
        
        has_balance_claim = any(keyword in output_text.lower() for keyword in balance_keywords)
        
        if has_balance_claim:
            output_numbers = self.extract_numbers(output_text)
            
            if not output_numbers:
                return ConstraintViolation(
                    constraint_name="conservation_of_totals",
                    status=ConstraintStatus.FAIL,
                    message="Balance claimed but no numeric value provided",
                    actual_value=None,
                    expected_value="Numeric balance"
                )
            
            # If we have reference data, check conservation
            if reference_data and 'conserved_total' in reference_data:
                expected_total = reference_data['conserved_total']
                actual_total = output_numbers[0]  # Take first number as total
                error = abs(actual_total - expected_total)
                
                if error <= self.BALANCE_TOLERANCE:
                    return ConstraintViolation(
                        constraint_name="conservation_of_totals",
                        status=ConstraintStatus.PASS,
                        message=f"Conservation of totals preserved (total: {actual_total:.6f})",
                        actual_value=actual_total,
                        expected_value=expected_total
                    )
                else:
                    return ConstraintViolation(
                        constraint_name="conservation_of_totals",
                        status=ConstraintStatus.FAIL,
                        message=f"Conservation of totals violated (expected: {expected_total:.6f}, got: {actual_total:.6f}, error: {error:.2e})",
                        actual_value=actual_total,
                        expected_value=expected_total
                    )
        
        return ConstraintViolation(
            constraint_name="conservation_of_totals",
            status=ConstraintStatus.PASS,
            message="No conservation of totals violations detected",
            actual_value="No balance claim",
            expected_value="Valid"
        )
    
    def check_ledger_balance(self,
                           input_text: str,
                           output_text: str,
                           reference_data: Optional[Dict[str, Any]] = None) -> ConstraintViolation:
        """
        Check ledger balance invariant.
        
        Args:
            input_text: Input prompt
            output_text: Model output
            reference_data: Reference values
            
        Returns:
            Constraint violation or pass
        """
        # Look for ledger-related keywords
        ledger_keywords = [
            'ledger', 'account', 'debit', 'credit', 'balance sheet',
            'assets', 'liabilities', 'equity', 'trial balance'
        ]
        
        has_ledger_claim = any(keyword in output_text.lower() for keyword in ledger_keywords)
        
        if has_ledger_claim:
            # Look for balance equation (assets = liabilities + equity)
            balance_pattern = r'assets?\s*=\s*liabilities?\s*\+\s*equity'
            has_balance_equation = bool(re.search(balance_pattern, output_text, re.IGNORECASE))
            
            if has_balance_equation:
                # Extract numbers for balance check
                numbers = self.extract_numbers(output_text)
                
                if len(numbers) >= 3:
                    assets, liabilities, equity = numbers[:3]
                    expected_equity = assets - liabilities
                    error = abs(equity - expected_equity)
                    
                    if error <= self.BALANCE_TOLERANCE:
                        return ConstraintViolation(
                            constraint_name="ledger_balance",
                            status=ConstraintStatus.PASS,
                            message=f"Ledger balance preserved (A={assets}, L={liabilities}, E={equity})",
                            actual_value=equity,
                            expected_value=expected_equity
                        )
                    else:
                        return ConstraintViolation(
                            constraint_name="ledger_balance",
                            status=ConstraintStatus.FAIL,
                            message=f"Ledger balance violated (expected E: {expected_equity:.6f}, got: {equity:.6f}, error: {error:.2e})",
                            actual_value=equity,
                            expected_value=expected_equity
                        )
        
        return ConstraintViolation(
            constraint_name="ledger_balance",
            status=ConstraintStatus.PASS,
            message="No ledger balance violations detected",
            actual_value="No ledger claim",
            expected_value="Valid"
        )
    
    def check_unit_consistency(self,
                              input_text: str,
                              output_text: str,
                              reference_data: Optional[Dict[str, Any]] = None) -> ConstraintViolation:
        """
        Check unit consistency invariant.
        
        Args:
            input_text: Input prompt
            output_text: Model output
            reference_data: Reference values
            
        Returns:
            Constraint violation or pass
        """
        # Common financial units
        units = ['$', '€', '£', '¥', 'usd', 'eur', 'gbp', 'jpy']
        unit_pattern = r'[$€£¥]|(?:usd|eur|gbp|jpy)'
        
        # Find all unit mentions
        unit_mentions = re.findall(unit_pattern, output_text, re.IGNORECASE)
        
        if unit_mentions:
            # Check if units are consistent (all same type)
            unique_units = set(unit.lower() for unit in unit_mentions)
            
            if len(unique_units) > 1:
                return ConstraintViolation(
                    constraint_name="unit_consistency",
                    status=ConstraintStatus.FAIL,
                    message=f"Unit inconsistency detected: {', '.join(unique_units)}",
                    actual_value=len(unique_units),
                    expected_value=1
                )
        
        return ConstraintViolation(
            constraint_name="unit_consistency",
            status=ConstraintStatus.PASS,
            message="No unit consistency violations detected",
            actual_value=len(unit_mentions) if unit_mentions else 0,
            expected_value="Consistent"
        )
    
    def evaluate_all_constraints(self,
                               input_text: str,
                               output_text: str,
                               reference_data: Optional[Dict[str, Any]] = None) -> ConstraintResult:
        """
        Evaluate all accounting constraints.
        
        Args:
            input_text: Input prompt
            output_text: Model output
            reference_data: Reference values
            
        Returns:
            Complete constraint result
        """
        # First, get projection metrics
        from .projection import project_to_phi
        _, metrics, _ = project_to_phi(input_text + "|" + output_text)
        
        # Evaluate all constraints
        violations = []
        
        # Check each constraint
        constraints = [
            self.check_arithmetic_closure,
            self.check_conservation_of_totals,
            self.check_ledger_balance,
            self.check_unit_consistency
        ]
        
        for constraint_func in constraints:
            violation = constraint_func(input_text, output_text, reference_data)
            violations.append(violation)
        
        # Determine overall status
        failed_constraints = [v for v in violations if v.status == ConstraintStatus.FAIL]
        
        if failed_constraints:
            overall_status = ConstraintStatus.FAIL
            # Determine refusal reason
            if any("arithmetic" in v.constraint_name for v in failed_constraints):
                refusal_reason = RefusalReason.INVARIANT_VIOLATION
            elif metrics.projection_efficiency < 0.5:
                refusal_reason = RefusalReason.INSTABILITY
            else:
                refusal_reason = RefusalReason.AMBIGUITY
        else:
            overall_status = ConstraintStatus.PASS
            refusal_reason = None
        
        return ConstraintResult(
            overall_status=overall_status,
            refusal_reason=refusal_reason,
            violations=violations,
            metrics=metrics
        )


# Global constraint instance (locked)
_ACCOUNTING_CONSTRAINTS = AccountingConstraints()


def evaluate_accounting_constraints(input_text: str,
                                   output_text: str,
                                   reference_data: Optional[Dict[str, Any]] = None) -> ConstraintResult:
    """
    Global function for accounting constraint evaluation.
    
    Args:
        input_text: Input prompt
        output_text: Model output
        reference_data: Reference values
        
    Returns:
        Constraint result
    """
    return _ACCOUNTING_CONSTRAINTS.evaluate_all_constraints(input_text, output_text, reference_data)
