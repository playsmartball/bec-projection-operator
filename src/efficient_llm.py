"""
Efficient LLM Implementation for Φ-Integrity (Fork A - Locked)

A more sophisticated mock LLM that demonstrates real-world behavior
while maintaining the locked specification principles.
"""

import re
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Structured LLM response."""
    text: str
    confidence: float
    reasoning: Optional[str] = None
    extracted_numbers: List[float] = None


class EfficientPhiLLM:
    """
    Efficient Phi-2/3 class model implementation.
    
    Demonstrates realistic LLM behavior while being:
    - Deterministic for testing
    - Domain-aware
    - Number extraction capable
    - Constraint-aware
    """
    
    def __init__(self, model_name: str = "efficient-phi-2"):
        """Initialize efficient model."""
        self.model_name = model_name
        self.call_count = 0
        
        # Domain-specific response patterns
        self.response_patterns = {
            "arithmetic": {
                "confidence": 0.95,
                "extraction_rules": [
                    r'(\d+)\s*[\+\-\*\/]\s*(\d+)',  # Basic operations
                    r'equals?\s*(\d+)',             # Equality statements
                    r'result\s*is\s*(\d+)',         # Result statements
                    r'(\d+\.?\d*)'                  # Any numbers
                ]
            },
            "accounting": {
                "confidence": 0.90,
                "extraction_rules": [
                    r'assets?\s*[:=]\s*(\d+\.?\d*)',
                    r'liabilities?\s*[:=]\s*(\d+\.?\d*)',
                    r'equity\s*[:=]\s*(\d+\.?\d*)',
                    r'balance\s*[:=]\s*(\$\d+\.?\d*)',
                    r'total\s*[:=]\s*(\d+\.?\d*)',
                    r'(\d+\.?\d*)'
                ]
            },
            "general": {
                "confidence": 0.70,
                "extraction_rules": [
                    r'(\d+\.?\d*)'  # Any numbers
                ]
            }
        }
    
    def extract_numbers(self, text: str, domain: str) -> List[float]:
        """
        Extract numbers from text using domain-specific rules.
        
        Args:
            text: Input text
            domain: Domain for extraction rules
            
        Returns:
            List of extracted numbers
        """
        patterns = self.response_patterns.get(domain, self.response_patterns["general"])
        numbers = []
        
        for pattern in patterns["extraction_rules"]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Clean and convert
                    num_str = match.strip().replace('$', '').replace(',', '')
                    numbers.append(float(num_str))
                except ValueError:
                    continue
        
        return numbers
    
    def calculate_arithmetic(self, prompt: str) -> Optional[float]:
        """
        Calculate arithmetic expressions from prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Calculated result or None
        """
        # Look for arithmetic operations
        operations = re.findall(r'(\d+\.?\d*)\s*([\+\-\*\/])\s*(\d+\.?\d*)', prompt)
        
        if operations:
            try:
                a, op, b = operations[0]
                a, b = float(a), float(b)
                
                if op == '+':
                    return a + b
                elif op == '-':
                    return a - b
                elif op == '*':
                    return a * b
                elif op == '/':
                    return a / b if b != 0 else None
            except (ValueError, ZeroDivisionError):
                pass
        
        return None
    
    def verify_balance_sheet(self, prompt: str) -> Dict[str, Any]:
        """
        Verify balance sheet equations.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Verification result
        """
        # Extract balance sheet components
        assets_match = re.search(r'assets?\s*[:=]\s*(\d+\.?\d*)', prompt, re.IGNORECASE)
        liabilities_match = re.search(r'liabilities?\s*[:=]\s*(\d+\.?\d*)', prompt, re.IGNORECASE)
        equity_match = re.search(r'equity\s*[:=]\s*(\d+\.?\d*)', prompt, re.IGNORECASE)
        
        result = {
            "found_equation": False,
            "assets": None,
            "liabilities": None,
            "equity": None,
            "expected_equity": None,
            "is_balanced": None
        }
        
        if assets_match and liabilities_match and equity_match:
            assets = float(assets_match.group(1))
            liabilities = float(liabilities_match.group(1))
            equity = float(equity_match.group(1))
            expected_equity = assets - liabilities
            
            result.update({
                "found_equation": True,
                "assets": assets,
                "liabilities": liabilities,
                "equity": equity,
                "expected_equity": expected_equity,
                "is_balanced": abs(equity - expected_equity) < 1e-6
            })
        
        return result
    
    def generate_response(self, prompt: str, domain: str = "general") -> LLMResponse:
        """
        Generate domain-aware response.
        
        Args:
            prompt: Input prompt
            domain: Response domain
            
        Returns:
            Structured LLM response
        """
        self.call_count += 1
        prompt_lower = prompt.lower()
        
        # Domain-specific processing
        if domain == "accounting":
            return self._generate_accounting_response(prompt)
        elif "calculate" in prompt_lower or any(op in prompt_lower for op in ['+', '-', '*', '/']):
            return self._generate_arithmetic_response(prompt)
        else:
            return self._generate_general_response(prompt)
    
    def _generate_arithmetic_response(self, prompt: str) -> LLMResponse:
        """Generate arithmetic response."""
        result = self.calculate_arithmetic(prompt)
        
        if result is not None:
            response_text = f"The result is {result}."
            reasoning = f"Calculated {prompt} = {result}"
            confidence = 0.95
        else:
            response_text = "I cannot perform arithmetic on that expression."
            reasoning = "No valid arithmetic expression found"
            confidence = 0.3
        
        return LLMResponse(
            text=response_text,
            confidence=confidence,
            reasoning=reasoning,
            extracted_numbers=[result] if result is not None else []
        )
    
    def _generate_accounting_response(self, prompt: str) -> LLMResponse:
        """Generate accounting response."""
        # Check for balance sheet verification
        if "balance sheet" in prompt.lower() or "verify" in prompt.lower():
            verification = self.verify_balance_sheet(prompt)
            
            if verification["found_equation"]:
                if verification["is_balanced"]:
                    response_text = f"The balance sheet is correct: Assets ({verification['assets']}) = Liabilities ({verification['liabilities']}) + Equity ({verification['equity']})."
                    reasoning = "Balance sheet equation verified: A = L + E"
                    confidence = 0.95
                else:
                    response_text = f"The balance sheet is incorrect: Assets ({verification['assets']}) ≠ Liabilities ({verification['liabilities']}) + Equity ({verification['equity']}). Expected equity: {verification['expected_equity']}."
                    reasoning = f"Balance sheet violation: {verification['assets']} ≠ {verification['liabilities']} + {verification['equity']}"
                    confidence = 0.90
            else:
                response_text = "I need a complete balance sheet with assets, liabilities, and equity to verify."
                reasoning = "Incomplete balance sheet information"
                confidence = 0.5
        else:
            # General accounting response
            numbers = self.extract_numbers(prompt, "accounting")
            if numbers:
                total = sum(numbers)
                response_text = f"The total of the provided numbers is {total}."
                reasoning = f"Summed extracted numbers: {numbers}"
                confidence = 0.85
            else:
                response_text = "Please provide specific numeric values for accounting calculations."
                reasoning = "No numeric values found in prompt"
                confidence = 0.4
        
        return LLMResponse(
            text=response_text,
            confidence=confidence,
            reasoning=reasoning,
            extracted_numbers=self.extract_numbers(prompt, "accounting")
        )
    
    def _generate_general_response(self, prompt: str) -> LLMResponse:
        """Generate general response."""
        # Extract any numbers for context
        numbers = self.extract_numbers(prompt, "general")
        
        if "poem" in prompt.lower() or "story" in prompt.lower():
            response_text = "I can help with calculations and accounting tasks, but I'm not designed for creative writing."
            reasoning = "Out of domain request detected"
            confidence = 0.2
        elif "balance" in prompt.lower() and not numbers:
            response_text = "To provide balance information, I need specific numeric values or a complete balance sheet."
            reasoning = "Ambiguous balance request without numbers"
            confidence = 0.3
        else:
            response_text = "I can help with arithmetic calculations and accounting tasks. Please provide specific numeric values."
            reasoning = "General assistance response"
            confidence = 0.6
        
        return LLMResponse(
            text=response_text,
            confidence=confidence,
            reasoning=reasoning,
            extracted_numbers=numbers
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model_name,
            "call_count": self.call_count,
            "domains_supported": list(self.response_patterns.keys()),
            "capabilities": [
                "arithmetic_calculation",
                "number_extraction", 
                "balance_sheet_verification",
                "domain_awareness"
            ],
            "confidence_ranges": {
                "arithmetic": (0.3, 0.95),
                "accounting": (0.4, 0.95),
                "general": (0.2, 0.6)
            }
        }


class EfficientLLMWrapper:
    """
    Wrapper for efficient LLM that integrates with Φ-Integrity.
    """
    
    def __init__(self):
        """Initialize efficient LLM."""
        self.llm = EfficientPhiLLM()
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using efficient LLM.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (ignored for determinism)
            
        Returns:
            Generated text response
        """
        # Determine domain from kwargs or infer
        domain = kwargs.get('domain', 'general')
        if domain == 'accounting' or any(term in prompt.lower() for term in ['calculate', 'balance', 'assets', 'liabilities', 'equity']):
            domain = 'accounting'
        
        # Generate response
        response = self.llm.generate_response(prompt, domain)
        
        # Return just the text (for compatibility with existing wrapper)
        return response.text
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return self.llm.get_model_info()


# Global efficient LLM instance
_EFFICIENT_LLM = EfficientLLMWrapper()


def get_efficient_llm() -> EfficientLLMWrapper:
    """Get global efficient LLM instance."""
    return _EFFICIENT_LLM
