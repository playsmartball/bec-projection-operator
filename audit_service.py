"""
Φ-Integrity Audit Service (Fork A - Locked)

Automated verification service for Φ-Integrity pipeline.
Provides comprehensive auditing of projection integrity, constraint satisfaction,
and decision reproducibility.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from src.wrapper import process_with_integrity, get_integrity_statistics
from src.projection import get_locked_spec
from src.constraints import ConstraintStatus, RefusalReason


@dataclass
class AuditReport:
    """Comprehensive audit report."""
    audit_id: str
    timestamp: str
    test_cases: List[Dict[str, Any]]
    summary_statistics: Dict[str, Any]
    integrity_verification: Dict[str, Any]
    reproducibility_test: Dict[str, Any]
    specification_compliance: Dict[str, Any]
    overall_status: str
    recommendations: List[str]


class PhiIntegrityAuditor:
    """
    Automated auditor for Φ-Integrity system.
    
    Verifies:
    - Projection integrity (fixed parameters, deterministic output)
    - Constraint satisfaction (proper refusal/allowance decisions)
    - Specification compliance (locked parameters not modified)
    - Reproducibility (same input = same output)
    """
    
    def __init__(self, log_dir: str = "audit_logs"):
        """Initialize auditor."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Test suite (locked)
        self.test_suite = [
            {
                "name": "Correct Arithmetic - Should Allow",
                "prompt": "Calculate 2+2",
                "domain": "accounting",
                "reference_data": {"expected_sum": 4.0},
                "expected_status": "ALLOW",
                "category": "basic_arithmetic"
            },
            {
                "name": "Incorrect Arithmetic Reference - Should Refuse",
                "prompt": "Calculate 2+2",
                "domain": "accounting", 
                "reference_data": {"expected_sum": 5.0},
                "expected_status": "REFUSE",
                "category": "basic_arithmetic"
            },
            {
                "name": "Balance Sheet Correct - Should Allow",
                "prompt": "Verify balance sheet: Assets: 1000, Liabilities: 600, Equity: 400",
                "domain": "accounting",
                "reference_data": {"conserved_total": 1000.0},
                "expected_status": "ALLOW",
                "category": "balance_sheet"
            },
            {
                "name": "Balance Sheet Incorrect - Should Refuse",
                "prompt": "Verify balance sheet: Assets: 1000, Liabilities: 600, Equity: 500",
                "domain": "accounting",
                "reference_data": {"conserved_total": 1000.0},
                "expected_status": "REFUSE", 
                "category": "balance_sheet"
            },
            {
                "name": "Out of Domain - Should Refuse",
                "prompt": "Write a poem about flowers",
                "domain": "accounting",
                "reference_data": None,
                "expected_status": "REFUSE",
                "category": "domain_validation"
            },
            {
                "name": "Ambiguous Prompt - Should Refuse",
                "prompt": "Tell me about the balance",
                "domain": "accounting",
                "reference_data": None,
                "expected_status": "REFUSE",
                "category": "domain_validation"
            }
        ]
    
    def run_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single test case and collect detailed metrics.
        
        Args:
            test_case: Test case definition
            
        Returns:
            Detailed test result
        """
        start_time = time.time()
        
        # Process through Φ-Integrity
        response = process_with_integrity(
            prompt=test_case["prompt"],
            domain=test_case["domain"],
            reference_data=test_case["reference_data"]
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        # Evaluate result
        actual_status = response["status"]
        expected_status = test_case["expected_status"]
        test_passed = actual_status == expected_status
        
        # Collect detailed metrics
        result = {
            "test_name": test_case["name"],
            "category": test_case["category"],
            "prompt": test_case["prompt"],
            "domain": test_case["domain"],
            "reference_data": test_case["reference_data"],
            "expected_status": expected_status,
            "actual_status": actual_status,
            "test_passed": test_passed,
            "execution_time_ms": execution_time,
            "response": {
                "status": response["status"],
                "output": response["output"],
                "reason": response["reason"],
                "run_id": response["trace"]["run_id"],
                "phi_hash": response["trace"]["phi_hash"],
                "metrics": response["trace"]["metrics"],
                "constraints_checked": response["trace"]["constraints_checked"]
            }
        }
        
        # Add violation details if refused
        if actual_status == "REFUSE" and "violations" in response["trace"]:
            result["response"]["violations"] = response["trace"]["violations"]
        
        return result
    
    def verify_specification_compliance(self) -> Dict[str, Any]:
        """
        Verify that locked specifications are still in place.
        
        Returns:
            Specification compliance report
        """
        # Get current specifications
        current_spec = get_locked_spec()
        
        # Expected locked values
        expected_spec = {
            "resolution": 5000,
            "phi_range": (0.0, 10.0),
            "sigma": 1.0
        }
        
        compliance = {}
        violations = []
        
        for key, expected_value in expected_spec.items():
            actual_value = current_spec[key]
            is_compliant = actual_value == expected_value
            
            compliance[key] = {
                "expected": expected_value,
                "actual": actual_value,
                "compliant": is_compliant
            }
            
            if not is_compliant:
                violations.append(f"{key}: expected {expected_value}, got {actual_value}")
        
        return {
            "overall_compliant": len(violations) == 0,
            "violations": violations,
            "detailed_compliance": compliance,
            "specification_hash": hashlib.sha256(str(current_spec).encode()).hexdigest()[:16]
        }
    
    def test_reproducibility(self, num_runs: int = 3) -> Dict[str, Any]:
        """
        Test reproducibility of the system.
        
        Args:
            num_runs: Number of runs to test
            
        Returns:
            Reproducibility test results
        """
        # Test case for reproducibility
        test_prompt = "Calculate 2+2"
        test_domain = "accounting"
        test_reference = {"expected_sum": 4.0}
        
        results = []
        phi_hashes = []
        run_ids = []
        
        for i in range(num_runs):
            response = process_with_integrity(test_prompt, test_domain, test_reference)
            
            results.append(response)
            phi_hashes.append(response["trace"]["phi_hash"])
            run_ids.append(response["trace"]["run_id"])
        
        # Check reproducibility
        unique_phi_hashes = set(phi_hashes)
        unique_run_ids = set(run_ids)
        
        # All phi hashes should be identical (deterministic projection)
        phi_reproducible = len(unique_phi_hashes) == 1
        
        # All run IDs should be unique (time-based uniqueness)
        run_id_unique = len(unique_run_ids) == num_runs
        
        # Check response consistency
        statuses = [r["status"] for r in results]
        outputs = [r["output"] for r in results]
        reasons = [r["reason"] for r in results]
        
        status_consistent = len(set(statuses)) == 1
        output_consistent = len(set(outputs)) == 1
        reason_consistent = len(set(reasons)) <= 1  # None for ALLOW is fine
        
        return {
            "test_prompt": test_prompt,
            "num_runs": num_runs,
            "phi_reproducible": phi_reproducible,
            "run_id_unique": run_id_unique,
            "status_consistent": status_consistent,
            "output_consistent": output_consistent,
            "reason_consistent": reason_consistent,
            "overall_reproducible": phi_reproducible and run_id_unique and status_consistent,
            "phi_hash": phi_hashes[0] if phi_hashes else None,
            "unique_phi_hashes": len(unique_phi_hashes),
            "unique_run_ids": len(unique_run_ids)
        }
    
    def run_full_audit(self) -> AuditReport:
        """
        Run comprehensive audit of Φ-Integrity system.
        
        Returns:
            Complete audit report
        """
        audit_id = hashlib.sha256(f"{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        timestamp = datetime.now().isoformat()
        
        print(f"Starting Φ-Integrity Audit {audit_id}...")
        
        # Run all test cases
        test_results = []
        for test_case in self.test_suite:
            print(f"Running: {test_case['name']}")
            result = self.run_test_case(test_case)
            test_results.append(result)
        
        # Get system statistics
        system_stats = get_integrity_statistics()
        
        # Verify specification compliance
        spec_compliance = self.verify_specification_compliance()
        
        # Test reproducibility
        reproducibility = self.test_reproducibility()
        
        # Calculate summary statistics
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r["test_passed"])
        failed_tests = total_tests - passed_tests
        
        # Category breakdown
        categories = {}
        for result in test_results:
            category = result["category"]
            if category not in categories:
                categories[category] = {"total": 0, "passed": 0}
            categories[category]["total"] += 1
            if result["test_passed"]:
                categories[category]["passed"] += 1
        
        # Performance metrics
        execution_times = [r["execution_time_ms"] for r in test_results]
        avg_execution_time = sum(execution_times) / len(execution_times)
        max_execution_time = max(execution_times)
        min_execution_time = min(execution_times)
        
        # Generate recommendations
        recommendations = []
        
        if failed_tests > 0:
            recommendations.append(f"{failed_tests} test(s) failed - review constraint implementation")
        
        if not spec_compliance["overall_compliant"]:
            recommendations.append("Specification violations detected - locked parameters may have been modified")
        
        if not reproducibility["overall_reproducible"]:
            recommendations.append("Reproducibility issues detected - deterministic execution may be compromised")
        
        if avg_execution_time > 100:
            recommendations.append("High average execution time - consider performance optimization")
        
        if system_stats["allowance_rate"] > 0.8:
            recommendations.append("High allowance rate - constraints may be too permissive")
        elif system_stats["allowance_rate"] < 0.2:
            recommendations.append("Low allowance rate - constraints may be too strict")
        
        # Determine overall status
        if (failed_tests == 0 and 
            spec_compliance["overall_compliant"] and 
            reproducibility["overall_reproducible"]):
            overall_status = "PASS"
        elif failed_tests > 0 or not spec_compliance["overall_compliant"]:
            overall_status = "FAIL"
        else:
            overall_status = "WARNING"
        
        # Compile report
        report = AuditReport(
            audit_id=audit_id,
            timestamp=timestamp,
            test_cases=test_results,
            summary_statistics={
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": passed_tests / total_tests,
                "categories": categories,
                "performance": {
                    "avg_execution_time_ms": avg_execution_time,
                    "max_execution_time_ms": max_execution_time,
                    "min_execution_time_ms": min_execution_time
                },
                "system_stats": system_stats
            },
            integrity_verification=spec_compliance,
            reproducibility_test=reproducibility,
            specification_compliance=spec_compliance,
            overall_status=overall_status,
            recommendations=recommendations
        )
        
        # Save report
        report_file = self.log_dir / f"audit_report_{audit_id}.json"
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        print(f"Audit complete. Report saved to {report_file}")
        
        return report
    
    def print_summary_report(self, report: AuditReport):
        """Print concise summary of audit results."""
        print(f"\n{'='*60}")
        print(f"Φ-INTEGRITY AUDIT REPORT {report.audit_id}")
        print(f"{'='*60}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Overall Status: {report.overall_status}")
        
        print(f"\n📊 SUMMARY STATISTICS")
        print(f"Total Tests: {report.summary_statistics['total_tests']}")
        print(f"Passed: {report.summary_statistics['passed_tests']}")
        print(f"Failed: {report.summary_statistics['failed_tests']}")
        print(f"Pass Rate: {report.summary_statistics['pass_rate']:.1%}")
        
        print(f"\n🔒 INTEGRITY VERIFICATION")
        print(f"Specification Compliant: {report.integrity_verification['overall_compliant']}")
        if not report.integrity_verification['overall_compliant']:
            for violation in report.integrity_verification['violations']:
                print(f"  ❌ {violation}")
        
        print(f"\n🔄 REPRODUCIBILITY TEST")
        print(f"Overall Reproducible: {report.reproducibility_test['overall_reproducible']}")
        print(f"Φ Hash Consistent: {report.reproducibility_test['phi_reproducible']}")
        print(f"Run ID Unique: {report.reproducibility_test['run_id_unique']}")
        
        print(f"\n📈 PERFORMANCE")
        perf = report.summary_statistics['performance']
        print(f"Avg Execution Time: {perf['avg_execution_time_ms']:.2f}ms")
        print(f"Max Execution Time: {perf['max_execution_time_ms']:.2f}ms")
        
        print(f"\n💡 RECOMMENDATIONS")
        if report.recommendations:
            for rec in report.recommendations:
                print(f"  • {rec}")
        else:
            print("  ✅ No recommendations - system operating optimally")
        
        print(f"\n{'='*60}")


def main():
    """Run audit service from command line."""
    auditor = PhiIntegrityAuditor()
    
    # Run full audit
    report = auditor.run_full_audit()
    
    # Print summary
    auditor.print_summary_report(report)
    
    # Return exit code based on overall status
    exit_code = 0 if report.overall_status == "PASS" else 1
    exit(exit_code)


if __name__ == "__main__":
    main()
