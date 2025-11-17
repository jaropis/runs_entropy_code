#!/usr/bin/env python3
"""
Quick test script to validate the enhanced MI analysis functionality.
This creates synthetic data and tests all the new statistical methods.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Test if imports work
try:
    from MI import (
        mutual_information_run_category_vs_stage_df,
        bootstrap_mi_ci,
        permutation_test_mi,
        compute_summary_statistics
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)

def create_synthetic_data(n_windows=100, seed=42):
    """Create synthetic data for testing."""
    np.random.seed(seed)
    
    # Create random run counts
    data = {
        'AR1': np.random.poisson(5, n_windows),
        'AR2': np.random.poisson(3, n_windows),
        'AR3': np.random.poisson(2, n_windows),
        'DR1': np.random.poisson(4, n_windows),
        'DR2': np.random.poisson(2, n_windows),
        'DR3': np.random.poisson(1, n_windows),
    }
    
    # Create stage columns (one-hot encoded)
    # Make stages correlated with runs for realistic MI
    stage_probs = np.random.dirichlet([1, 1, 1], n_windows)
    stage_idx = np.array([np.random.choice(3, p=p) for p in stage_probs])
    
    data['Stage_W'] = (stage_idx == 0).astype(int)
    data['Stage_N2'] = (stage_idx == 1).astype(int)
    data['Stage_R'] = (stage_idx == 2).astype(int)
    
    return pd.DataFrame(data)

def test_basic_mi():
    """Test basic MI computation with metadata."""
    print("\n" + "="*60)
    print("TEST 1: Basic MI computation with metadata")
    print("="*60)
    
    df = create_synthetic_data(n_windows=50)
    
    try:
        mi, details, metadata = mutual_information_run_category_vs_stage_df(
            df,
            stage_prefixes=("Stage_",),
            drop_ambiguous_rows=True,
            eps=0.0
        )
        
        print(f"✓ MI (nats): {mi:.6f}")
        print(f"✓ MI (bits): {mi / np.log(2):.6f}")
        print(f"✓ Total beats: {metadata['total_beats']}")
        print(f"✓ Non-zero cells (K): {metadata['K']}")
        print(f"✓ H(runs) [nats]: {metadata['H_runs']:.6f}")
        print(f"✓ H(stages) [nats]: {metadata['H_stages']:.6f}")
        
        # Test Miller-Madow correction
        K = metadata['K']
        N = metadata['total_beats']
        mi_corrected_bits = mi / np.log(2) - (K - 1) / (2 * N * np.log(2))
        print(f"✓ MI corrected (bits): {mi_corrected_bits:.6f}")
        
        # Test normalized MI
        mi_norm = mi / min(metadata['H_runs'], metadata['H_stages'])
        print(f"✓ MI normalized: {mi_norm:.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_bootstrap():
    """Test bootstrap CI computation."""
    print("\n" + "="*60)
    print("TEST 2: Bootstrap confidence intervals (fast test: 100 iterations)")
    print("="*60)
    
    df = create_synthetic_data(n_windows=50)
    
    try:
        ci_lower, ci_upper = bootstrap_mi_ci(
            df,
            stage_prefixes=("Stage_",),
            n_bootstrap=100,  # Use fewer for testing
            confidence_level=0.95,
            random_seed=42
        )
        
        print(f"✓ CI lower (nats): {ci_lower:.6f}")
        print(f"✓ CI upper (nats): {ci_upper:.6f}")
        print(f"✓ CI lower (bits): {ci_lower / np.log(2):.6f}")
        print(f"✓ CI upper (bits): {ci_upper / np.log(2):.6f}")
        print(f"✓ CI width (bits): {(ci_upper - ci_lower) / np.log(2):.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_permutation():
    """Test permutation testing."""
    print("\n" + "="*60)
    print("TEST 3: Permutation test (fast test: 100 permutations)")
    print("="*60)
    
    df = create_synthetic_data(n_windows=50)
    
    try:
        p_value, null_dist = permutation_test_mi(
            df,
            stage_prefixes=("Stage_",),
            n_permutations=100,  # Use fewer for testing
            random_seed=42
        )
        
        print(f"✓ p-value: {p_value:.4f}")
        print(f"✓ Null distribution size: {len(null_dist)}")
        if len(null_dist) > 0:
            print(f"✓ Null MI mean (bits): {null_dist.mean() / np.log(2):.6f}")
            print(f"✓ Null MI std (bits): {null_dist.std() / np.log(2):.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_summary_statistics():
    """Test aggregate statistics computation."""
    print("\n" + "="*60)
    print("TEST 4: Summary statistics computation")
    print("="*60)
    
    # Create synthetic summary data
    np.random.seed(42)
    n_subjects = 10
    
    summary_data = {
        'file': [f'subject_{i}.csv' for i in range(n_subjects)],
        'mi_bits': np.random.uniform(0.1, 0.5, n_subjects),
        'mi_corrected_bits': np.random.uniform(0.08, 0.48, n_subjects),
        'mi_normalized': np.random.uniform(0.1, 0.8, n_subjects),
        'p_value': np.random.uniform(0.001, 0.1, n_subjects),
        'ci_lower_bits': np.random.uniform(0.05, 0.3, n_subjects),
        'ci_upper_bits': np.random.uniform(0.15, 0.6, n_subjects),
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    try:
        stats = compute_summary_statistics(summary_df)
        print("✓ Summary statistics computed:")
        print(stats.to_string(index=False))
        
        # Verify expected columns
        expected_metrics = ['MI_corrected (bits)', 'MI_normalized', 'p_value']
        for metric in expected_metrics:
            if metric in stats['metric'].values:
                print(f"✓ Metric '{metric}' present in output")
            else:
                print(f"✗ Metric '{metric}' missing from output")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("TESTING ENHANCED MI ANALYSIS FEATURES")
    print("="*60)
    
    tests = [
        ("Basic MI with metadata", test_basic_mi),
        ("Bootstrap CI", test_bootstrap),
        ("Permutation test", test_permutation),
        ("Summary statistics", test_summary_statistics),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    n_passed = sum(1 for _, s in results if s)
    n_total = len(results)
    print(f"\nPassed: {n_passed}/{n_total}")
    
    if n_passed == n_total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {n_total - n_passed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit(main())
