#!/usr/bin/env python3
"""check_noise_presence.py - Verify DP noise injection patterns in Python files.

This script checks for the presence of differential privacy related patterns
in Python source files, including:
- Noise injection (torch.randn, np.random.normal)
- Gradient clipping operations
- DP parameter variables (dp_sigma, clipping_norm)

Usage:
    python tools/check_noise_presence.py <file.py>
    python tools/check_noise_presence.py exp/exp_main.py
    
Batch usage:
    for f in exp/*.py data_provider/*.py; do python tools/check_noise_presence.py "$f"; done
"""

import re
import sys
import os
from pathlib import Path


def check_file(path: str) -> dict:
    """Check a Python file for DP-related patterns.
    
    Parameters:
        path: Path to the Python file to check.
        
    Returns:
        Dictionary mapping pattern names to boolean (found/not found).
    """
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()
    
    patterns = {
        'torch.randn': r"torch\.randn",
        'np.random.normal': r"np\.random\.normal",
        'dp_sigma': r"\bdp_sigma\b",
        'clipping_norm': r"clipping_norm|clip_norm",
        'noise_injection': r"\bnoise\s*=\s*torch\.randn",
        'gradient_clipping': r"clip_grad|grad_norm|clip_factor",
        'dp_comment': r"#\s*DP:",
        'module_docstring': r'^"""[^"]*DP notes:',
    }
    
    results = {}
    for name, pattern in patterns.items():
        if name == 'module_docstring':
            # Check only at the beginning of file
            results[name] = bool(re.search(pattern, txt[:2000], re.MULTILINE))
        else:
            results[name] = bool(re.search(pattern, txt, re.IGNORECASE if name != 'dp_comment' else 0))
    
    return results


def analyze_noise_formula(path: str) -> dict:
    """Analyze the noise injection formula in a file.
    
    Parameters:
        path: Path to the Python file to analyze.
        
    Returns:
        Dictionary with noise formula analysis.
    """
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()
    
    analysis = {
        'has_noise_injection': False,
        'noise_formula': None,
        'sigma_multiplier': None,
        'compliant': None,
    }
    
    # Look for noise injection patterns
    # Pattern: noise = torch.randn_like(...) * dp_sigma * clipping_norm [* 2]
    noise_pattern = r"noise\s*=\s*torch\.randn[_a-z]*\([^)]+\)\s*\*\s*([^#\n]+)"
    match = re.search(noise_pattern, txt)
    
    if match:
        analysis['has_noise_injection'] = True
        formula = match.group(1).strip()
        analysis['noise_formula'] = formula
        
        # Check if it's sigma = C or sigma = 2*C
        if '* 2' in formula or '*2' in formula:
            analysis['sigma_multiplier'] = '2*C (spaced sampling)'
            analysis['compliant'] = True
        elif 'dp_sigma' in formula and 'clipping_norm' in formula:
            analysis['sigma_multiplier'] = 'C (standard DP-SGD)'
            analysis['compliant'] = True
        else:
            analysis['sigma_multiplier'] = 'unknown'
            analysis['compliant'] = False
    
    return analysis


def print_results(path: str, results: dict, analysis: dict):
    """Print formatted results for a file.
    
    Parameters:
        path: Path to the checked file.
        results: Pattern matching results.
        analysis: Noise formula analysis.
    """
    filename = os.path.basename(path)
    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"{'='*60}")
    
    print("\nPattern Matches:")
    for pattern, found in results.items():
        status = "✓ found" if found else "✗ not found"
        print(f"  {pattern:20s}: {status}")
    
    print("\nNoise Formula Analysis:")
    if analysis['has_noise_injection']:
        print(f"  Formula: {analysis['noise_formula']}")
        print(f"  Sigma multiplier: {analysis['sigma_multiplier']}")
        print(f"  Compliant: {'✓ Yes' if analysis['compliant'] else '✗ No'}")
    else:
        print("  No noise injection found in this file")
    
    # Summary
    has_dp_code = results['dp_sigma'] or results['noise_injection'] or results['gradient_clipping']
    has_comments = results['dp_comment'] or results['module_docstring']
    
    print("\nSummary:")
    if has_dp_code:
        print(f"  DP code present: ✓ Yes")
        print(f"  DP comments: {'✓ Yes' if has_comments else '✗ Missing'}")
    else:
        print(f"  DP code present: ✗ No (may be data loading only)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_noise_presence.py <file.py> [file2.py ...]")
        print("\nExample:")
        print("  python tools/check_noise_presence.py exp/exp_main.py")
        print("  python tools/check_noise_presence.py exp/*.py")
        sys.exit(1)
    
    files = sys.argv[1:]
    
    summary = []
    
    for filepath in files:
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue
            
        if not filepath.endswith('.py'):
            print(f"Warning: Skipping non-Python file: {filepath}")
            continue
        
        results = check_file(filepath)
        analysis = analyze_noise_formula(filepath)
        print_results(filepath, results, analysis)
        
        # Collect for summary
        summary.append({
            'file': filepath,
            'has_dp': results['dp_sigma'] or results['noise_injection'],
            'has_comments': results['dp_comment'] or results['module_docstring'],
            'compliant': analysis['compliant'],
        })
    
    # Print overall summary
    if len(summary) > 1:
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        
        dp_files = [s for s in summary if s['has_dp']]
        commented = [s for s in summary if s['has_comments']]
        compliant = [s for s in summary if s['compliant'] is True]
        
        print(f"Total files checked: {len(summary)}")
        print(f"Files with DP code: {len(dp_files)}")
        print(f"Files with DP comments: {len(commented)}")
        print(f"Compliant noise injection: {len(compliant)}")


if __name__ == '__main__':
    main()
