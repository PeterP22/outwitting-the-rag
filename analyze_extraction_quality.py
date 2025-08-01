#!/usr/bin/env python3
"""
PDF Extraction Quality Analysis
================================
This script performs a detailed analysis of the extracted text files
to identify quality issues and help choose the best extraction method.
"""

import os
import re
from collections import Counter
from pathlib import Path

def analyze_file(filepath):
    """
    Analyze a single extraction file for quality issues
    
    Returns a dictionary with quality metrics
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Initialize metrics
    metrics = {
        'filename': os.path.basename(filepath),
        'total_chars': len(content),
        'total_lines': len(lines),
        'empty_lines': sum(1 for line in lines if line.strip() == ''),
        'issues': []
    }
    
    # Check for page markers (our added separators)
    page_markers = [line for line in lines if '====' in line and 'PAGE' in line]
    metrics['page_markers'] = len(page_markers)
    
    # Check for actual page numbers (standalone numbers)
    standalone_numbers = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Look for lines that are just numbers (likely page numbers)
        if stripped.isdigit() and 1 <= int(stripped) <= 500:
            # Check context - if surrounded by empty lines, likely a page number
            prev_empty = i > 0 and lines[i-1].strip() == ''
            next_empty = i < len(lines)-1 and lines[i+1].strip() == ''
            if prev_empty or next_empty:
                standalone_numbers.append((i+1, stripped))
    
    if standalone_numbers:
        metrics['issues'].append(f"Found {len(standalone_numbers)} potential page numbers: {standalone_numbers[:5]}")
    
    # Check for headers/footers (repeated lines)
    # Skip our page markers and very short lines
    content_lines = [line.strip() for line in lines 
                    if line.strip() and '====' not in line and len(line.strip()) > 10]
    
    line_frequency = Counter(content_lines)
    repeated_lines = [(line, count) for line, count in line_frequency.items() 
                     if count >= 3 and len(line) < 100]
    
    if repeated_lines:
        metrics['issues'].append(f"Repeated lines (potential headers/footers): {repeated_lines[:3]}")
    
    # Check for broken words (hyphenation at line ends)
    hyphenated_lines = 0
    for line in lines:
        if line.strip().endswith('-') and len(line.strip()) > 1:
            # Check if next line starts with lowercase (likely a broken word)
            idx = lines.index(line)
            if idx < len(lines) - 1 and lines[idx + 1] and lines[idx + 1][0].islower():
                hyphenated_lines += 1
    
    metrics['hyphenated_lines'] = hyphenated_lines
    if hyphenated_lines > 5:
        metrics['issues'].append(f"Many hyphenated line breaks ({hyphenated_lines} found)")
    
    # Check for formatting artifacts
    artifacts = {
        'double_spaces': len(re.findall(r'  +', content)),
        'special_chars': len(re.findall(r'[™®©°•·]', content)),
        'broken_encoding': len(re.findall(r'[�]', content))
    }
    
    for artifact, count in artifacts.items():
        if count > 0:
            metrics['issues'].append(f"{artifact}: {count} occurrences")
    
    # Sample text quality (first real content after page markers)
    content_start = -1
    for i, line in enumerate(lines):
        if 'FEAR is the tool' in line:  # Known start of content
            content_start = i
            break
    
    if content_start >= 0:
        sample_lines = lines[content_start:content_start+10]
        metrics['sample_text'] = '\n'.join(sample_lines)
    
    # Calculate text density (chars per line average, excluding empty lines)
    non_empty_lines = [line for line in lines if line.strip()]
    if non_empty_lines:
        metrics['avg_chars_per_line'] = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
    
    return metrics

def compare_extractions():
    """Compare all extraction methods and recommend the best one"""
    
    print("PDF Extraction Quality Comparison")
    print("=" * 60)
    
    results_dir = "extraction_results"
    all_metrics = []
    
    # Analyze each file
    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith('.txt'):
            filepath = os.path.join(results_dir, filename)
            metrics = analyze_file(filepath)
            all_metrics.append(metrics)
            
            # Print analysis for this file
            print(f"\n{metrics['filename']}")
            print("-" * 40)
            print(f"Total characters: {metrics['total_chars']:,}")
            print(f"Total lines: {metrics['total_lines']:,}")
            print(f"Empty lines: {metrics['empty_lines']:,}")
            print(f"Hyphenated lines: {metrics['hyphenated_lines']}")
            print(f"Avg chars/line: {metrics.get('avg_chars_per_line', 0):.1f}")
            
            if metrics['issues']:
                print("\nIssues found:")
                for issue in metrics['issues']:
                    print(f"  • {issue}")
            else:
                print("\nNo major issues found!")
            
            if 'sample_text' in metrics:
                print("\nSample text:")
                print("---")
                print(metrics['sample_text'][:200] + "...")
    
    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    # Score each extraction method
    scores = []
    for m in all_metrics:
        score = 100
        score -= len(m['issues']) * 10  # Penalty for each issue type
        score -= m['hyphenated_lines'] * 0.5  # Small penalty for hyphenation
        score -= (m['empty_lines'] / m['total_lines']) * 20  # Penalty for too many empty lines
        
        # Bonus for good text density
        if 'avg_chars_per_line' in m and 40 <= m['avg_chars_per_line'] <= 80:
            score += 5
        
        scores.append((m['filename'], score, m))
    
    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nBest extraction method: {scores[0][0]} (score: {scores[0][1]:.1f})")
    print("\nRanking:")
    for filename, score, metrics in scores:
        extractor = filename.replace('_extraction.txt', '')
        print(f"  {extractor}: {score:.1f}/100")
    
    # Specific recommendations for cleanup
    print("\nCleanup recommendations for all methods:")
    print("  • Remove page separators (=== lines)")
    print("  • Remove standalone page numbers")
    print("  • Join hyphenated words at line breaks")
    print("  • Remove any repeated headers/footers")
    print("  • Normalize excessive whitespace")
    
    return scores[0][2]  # Return best metrics

if __name__ == "__main__":
    best_metrics = compare_extractions()
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Use the recommended extraction method for the full PDF")
    print("2. Implement the cleanup recommendations")
    print("3. Process the full book with the chosen method")