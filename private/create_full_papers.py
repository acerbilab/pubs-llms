#!/usr/bin/env python3
# AI Summary: Creates full versions of papers by concatenating main, backmatter, and appendix files.
# Processes all papers in publications directory or specific files when provided.
# Preserves document structure with proper section separators.

import os
import glob
import argparse
import sys
from pathlib import Path

def create_full_paper(main_file_path, output_dir=None, overwrite=False):
    """
    Create a full paper file by concatenating main, backmatter, and appendix files.
    
    Args:
        main_file_path (str): Path to the main file
        output_dir (str, optional): Directory to save the full file. Defaults to same directory as main file.
        overwrite (bool, optional): Whether to overwrite existing full files. Defaults to False.
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Extract base name by removing "_main" suffix
    base_name = Path(main_file_path).stem.replace("_main", "")
    main_dir = os.path.dirname(main_file_path)
    
    # Determine output directory
    if output_dir is None:
        output_dir = main_dir
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths
    backmatter_file = os.path.join(main_dir, f"{base_name}_backmatter.md")
    appendix_file = os.path.join(main_dir, f"{base_name}_appendix.md")
    full_file = os.path.join(output_dir, f"{base_name}_full.md")
    
    # Check if full file already exists
    if os.path.exists(full_file):
        if overwrite:
            print(f"Full file already exists: {full_file} (will overwrite)")
        else:
            print(f"Skipping: Full file already exists: {full_file}")
            return True  # Return True because skipping is expected behavior, not an error
    
    try:
        # Read main content
        with open(main_file_path, 'r', encoding='utf-8') as f:
            main_content = f.read().strip()
        
        # Initialize full content with main content
        full_content = main_content
        
        # Add backmatter if exists
        if os.path.exists(backmatter_file):
            with open(backmatter_file, 'r', encoding='utf-8') as f:
                backmatter_content = f.read().strip()
            full_content += f"\n\n---\n\n{backmatter_content}"
        
        # Add appendix if exists
        if os.path.exists(appendix_file):
            with open(appendix_file, 'r', encoding='utf-8') as f:
                appendix_content = f.read().strip()
            full_content += f"\n\n---\n\n{appendix_content}"
        
        # Write the full file
        with open(full_file, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        print(f"Created full file: {full_file}")
        return True
    
    except Exception as e:
        print(f"Error creating full file for {main_file_path}: {e}")
        return False

def process_all_papers(publications_dir, output_dir=None, overwrite=False):
    """
    Process all papers in the publications directory to create full versions.
    
    Args:
        publications_dir (str): Path to the publications directory
        output_dir (str, optional): Directory to save full files. Defaults to same as source.
        overwrite (bool, optional): Whether to overwrite existing full files. Defaults to False.
        
    Returns:
        tuple: (success_count, total_count)
    """
    # Find all main files
    main_files = glob.glob(os.path.join(publications_dir, '*_main.md'))
    
    # Process each file
    success_count = 0
    for main_file in main_files:
        if create_full_paper(main_file, output_dir, overwrite):
            success_count += 1
    
    return success_count, len(main_files)

def main():
    parser = argparse.ArgumentParser(
        description='Create full paper files by concatenating main, backmatter, and appendix files.')
    parser.add_argument('--input', '-i', help='Specific main file(s) to process', nargs='*')
    parser.add_argument('--output-dir', '-o', help='Directory to save full files (defaults to same as source)')
    parser.add_argument('--publications-dir', '-p', 
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'publications'),
                        help='Path to the publications directory')
    parser.add_argument('--overwrite', '-w', action='store_true', help='Overwrite existing full files')
    
    args = parser.parse_args()
    
    # Process specific files if provided
    if args.input:
        success_count = 0
        for input_file in args.input:
            if not os.path.exists(input_file):
                print(f"Error: Input file '{input_file}' not found.")
                continue
            
            if create_full_paper(input_file, args.output_dir, args.overwrite):
                success_count += 1
                
        print(f"Successfully processed {success_count} out of {len(args.input)} files.")
        return 0 if success_count == len(args.input) else 1
    
    # Process all papers
    else:
        if not os.path.isdir(args.publications_dir):
            print(f"Error: Publications directory '{args.publications_dir}' not found.")
            return 1
        
        success_count, total_count = process_all_papers(
            args.publications_dir, args.output_dir, args.overwrite)
        
        print(f"Successfully processed {success_count} out of {total_count} files.")
        return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main())
