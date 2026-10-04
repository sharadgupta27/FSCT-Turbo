"""
Batch Processing Utility for FSCT Wrapper
Process multiple point clouds with the same parameters
"""

import os
import sys
import json
import glob
import pandas as pd
from datetime import datetime

# The FSCT modules in scripts/ import each other by bare name, so scripts/
# must be importable in its own right; the project root is added so this works
# from any working directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
for _path in (os.path.join(PROJECT_ROOT, 'scripts'), PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from version import __version__
from scripts.run_tools import FSCT
from fsct_job import job_parameters


def load_config(config_file='wrapper_config.json'):
    """
    Load configuration from a JSON file, falling back to the built-in defaults.

    A malformed or unreadable config used to abort the whole batch with a raw
    JSONDecodeError traceback before a single file was processed. Report it and
    carry on instead - fsct_job.DEFAULT_PARAMETERS is enough to run.

    A relative path that does not exist from the current directory is looked
    up next to this script, so `python path/to/batch_process.py` finds the
    shipped wrapper_config.json from anywhere.
    """
    if not os.path.isabs(config_file) and not os.path.exists(config_file):
        beside_script = os.path.join(PROJECT_ROOT, config_file)
        if os.path.exists(beside_script):
            config_file = beside_script
    if not os.path.exists(config_file):
        print(f"Note: {config_file} not found; using the default parameters.")
        return {}
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except (ValueError, OSError) as error:
        print(f"Warning: could not read {config_file} ({error}).")
        print("Continuing with default parameters.")
        return {}
    if not isinstance(config, dict):
        print(f"Warning: {config_file} does not contain a JSON object.")
        print("Continuing with default parameters.")
        return {}
    return config


def get_las_files(directory, recursive=True):
    """Find all LAS/LAZ files in a directory"""
    # glob is case-sensitive on the pattern, so '*.las' skipped PLOT.LAS
    # entirely - the file simply never appeared in the batch. Match on the
    # extension after the fact instead, which is case-insensitive and also
    # avoids listing the tree twice.
    pattern = '**/*' if recursive else '*'
    candidates = glob.glob(os.path.join(directory, pattern), recursive=recursive)

    filtered_files = [
        f for f in candidates
        if os.path.splitext(f)[1].lower() in ('.las', '.laz')
        and 'FSCT_output' not in f
    ]

    return sorted(filtered_files)


def batch_process_fsct(input_directory, parameters=None, recursive=True,
                       config_file='wrapper_config.json'):
    """
    Batch process all point clouds in a directory

    Args:
        input_directory: Directory containing LAS/LAZ files
        parameters: Dictionary of FSCT parameters (uses defaults if None)
        recursive: Whether to search subdirectories
        config_file: JSON file to read default_parameters from

    Returns:
        Dictionary with processing results
    """

    # Start from the shared defaults and lay the config on top. Taking the
    # config's dict as the whole parameter set made every key it lacked a
    # KeyError inside FSCT - the shipped wrapper_config.json had no
    # plot_centre, so every file failed before a point was read.
    if parameters is None:
        parameters = load_config(config_file).get('default_parameters', {})
    parameters = job_parameters(**parameters)

    # Find all point cloud files
    point_clouds = get_las_files(input_directory, recursive)

    # Every early return has to carry the same keys as the normal one -
    # main() reads results['processed'] unconditionally.
    if not point_clouds:
        print(f"No LAS/LAZ files found in {input_directory}")
        return {'total_files': 0, 'processed': 0, 'failed': 0, 'files': [],
                'message': 'No files found'}

    print(f"Found {len(point_clouds)} point cloud files to process")
    print("-" * 50)

    # Process each file
    results = {
        'total_files': len(point_clouds),
        'processed': 0,
        'failed': 0,
        'files': []
    }
    
    start_time = datetime.now()
    
    for idx, pc_file in enumerate(point_clouds, 1):
        print(f"\n[{idx}/{len(point_clouds)}] Processing: {os.path.basename(pc_file)}")
        print("-" * 50)
        
        file_result = {
            'filename': pc_file,
            'success': False,
            'output_dir': None,
            'error': None
        }
        
        try:
            # Update parameters with current file
            params = parameters.copy()
            params['point_cloud_filename'] = pc_file
            
            # Run FSCT
            FSCT(
                parameters=params,
                preprocess=True,
                segmentation=True,
                postprocessing=True,
                measure_plot=True,
                make_report=True,
                clean_up_files=False
            )
            
            # Get output directory
            output_dir = os.path.splitext(pc_file.replace('\\', '/'))[0] + "_FSCT_output/"
            
            file_result['success'] = True
            file_result['output_dir'] = output_dir
            results['processed'] += 1
            
            print(f"✓ Successfully processed: {os.path.basename(pc_file)}")
            
        except Exception as e:
            file_result['error'] = str(e)
            results['failed'] += 1
            print(f"✗ Failed to process: {os.path.basename(pc_file)}")
            print(f"  Error: {str(e)}")
        
        results['files'].append(file_result)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Print summary
    print("\n" + "=" * 50)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total files: {results['total_files']}")
    print(f"Successfully processed: {results['processed']}")
    print(f"Failed: {results['failed']}")
    print(f"Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    if results['processed'] > 0:
        print(f"Average time per file: {duration/results['processed']:.1f} seconds")
    print("=" * 50)
    
    # Save results to CSV
    results_df = pd.DataFrame(results['files'])
    results_file = os.path.join(input_directory, 
                               f"batch_processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    return results


def combine_plot_summaries(input_directory, output_file=None):
    """
    Combine all plot_summary.csv files into a single CSV
    
    Args:
        input_directory: Root directory to search for FSCT outputs
        output_file: Path to save combined results (auto-generated if None)
    """
    
    # Find all plot_summary.csv files
    summary_files = glob.glob(
        os.path.join(input_directory, '**/*FSCT_output/plot_summary.csv'),
        recursive=True
    )
    
    if not summary_files:
        print(f"No plot_summary.csv files found in {input_directory}")
        return None
    
    print(f"Found {len(summary_files)} plot summary files")
    
    # Combine all summaries
    all_summaries = []
    
    for summary_file in summary_files:
        try:
            df = pd.read_csv(summary_file)
            # Add source file column
            df['Source_File'] = os.path.basename(os.path.dirname(summary_file)).replace('_FSCT_output', '')
            all_summaries.append(df)
        except Exception as e:
            print(f"Error reading {summary_file}: {str(e)}")
    
    if not all_summaries:
        print("No valid summary files could be read")
        return None
    
    # Combine into single dataframe
    combined_df = pd.concat(all_summaries, ignore_index=True)
    
    # Save to file
    if output_file is None:
        output_file = os.path.join(
            input_directory,
            f"combined_plot_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    
    combined_df.to_csv(output_file, index=False)
    print(f"Combined summaries saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("COMBINED SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Total plots processed: {len(combined_df)}")
    
    numeric_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        print("\nMean values across all plots:")
        for col in numeric_cols:
            if col != 'Source_File':
                print(f"  {col}: {combined_df[col].mean():.2f}")
    
    return combined_df


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process point clouds with FSCT')
    parser.add_argument('--version', action='version', version=f'FSCT {__version__}')
    parser.add_argument('input_dir', help='Directory containing LAS/LAZ files')
    parser.add_argument('--config', default='wrapper_config.json', 
                       help='Configuration file (default: wrapper_config.json)')
    parser.add_argument('--no-recursive', action='store_true',
                       help='Do not search subdirectories')
    parser.add_argument('--combine-only', action='store_true',
                       help='Only combine existing results, do not process')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Directory not found: {args.input_dir}")
        return 1
    
    if args.combine_only:
        # Only combine existing results
        combine_plot_summaries(args.input_dir)
    else:
        # Process files. --config was previously parsed and then ignored.
        try:
            results = batch_process_fsct(
                args.input_dir,
                recursive=not args.no_recursive,
                config_file=args.config
            )
        except KeyError as error:  # a misspelt parameter in the config
            print(f"Error in {args.config}: {error.args[0]}")
            return 2

        # Combine results
        if results['processed'] > 0:
            print("\nCombining plot summaries...")
            combine_plot_summaries(args.input_dir)

        # Non-zero if anything failed, so a script or scheduler running the
        # batch can tell. It used to exit 0 even when every file had failed.
        if results['failed'] > 0:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
