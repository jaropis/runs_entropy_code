import pandas as pd
import os
from pathlib import Path
import re

def process_run_statistics(folder_path, output_csv='run_statistics.csv'):
    """
    calculating run length statistics across all xlsx files in a folder
    
    parameters:
    folder_path: path to folder containing xlsx files
    output_csv: name of output CSV file
    """
    
    # initializing dictionary to store counts for each run type
    run_counts = {}
    
    # finding all xlsx files in the folder
    xlsx_files = list(Path(folder_path).glob('*.xlsx'))
    
    if not xlsx_files:
        print(f"No .xlsx files found in {folder_path}")
        return
    
    print(f"Found {len(xlsx_files)} xlsx files")
    
    # processing each file
    for file_path in xlsx_files:
        print(f"Processing: {file_path.name}")
        
        try:
            # reading the Excel file (all sheets)
            excel_file = pd.ExcelFile(file_path)
            
            # processing each sheet
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # finding DR and AR columns (excluding DR_MAX and AR_MAX)
                dr_cols = [col for col in df.columns 
                          if isinstance(col, str) and col.startswith('DR') 
                          and not col.endswith('_MAX') and re.match(r'DR\d+$', col)]
                ar_cols = [col for col in df.columns 
                          if isinstance(col, str) and col.startswith('AR') 
                          and not col.endswith('_MAX') and re.match(r'AR\d+$', col)]
                
                all_run_cols = dr_cols + ar_cols
                
                # summing up counts for each run type
                for col in all_run_cols:
                    # summing all non-null values in this column
                    total = df[col].fillna(0).sum()
                    
                    # adding to overall count
                    if col in run_counts:
                        run_counts[col] += total
                    else:
                        run_counts[col] = total
                        
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
            continue
    
    # converting to dataframe
    results_df = pd.DataFrame([
        {'run_type': run_type, 'total_count': int(count)}
        for run_type, count in sorted(run_counts.items())
    ])
    
    # sorting by run type (DR first, then AR, both numerically)
    def sort_key(run_type):
        if run_type.startswith('DR'):
            return (0, int(run_type[2:]))
        else:  # AR
            return (1, int(run_type[2:]))
    
    results_df['sort_key'] = results_df['run_type'].apply(sort_key)
    results_df = results_df.sort_values('sort_key').drop('sort_key', axis=1)
    
    # saving to CSV
    results_df.to_csv(output_csv, index=False)
    
    print(f"\nResults saved to {output_csv}")
    print(f"\nSummary:")
    print(f"Total run types found: {len(results_df)}")
    print(f"Total runs counted: {int(results_df['total_count'].sum())}")
    print(f"\nFirst few rows:")
    print(results_df.head(10))
    
    return results_df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        # defaulting to /mnt/user-data/uploads if no argument provided
        folder_path = '/mnt/user-data/uploads'
    
    print(f"Analyzing files in: {folder_path}\n")
    process_run_statistics(folder_path)