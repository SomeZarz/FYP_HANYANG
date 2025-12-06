import os
import shutil

BASE_DIR = 'data/raw'

for i in range(1, 301):
    subdir = os.path.join(BASE_DIR, f'vin{i}')
    csv_file = os.path.join(subdir, f'vin{i}.csv')
    
    if os.path.exists(csv_file):
        target_file = os.path.join(BASE_DIR, f'vin{i}.csv')
        print(f'Moving {csv_file} to {target_file}')
        shutil.move(csv_file, target_file)
        
        # Optionally remove the now-empty directory
        try:
            os.rmdir(subdir)
        except OSError:
            print(f'Could not remove directory {subdir}, it may not be empty.')
    else:
        print(f'File {csv_file} does not exist, skipping.')

print("Finished moving all CSV files to the data directory.")