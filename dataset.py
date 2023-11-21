# Ran after collecting all images to build CSV of resistor images and their respective values.
# Prepares zip file for kaggle dataset.
# Note: This script isn't meant to be ran after building the Kaggle dataset.

import csv
import json
import os
import shutil
import traceback
from pathlib import Path

WATTS = 0.25
DATA_ROOT = Path('./data/')

# parse resistor values from filenames
def parse_filename(f: str) -> dict:
    try:
        split = f.split('-')
        ohms = split[1]

        if 'R' in ohms:
            i = ohms.find('R')
            hundreds = int(ohms[:i])
            decimal = 0 if i == len(ohms) - 1 else (int(ohms[i+1]) / 10.0)
            ohms = hundreds + decimal
        elif 'K' in ohms:
            i = ohms.find('K')
            thousands = (int(ohms[:i]) * 1000)
            hundreds = 0 if i == len(ohms) - 1 else (int(ohms[i+1]) * 100)
            ohms = thousands + hundreds
        elif 'M' in ohms:
            i = ohms.find('M')
            millions = (int(ohms[:i]) * 1000000)
            thousands = 0 if i == len(ohms) - 1 else (int(ohms[i+1]) * 100000)
            ohms = millions + thousands
        else:
            raise ValueError('Bad filename found', f)
        
        band_count = int(split[0].replace('B', ''))
        tolerance = float(split[2].replace('T', '')) / 100.0

        with open(DATA_ROOT / f'{band_count}-band{os.sep}{band_count}-band.json', 'r') as bands_f:
            band_mapping = json.load(bands_f)
        bands = ' '.join(band_mapping[str(ohms)])
        ohms = float(ohms)

        return {'band_count': band_count, 'ohms': ohms, 'tolerance': tolerance, 'watts': WATTS, 'bands': bands}
    
    except Exception:
        print('Failed to parse', f)
        print(traceback.format_exc())
        exit(1)

def main():
    with open(DATA_ROOT / 'train.csv', 'w+', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)

        for i, p in enumerate(Path(DATA_ROOT).rglob('*.jpg')):
            r = parse_filename(p.name)
            if i == 0:
                header = ['image']
                header.extend(list(r.keys()))
                csv_writer.writerow(header)
            
            img_path = os.path.join(*os.path.split(p)[0].split(os.sep)[1:], p.name)
            row = [img_path, r['band_count'], r['ohms'], r['tolerance'], r['watts'], r['bands']]
            csv_writer.writerow(row)
    
    shutil.make_archive('resistors', 'zip', DATA_ROOT)

if __name__ == '__main__':
    main()
