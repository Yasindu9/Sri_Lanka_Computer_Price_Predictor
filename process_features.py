import pandas as pd
import numpy as np
import re
import os
import shutil

# 1. Define folder structure
folders = [
    "data/raw",
    "data/processed",
    "src/data",
    "src/features",
    "src/models",
    "notebooks"
]

def setup_structure():
    print("Setting up project structure...")
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    # Move raw data to data/raw/
    raw_file = 'computers_data_withoutprocessed.csv'
    if os.path.exists(raw_file):
        shutil.copy(raw_file, 'data/raw/')
        print(f"Copied {raw_file} to data/raw/")
    else:
        print(f"Warning: {raw_file} not found in current directory.")

# 2. Extract Features
def clean_price(price_str):
    if pd.isna(price_str):
        return np.nan
    # Remove 'Rs', commas and spaces
    clean_str = re.sub(r'[^\d]', '', str(price_str))
    if clean_str:
        return float(clean_str)
    return np.nan

def extract_brand(title):
    if pd.isna(title):
        return 'Unknown'
    brands = ['HP', 'Dell', 'Lenovo', 'Asus', 'Apple', 'Acer', 'MSI', 'Samsung', 'Huawei', 'Microsoft', 'Gigabyte', 'Sony']
    title_lower = title.lower()
    for b in brands:
        if b.lower() in title_lower:
            return b
    if 'macbook' in title_lower or 'ipad' in title_lower or 'imac' in title_lower:
         return 'Apple'
    return 'Other'

def extract_cpu_tier(title):
    if pd.isna(title):
        return 'Unknown'
    title = title.lower()
    # Intel Core
    if re.search(r'i9', title): return 'Core i9'
    elif re.search(r'i7', title): return 'Core i7'
    elif re.search(r'i5', title): return 'Core i5'
    elif re.search(r'i3', title): return 'Core i3'
    # Intel Ultra
    elif re.search(r'ultra\s*9', title): return 'Core Ultra 9'
    elif re.search(r'ultra\s*7', title): return 'Core Ultra 7'
    elif re.search(r'ultra\s*5', title): return 'Core Ultra 5'
    # AMD Ryzen
    elif re.search(r'ryzen\s*9', title): return 'Ryzen 9'
    elif re.search(r'ryzen\s*7', title): return 'Ryzen 7'
    elif re.search(r'ryzen\s*5', title): return 'Ryzen 5'
    elif re.search(r'ryzen\s*3', title): return 'Ryzen 3'
    # Apple Silicon
    elif re.search(r'\bm4\b', title): return 'Apple M4'
    elif re.search(r'\bm3\b', title): return 'Apple M3'
    elif re.search(r'\bm2\b', title): return 'Apple M2'
    elif re.search(r'\bm1\b', title): return 'Apple M1'
    # Other
    elif re.search(r'pentium', title): return 'Pentium'
    elif re.search(r'celeron', title): return 'Celeron'
    return 'Unknown'

def extract_cpu_gen(title):
    if pd.isna(title):
        return 'Unknown'
    # match patterns like '12th gen', '11th gen', '13th', 'gen 10'
    match = re.search(r'(\d{1,2})(?:st|nd|rd|th)?\s*gen', title.lower())
    if match:
        return int(match.group(1))
    
    # Check if title has something like i5-1135g7 (11th gen)
    match2 = re.search(r'i\d?-?(\d{2})\d{2}', title.lower())
    if match2:
        return int(match2.group(1))
        
    return 'Unknown'

def extract_ram(title):
    if pd.isna(title):
        return np.nan
    title = title.upper()
    match = re.search(r'(\d+)\s*(?:GB|G)\b(?!\s*VGA|\s*GRAPHICS)(?:.*RAM)?', title)
    if match:
        return int(match.group(1))
    return np.nan

def extract_storage_gb(title):
    if pd.isna(title):
        return np.nan
    title = title.upper()
    # 1TB or 2TB
    match_tb = re.search(r'(\d+)\s*TB', title)
    if match_tb:
        return int(match_tb.group(1)) * 1024
        
    # GB storage usually 128, 256, 512
    # Be careful not to match RAM if it says 8GB or 16GB, usually storage is larger
    matches = re.findall(r'(\d+)\s*GB', title)
    for m in matches:
        val = int(m)
        if val in [128, 240, 256, 500, 512, 1000]:
            return val
    return np.nan

def extract_storage_type(title):
    if pd.isna(title):
        return 'Unknown'
    title = title.lower()
    if 'nvme' in title:
        return 'NVMe SSD'
    elif 'ssd' in title or 'm.2' in title:
        return 'SSD'
    elif 'hdd' in title or 'hard' in title:
        return 'HDD'
    return 'Unknown/eMMC'

def has_dedicated_gpu(title):
    if pd.isna(title):
        return 0
    title = title.lower()
    gpu_keywords = ['rtx', 'gtx', 'rx ', 'radeon rx', 'mx130', 'mx230', 'mx330', 'mx450', 'vga']
    for keyword in gpu_keywords:
        if keyword in title:
            return 1
    return 0

def extract_condition(title):
    # ikman usually has separate condition tags, but from title we can guess
    if pd.isna(title):
        return 'Used'
    title = title.lower()
    if 'brand new' in title or 'brandnew' in title or 'new' in title:
        return 'New'
    return 'Used' # Default to Used if not specified as new

def process_data():
    input_file = 'data/raw/computers_data_withoutprocessed.csv'
    if not os.path.exists(input_file):
        input_file = 'computers_data_withoutprocessed.csv' # Fallback
        
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} records.")
    
    processed = pd.DataFrame()
    processed['Price (Target)'] = df['Price'].apply(clean_price)
    processed['Brand'] = df['Title'].apply(extract_brand)
    processed['CPU_Tier'] = df['Title'].apply(extract_cpu_tier)
    processed['CPU_Gen'] = df['Title'].apply(extract_cpu_gen)
    processed['RAM_GB'] = df['Title'].apply(extract_ram)
    processed['Storage_GB'] = df['Title'].apply(extract_storage_gb)
    processed['Storage_Type'] = df['Title'].apply(extract_storage_type)
    processed['Has_Dedicated_GPU'] = df['Title'].apply(has_dedicated_gpu)
    processed['Condition'] = df['Title'].apply(extract_condition)
    
    # Extract Location and Category
    def parse_location(loc_cat):
        if pd.isna(loc_cat):
            return 'Unknown'
        parts = str(loc_cat).split(',')
        if len(parts) > 0:
            return parts[0].strip()
        return 'Unknown'
        
    processed['Location'] = df['Location_Category'].apply(parse_location)
    
    # Is Member
    processed['Is_Member'] = df['Member'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
    
    # Filter out outliers or null prices
    processed = processed.dropna(subset=['Price (Target)'])
    
    # Filter out extreme outliers (e.g. less than Rs 5000 or more than Rs 3,000,000)
    processed = processed[(processed['Price (Target)'] > 5000) & (processed['Price (Target)'] < 3000000)]
    
    # Save the processed data
    output_file = 'computers_data_withprocessed.csv'
    processed.to_csv(output_file, index=False)
    
    # Also save a copy in processed folder
    processed.to_csv('data/processed/' + output_file, index=False)
    
    print(f"Data processing complete. {len(processed)} clean records saved to {output_file}.")
    print("\n--- Processed Data Info ---")
    processed.info()
    print("\n--- Processed Data Sample ---")
    print(processed.head())

if __name__ == "__main__":
    setup_structure()
    process_data()
