import os
import pathlib
import hashlib
from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi

# Configuration
# TODO: Update this if the 2026 competition has a different slug
COMPETITION_NAME = "vesuvius-challenge-ink-detection" 
# Target: Scroll 1, first 100 slices (00 to 99)
SCROLL_ID = "1"
SLICE_RANGE = range(100) # 0 to 99
DATA_ROOT = pathlib.Path("data")

# Optimized Directory Structure for RTX 3060 (Simple local storage)
# We store raw data separately.
RAW_DIR = DATA_ROOT / "native" / "train" / SCROLL_ID
SURFACE_VOLUME_DIR = RAW_DIR / "surface_volume"
MASK_DIR = RAW_DIR  # mask.png usually sits at root of scroll dir or in distinct folder depending on comp

def get_remote_paths():
    """Generates the list of remote file paths to download."""
    files = []
    
    # Surface Volume Slices (00.tif to 99.tif)
    for i in SLICE_RANGE:
        filename = f"{i:02d}.tif"
        # Remote path structure usually follows: train/1/surface_volume/00.tif
        remote_path = f"train/{SCROLL_ID}/surface_volume/{filename}"
        local_path = SURFACE_VOLUME_DIR / filename
        files.append((remote_path, local_path))
    
    # Mask file
    # Remote path: train/1/mask.png
    mask_remote = f"train/{SCROLL_ID}/mask.png"
    mask_local = RAW_DIR / "mask.png"
    files.append((mask_remote, mask_local))
    
    # Inklabels (if strictly needed for training, usually yes)
    # Remote path: train/1/inklabels.png
    ink_remote = f"train/{SCROLL_ID}/inklabels.png"
    ink_local = RAW_DIR / "inklabels.png"
    files.append((ink_remote, ink_local))

    return files

def calculate_md5(file_path):
    """Calculates MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def ensure_folders():
    """Creates necessary local directories."""
    SURFACE_VOLUME_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created/Verified directory: {SURFACE_VOLUME_DIR}")

def download_file_resumable(api, competition, remote_path, local_path):
    """Downloads a file if it doesn't exist or is incomplete (simple check)."""
    
    # Kaggle API doesn't give us easy remote file size without listing files first
    # So we will try to list it to get metadata
    # listing = api.competition_list_files(competition, name=str(pathlib.Path(remote_path).name))
    # This is often slow/flakes for single files inside deep folders.
    # Instead, we'll try the download. If local file exists, we check if it seems valid.
    
    if local_path.exists():
        # Check if file has reasonable size (avoid empty or error-page files)
        if local_path.stat().st_size > 1000:
             print(f"[SKIP] {local_path.name} exists and seems valid.")
             return
        else:
             print(f"[RE-DOWNLOAD] {local_path.name} exists but is too small ({local_path.stat().st_size} bytes). Deleting.")
             local_path.unlink()

    print(f"[DOWNLOADING] {remote_path} -> {local_path}")
    
    # Ensure parent exists usually handled by ensure_folders, but just in case
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # api.competition_download_file downloads to local path, but usually as a zip if it's large?
    # Or direct file. For many competition files, it puts them in the current working dir.
    # We need to be careful with `path` argument.
    
    # NOTE: Kaggle API's `competition_download_file` behaves oddly with paths.
    # It often downloads to `path` with the base filename.
    
    api.competition_download_file(
        competition=competition,
        file_name=remote_path,
        path=local_path.parent,
        force=False,
        quiet=False
    )
    
    # Check if it was downloaded as a zip (common for some extensions)
    # If remote is .tif, it usually comes as .tif
    # However, checking the result is good.
    downloaded_file = local_path.parent / pathlib.Path(remote_path).name
    
    # If the API extracted relative path logic, it might be in a subdir.
    # Often it just drops `00.tif` in the `path`.
    possible_download = local_path.parent / local_path.name
    
    if not possible_download.exists():
            # Try finding it if api created full path structure?
            pass

def main():
    print("Initializing Kaggle API...")
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print("Error: Could not authenticate with Kaggle API. Ensure ~/.kaggle/kaggle.json is present.")
        print(e)
        return

    print(f"Target Competition: {COMPETITION_NAME}")
    print(f"Target Scroll: {SCROLL_ID}, Slices: 0-{SLICE_RANGE[-1]}")
    
    ensure_folders()
    
    files_to_download = get_remote_paths()
    checksums = []

    import concurrent.futures

    import time

    # Thread safe download worker
    def process_file(args):
        remote, local = args
        
        # Retry logic
        max_retries = 10
        base_delay = 5
        
        for attempt in range(max_retries):
            try:
                # Exponential backoff with jitter
                sleep_time = base_delay * (2 ** attempt) 
                if attempt > 0:
                    print(f"Waiting {sleep_time}s before retry {attempt+1}...")
                    time.sleep(sleep_time)
                
                download_file_resumable(api, COMPETITION_NAME, remote, local)
                break # Success
            except Exception as e:
                print(f"[WARNING] Attempt {attempt+1}/{max_retries} failed for {remote}: {e}")
                if "429" in str(e):
                     print("Rate limit hit. Sleeping longer...")
                     time.sleep(60) # Extra minute for 429s

                if attempt == max_retries - 1:
                    print(f"[ERROR] Final failure for {remote}")
                    return None
        
        try:
            # Handle potential zip file download
            zip_path = local.parent / (local.name + ".zip")
            
            # Handle potential zip file download
            zip_path = local.parent / (local.name + ".zip")
            if not local.exists() and zip_path.exists():
                # print(f"[INFO] Unzipping {zip_path.name}...") # Reduce noise in threads
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Debug: print content of zip if unexpected
                    # names = zip_ref.namelist()
                    # print(f"[DEBUG] Zip contents for {zip_path.name}: {names}")
                    zip_ref.extractall(local.parent)
                
                # Check if file exists now, if not maybe it was in a subdir in the zip?
                if not local.exists():
                     # Force move if it's in a subdirectory? 
                     # For now just warn with content list
                     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                         print(f"[ERROR] Extracted {zip_path.name} but {local.name} not found. Zip contents: {zip_ref.namelist()}")

                zip_path.unlink() 
            
            # Verify / Checksum
            if local.exists():
                md5 = calculate_md5(local)
                return f"{md5}  {local.name}"
            else:
                print(f"[WARNING] File {local.name} missing after download attempt.")
                return None
        except Exception as e:
            print(f"Error processing {remote}: {e}")
            return None

    # Parallel Download
    MAX_WORKERS = 1 # Reduced to avoid 429 Rate Limits
    print(f"Starting parallel download with {MAX_WORKERS} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_file, (remote, local)): local for remote, local in files_to_download}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(files_to_download), desc="Downloading"):
             result = future.result()
             if result:
                 checksums.append(result)

    # Save checksums
    checksum_file = RAW_DIR / "checksums.txt"
    with open(checksum_file, "w") as f:
        f.write("\n".join(checksums))
    
    print(f"Done! Checksums saved to {checksum_file}")

if __name__ == "__main__":
    main()
