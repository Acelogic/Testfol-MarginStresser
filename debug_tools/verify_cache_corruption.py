import pickle
import os
import sys

# Mocking the updated logic snippet for testing independently or just run this
# Better: Just attempt to load the corrupt file using the logic I just added.

cache_path = "data/api_cache/corrupt_test_file.pkl"

if not os.path.exists(cache_path):
    print("Test file missing!")
    sys.exit(1)

print(f"Testing load on corrupted file: {cache_path}")

try:
    with open(cache_path, "rb") as f:
        pickle.load(f)
except (EOFError, pickle.UnpicklingError, IndexError, ImportError, ValueError) as e:
    print(f"⚠️ Cache Corruption Detected ({e}). Deleting corrupted file: {cache_path}")
    os.remove(cache_path)
    print("SUCCESS: File detected as corrupt and deleted.")
except Exception as e:
    print(f"FAILED: Caught generic exception instead of specific ones: {type(e)}")

if os.path.exists(cache_path):
    print("FAILED: File still exists!")
else:
    print("CONFIRMED: File is gone.")
