import json

try:
    with open("api_dump.json", "r") as f:
        data = json.load(f)
    print("Keys in file:", list(data.keys()))
    if "assets" in data:
        print("Assets found!")
        print("Assets keys:", list(data["assets"].keys()))
        first = list(data["assets"].keys())[0]
        print(f"First asset: {first}")
        print(f"First asset keys: {list(data['assets'][first].keys())}")
    else:
        print("Assets NOT found in file.")
except Exception as e:
    print(f"Error: {e}")
