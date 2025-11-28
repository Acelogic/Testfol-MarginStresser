import json

try:
    with open("api_dump.json", "r") as f:
        data = json.load(f)
    
    if "rebalancing_events" in data:
        print("rebalancing_events type:", type(data["rebalancing_events"]))
        print("rebalancing_events sample:", json.dumps(data["rebalancing_events"][:2], indent=2))
        
    if "rebalancing_stats" in data:
        print("rebalancing_stats type:", type(data["rebalancing_stats"]))
        print("rebalancing_stats:", json.dumps(data["rebalancing_stats"], indent=2))
        
except Exception as e:
    print(f"Error: {e}")
