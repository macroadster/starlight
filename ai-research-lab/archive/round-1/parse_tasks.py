import json
import sys

if __name__ == "__main__":
    log_file_path = "/Users/eric/.gemini/tmp/9c90e59469c631c9273a67444b560a114d73945daa1d131659c5ea003fa3aec1/mcp_response.log"
    
    json_content = ""
    with open(log_file_path, 'r') as f:
        json_content = f.read().strip() # Read the whole file, assuming it's just JSON now.

    try:
        data = json.loads(json_content)
        # Print the entire result object for inspection
        print(json.dumps(data.get("result", {}), indent=2))
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}", file=sys.stderr)