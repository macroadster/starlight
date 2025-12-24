
import requests
import json

url = "https://starlight.local/mcp/"
headers = {
    "Content-Type": "application/json",
    "X-API-KEY": "8574c188f8953ed7568aa4ea41f94f24a8abe88f4de04fb2ffef789b601f87ae"
}
data = {
    "tool_name": "list_tasks",
    "status": "available"
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response status code: {e.response.status_code}")
        print(f"Response body: {e.response.text}")

