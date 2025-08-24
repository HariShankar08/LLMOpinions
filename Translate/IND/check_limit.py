import requests
import json
response = requests.get(
  url="https://openrouter.ai/api/v1/key",
  headers={
    "Authorization": f"Bearer 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"  # Set your OpenRouter API key here
}
)
print(json.dumps(response.json(), indent=2))