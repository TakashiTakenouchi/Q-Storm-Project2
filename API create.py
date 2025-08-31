import requests

url = "https://api.anthropic.com/v1/organizations/api_keys/{api_key_id}"

headers = {
    "x-api-key": "<x-api-key>",
    "anthropic-version": "<anthropic-version>"
}

response = requests.request("GET", url, headers=headers)

print(response.text)
{
  "id": "apikey_01Rj2N8SVvo6BePZj99NhmiT",
  "type": "api_key",
  "name": "Developer Key",
  "workspace_id": "wrkspc_01JwQvzr7rXLA5AGx3HKfFUJ",
  "created_at": "2024-10-30T23:58:27.427722Z",
  "created_by": {
    "id": "user_01WCz1FkmYMm4gnmykNKUu3Q",
    "type": "user"
  },
  "partial_key_hint": "sk-ant-api03-R2D...igAA",
  "status": "active"
}