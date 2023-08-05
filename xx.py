import requests

url = "https://api.openaq.org/v2/cities?limit=100&page=1&country=US&city=Houston&order_by=city"

headers = {"accept": "application/json"}

response = requests.get(url, headers=headers)

print(response.text)