import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "features": [
        20000,2,2,1,24,2,2,0,0,0,0,
        3913,3102,689,0,0,0,
        0,689,0,0,0,0
    ]
}

response = requests.post(url, json=data)

print(response.json())
