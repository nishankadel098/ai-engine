import requests

url = "http://127.0.0.1:5000/ask"
data = {"message": "I need math notes"} # Aapne jo subject upload kiya hai wahi pucho

response = requests.post(url, json=data)
print("AI ka Jawab:", response.json())