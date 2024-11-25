# load_test_post.py
import requests
import time
from concurrent.futures import ThreadPoolExecutor

API_URLS = {
    "flask": "http://127.0.0.1:5000/predict",
    "fastapi": "http://127.0.0.1:8000/predict",
}

def test_api(url, payload):
    start_time = time.time()
    response = requests.post(url, json=payload)
    end_time = time.time()
    return end_time - start_time, response.json()

def run_test(api, num_requests=10):
    url = API_URLS[api]
    payload = {"number": 10}  # Example payload
    
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        results = list(executor.map(lambda _: test_api(url, payload), range(num_requests)))
  
    total_time = sum(result[0] for result in results)
    return {
        "api": api,
        "total_time": total_time,
        "average_time": total_time / num_requests,
        "responses": [result[1] for result in results],
    }

if __name__ == "__main__":
    #flask_results = run_test("flask", num_requests=10)
    fastapi_results = run_test("fastapi", num_requests=10)

    #print("\nFlask Results:", flask_results)
    print("\nFastAPI Results:", fastapi_results)
