from flask import Flask, request, jsonify, Response
import requests
import time
import psutil  # For system metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Metrics for API requests
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')  # Total requests
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')  # Request latency
THROUGHPUT = Counter('http_requests_throughput', 'Total number of requests per second')  # Throughput
SUCCESS_COUNT = Counter('prediction_success_total', 'Total successful predictions') # NEW METRIC: Success Count
ERROR_COUNT = Counter('prediction_error_total', 'Total failed predictions') # NEW METRIC: Error Count


# System metrics
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')  # CPU Usage
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')  # RAM Usage

# Prometheus metrics endpoint
@app.route('/metrics', methods=['GET'])
def metrics():
    # Update system metrics every time /metrics is accessed
    CPU_USAGE.set(psutil.cpu_percent(interval=1))  # Ambil data CPU usage (persentase)
    RAM_USAGE.set(psutil.virtual_memory().percent)  # Ambil data RAM usage (persentase)

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# Endpoint for API prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()  # Generate request count metric
    THROUGHPUT.inc()  # Generate throughput metric

    # Set the API URL for the model server
    api_url = "http://mlserver:8080/invocations"
    data = request.get_json()

    try:
        response = requests.post(api_url, json=data)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        end_time = time.time()
        REQUEST_LATENCY.observe(end_time - start_time) # Get response latency

        response_data = response.json()
        SUCCESS_COUNT.inc() # Increment success count on successful proxy
        return jsonify(response_data), 200

    except requests.exceptions.RequestException as e:
        end_time = time.time()
        REQUEST_LATENCY.observe(end_time - start_time) # Observe latency for errors
        ERROR_COUNT.inc() # Increment error count on proxy failure
        return jsonify({"error": str(e), "message": "Failed to get prediction from model API"}), 500

if __name__ == '__main__':
    # Flask app will run on port 5000
    app.run(host='0.0.0.0', port=5000)