import requests
import json
import os
import pandas as pd
import numpy as np

def get_predictions_from_csv(
    csv_filename="diabetes_preprocessing.csv",
    service_url="http://localhost:5000",
    batch_size=500 # Define a batch size for sending data in chunks
):
    """
    Reads data from a CSV file, prepares it in batches, and sends prediction requests
    to the ML model service via the custom exporter.

    Args:
        csv_filename (str): The name of the CSV file containing the input data.
        service_url (str): The base URL of your custom metrics exporter.
        batch_size (int): The number of rows to send in each prediction request.
    """
    try:
        script_dir = os.path.dirname(__file__)
        csv_filepath = os.path.join(script_dir, csv_filename)

        print(f"Attempting to load data from: {csv_filepath}")
        df = pd.read_csv(csv_filepath)
        print(f"Successfully loaded {len(df)} rows from {csv_filename}.")

        # IMPORTANT: Drop the target column if it exists, as the model expects only features
        # Adjust 'Diabetes_binary' if your target column has a different name
        if 'Diabetes_binary' in df.columns:
            df = df.drop(columns=['Diabetes_binary'])
            print(f"Dropped 'Diabetes_binary' column. Remaining features: {df.columns.tolist()}")
        else:
            print("No 'Diabetes_binary' column found to drop. Assuming all columns are features.")

        # Prepare columns for the MLflow dataframe_split format
        columns = df.columns.tolist()
        total_rows = len(df)
        num_batches = (total_rows + batch_size - 1) // batch_size # Calculate total number of batches

        all_prediction_results = []

        print(f"\nStarting batch predictions. Total rows: {total_rows}, Batch size: {batch_size}, Total batches: {num_batches}")

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]

            # Construct the dataframe_split JSON payload for the current batch
            payload = {
                "dataframe_split": {
                    "columns": columns,
                    "data": batch_df.values.tolist() # Convert DataFrame values to list of lists
                }
            }

            inference_url = f"{service_url}/predict" # Target the exporter's /predict endpoint
            headers = {"Content-Type": "application/json"}

            print(f"Processing batch {i + 1}/{num_batches} (rows {start_idx} to {end_idx - 1}). Sending request to: {inference_url}")

            try:
                response = requests.post(inference_url, headers=headers, json=payload, timeout=60) # Increased timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                prediction_results = response.json()
                all_prediction_results.append(prediction_results)

                # Process and print results for this batch (optional, can be very verbose for large datasets)
                if "predictions" in prediction_results and isinstance(prediction_results["predictions"], list):
                    # For large datasets, don't print every single prediction
                    # print(f"  Batch {i+1} predictions received: {prediction_results['predictions'][:5]}...") # Print first 5
                    pass # Keep silent for large logs
                else:
                    print(f"  Batch {i+1} unexpected prediction format: {prediction_results}")

            except requests.exceptions.ConnectionError as e:
                print(f"  Caught ConnectionError for batch {i+1}: Could not connect to the service at {service_url}. Details: {e}")
            except requests.exceptions.Timeout as e:
                print(f"  Caught Timeout for batch {i+1}: The request timed out connecting to {service_url}. Details: {e}")
            except requests.exceptions.HTTPError as e:
                print(f"  Caught HTTPError for batch {i+1}: {e.response.status_code} - {e.response.text}")
            except json.JSONDecodeError:
                print(f"  Caught JSONDecodeError for batch {i+1}: Invalid JSON response from service.")
            except Exception as e:
                print(f"  Caught an unexpected error for batch {i+1}: {e}")

        print(f"\nFinished processing all {num_batches} batches.")
        # You can now process all_prediction_results if needed
        print(f"Total prediction responses collected: {len(all_prediction_results)}")

    except FileNotFoundError:
        print(f"Error: Data file '{csv_filepath}' not found. Make sure it's in the same directory as the script.")
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file '{csv_filepath}' is empty.")
    except Exception as e:
        print(f"An unexpected error occurred during CSV processing: {e}")
    
get_predictions_from_csv("diabetes_preprocessing.csv", "http://localhost:5000", batch_size=500)