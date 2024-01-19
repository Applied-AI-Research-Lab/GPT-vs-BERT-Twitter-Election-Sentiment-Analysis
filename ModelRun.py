import csv
from GPTmethods import GPTmethods
from LLAMAmethods import LLAMAmethods
# import replicate
import time

import csv


def run_sentiment(model, model_id, dataset, identifier):
    # Specify the output CSV file name
    output_file = f'datasets/{model}_{identifier}_predictions.csv'

    with open(dataset, 'r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)  # Create a CSV reader
        header = next(csvreader, None)  # Skip the header row

        # Open the output CSV file for writing
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile_out:
            csvwriter = csv.writer(csvfile_out)

            # Add 'prediction' column to the header only once
            if 'prediction' not in header:
                header.append(model+'_prediction')
                csvwriter.writerow(header)

            # Loop through each row in the CSV file
            for row in csvreader:
                comment = row[0]
                comment = comment[:200]
                print(f"Comment: {comment}")

                if model == 'gpt':
                    # Run GPT Model
                    GPT = GPTmethods(model_id=model_id)
                    prediction = GPT.gpt_ratings(comment)  # the prediction
                    print(f"Prediction: {prediction}")
                elif model == 'llama':
                    # Run LLAMA2 Model
                    LLAMA = LLAMAmethods(model_id=model_id)
                    prediction = LLAMA.llama_ratings(comment)  # the prediction
                    print(f"Prediction: {prediction}")
                else:
                    prediction = 'error'

                # Add the prediction value to the 'prediction' column for each row
                row.append(prediction)

                # Write the row to the output CSV file
                csvwriter.writerow(row)

                # Add a delay of 5 seconds (reduced for testing)
                time.sleep(2)

            print(f"Predictions added to the 'prediction' column in {output_file}")

# Base models
# run_sentiment('gpt', 'gpt-3.5-turbo', 'datasets/reddit_test_dataset.csv', 'reddit')
# run_sentiment('llama', 'meta/llama-2-70b-chat', 'datasets/reddit_test_dataset.csv', 'reddit')
# Fine-tuned models
# run_sentiment('gpt', 'ft:gpt-3.5-turbo-1106:personal::8ftGgVPT', 'datasets/twitter_test_dataset.csv', 'twitter')