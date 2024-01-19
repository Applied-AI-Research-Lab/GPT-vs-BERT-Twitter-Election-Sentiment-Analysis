import pandas as pd
from sklearn.model_selection import train_test_split
from DBmethods import DBmethods
import re
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Rename column to comment
def rename_columns(csv_name):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('datasets/' + csv_name)

    if csv_name == 'twitter_full.csv':
        df = df.rename(columns={'clean_text': 'comment'})
        df = df.rename(columns={'category': 'sentiment'})
    else:
        df = df.rename(columns={'clean_comment': 'comment'})
        df = df.rename(columns={'category': 'sentiment'})

    # Save the modified DataFrame back to the CSV file
    df.to_csv('datasets/' + csv_name, index=False)


# Clean the dataset by removing empty rows, spaces, and breaks at the beginning and end of each comment.
def clean_dataset(csv_name):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('datasets/' + csv_name, encoding='utf-8')

    # Remove rows with empty values in the 'comment' column
    df = df.dropna(subset=['comment'])

    # Remove leading and trailing spaces from the 'comment' column
    df['comment'] = df['comment'].str.strip()

    # Remove non-ASCII characters and handle both escaped and unescaped newline symbols
    df['comment'] = df['comment'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
    df['comment'] = df['comment'].replace(r'\\n', ' ', regex=True)
    df['comment'] = df['comment'].replace(r'\n', ' ', regex=True)

    # Save the modified DataFrame back to the CSV file
    df.to_csv('datasets/' + csv_name, index=False, encoding='utf-8')

def randomly_select_dataset(csv_name):
    # Read the original CSV file into a pandas DataFrame
    df_full = pd.read_csv('datasets/' + csv_name)

    # Randomly sample 2,000 rows from the DataFrame
    df_sampled = df_full.sample(n=2000, random_state=42)

    if csv_name == 'twitter_full.csv':
        # Save the sampled DataFrame to a new CSV file named 'twitter.csv'
        df_sampled.to_csv('datasets/twitter.csv', index=False)
    else:
        df_sampled.to_csv('datasets/reddit.csv', index=False)


# Split dataset to train, validation, test
def split_dataset(csv_name):
    df = pd.read_csv('datasets/' + csv_name)  # Load the dataset

    # Split the dataset into training (70%), validation (15%), and test (15%) sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    name = csv_name.split('.')[0]  # remove the .csv from csv_name

    # Save each split dataset to a CSV file
    train_df.to_csv("datasets/" + name + "_train_dataset.csv", index=False)
    valid_df.to_csv("datasets/" + name + "_validation_dataset.csv", index=False)
    test_df.to_csv("datasets/" + name + "_test_dataset.csv", index=False)

def fix_dot_sentiment_values(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Map the values in the "sentiment" column to 1, -1, or 0
    df['sentiment'] = df['sentiment'].map({1.0: 1, -1.0: -1, 0.0: 0})

    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path, index=False)


def rename_column(csv_path, old_column_name, new_column_name, output_csv_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Check if the old column exists
    if old_column_name not in df.columns:
        print(f"Column '{old_column_name}' not found in the DataFrame.")
        return

    # Rename the old column to the new column name
    df.rename(columns={old_column_name: new_column_name}, inplace=True)

    # Save the modified DataFrame back to a new CSV file
    df.to_csv(output_csv_path, index=False)

def merge_reddit_csvs(gpt_csv_path, llama_csv_path, output_csv_path):
    # Load the two CSV files into DataFrames
    gpt_df = pd.read_csv(gpt_csv_path)
    llama_df = pd.read_csv(llama_csv_path)

    # Merge DataFrames based on common columns 'comment' and 'sentiment'
    merged_df = pd.merge(gpt_df, llama_df, on=['comment', 'sentiment'], how='inner')

    # Save the merged DataFrame to a new CSV file with the specified columns
    merged_df.to_csv(output_csv_path, columns=['comment', 'sentiment', 'gpt_prediction', 'llama_prediction'], index=False)

def evaluate_predictions(csv_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Extract actual sentiments and predictions
    actual_sentiments = df['sentiment']
    gpt_predictions = df['gpt_predictions']
    bert_predictions = df['bert_predictions']

    # Evaluate GPT predictions
    print("Evaluation for GPT predictions:")
    evaluate_metrics(actual_sentiments, gpt_predictions)

    # Evaluate BERT predictions
    print("\nEvaluation for BERT predictions:")
    evaluate_metrics(actual_sentiments, bert_predictions)

def evaluate_metrics(actual, predictions):
    # Evaluate metrics
    accuracy = accuracy_score(actual, predictions)
    precision = precision_score(actual, predictions, average='weighted')
    recall = recall_score(actual, predictions, average='weighted')
    f1 = f1_score(actual, predictions, average='weighted')

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Print classification report
    class_report = classification_report(actual, predictions)
    print("Classification Report:\n", class_report)


# Clean and Split Dataset
rename_columns('twitter_full.csv')
rename_columns('reddit_full.csv')
clean_dataset('twitter_full.csv')
clean_dataset('reddit_full.csv')
randomly_select_dataset('twitter_full.csv')
randomly_select_dataset('reddit_full.csv')
split_dataset('twitter.csv')
split_dataset('reddit.csv')

# Create train and validation json files for fine-tuning
DB = DBmethods()
print(DB.create_jsonl('gpt', 'twitter_train_dataset.csv'))
print(DB.create_jsonl('llama', 'reddit_train_dataset.csv'))
print(DB.create_jsonl('gpt', 'twitter_validation_dataset.csv'))
print(DB.create_jsonl('llama', 'reddit_validation_dataset.csv'))

# Fix dot sentiment values
fix_dot_sentiment_values('datasets/twitter_test_dataset.csv')
fix_dot_sentiment_values('datasets/twitter_train_dataset.csv')
fix_dot_sentiment_values('datasets/twitter_validation_dataset.csv')
fix_dot_sentiment_values('datasets/gpt_twitter_predictions.csv')

# Rename column
rename_column('datasets/twitter_predictions-down.csv', 'prediction', 'gpt_predictions',
              'datasets/twitter_predictions.csv')

# Merge Reddit CSVs
merge_reddit_csvs('datasets/gpt_reddit_predictions.csv', 'datasets/llama_reddit_predictions.csv', 'datasets/reddit_predictions.csv')

# evaluate predictions
evaluate_predictions('datasets/twitter_predictions.csv')

def plot_confusion_matrix(actual, predictions, labels):
    # Compute confusion matrix
    cm = confusion_matrix(actual, predictions, labels=labels)

    # Create a DataFrame for better visualization with seaborn
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('BERT AdamW Prediction')
    plt.ylabel('Sentiment')
    plt.show()

# Example usage
csv_path = 'datasets/twitter_predictions.csv'
df = pd.read_csv(csv_path)

# Extract actual sentiments and predictions
actual_sentiments = df['sentiment']
gpt_predictions = df['bert_predictions']

# Define labels (unique sentiments)
labels = df['sentiment'].unique()

# Plot confusion matrix for GPT predictions
plot_confusion_matrix(actual_sentiments, gpt_predictions, labels)