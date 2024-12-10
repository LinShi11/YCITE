import boto3
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
import pandas as pd
import os
import csv
import re

def list_avail_models():
    # Initialize the Bedrock client
    load_dotenv()
    client = boto3.client('bedrock', region_name='us-east-1')
    # List available foundation models
    response = client.list_foundation_models()
    # Check and iterate over the ‘modelSummaries’ key
    if 'modelSummaries' in response:
        for model in response['modelSummaries']:
            print(f"Model ID: {model['modelId']}")
            print(f"Model Name: {model['modelName']}")
            print(f"Provider: {model['providerName']}")
            print(f"Input Modalities: {model['inputModalities']}")
            print(f"Output Modalities: {model['outputModalities']}")
            print(f"Inference Types Supported: {model['inferenceTypesSupported']}")
            print(f"Model Lifecycle Status: {model['modelLifecycle']['status']}")
            print("-" * 40)
    else:
        print("No models found or an error occurred.")

def load_model(model_id:str):
    load_dotenv()
    llm = ChatBedrock(
        model_id = model_id
        )
    return llm

def test_model(model_id: str):
    # llm = load_model(model_id)
    test_query = "Hello, can you hear me?"
    try:
        response = llm.invoke(test_query)
        print(f"Model response: {response}")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    

def extract_label_and_explanation(response):
    """
    Extracts the label and explanation from the response text using regex.
    If the label is not found, it prints the response for debugging.
    """
    # Regex to extract label and explanation, with optional < and > brackets
    label_match = re.search(r"label:\s*<?(\d+)>?", response, re.IGNORECASE)
    explanation_match = re.search(r"explanation:\s*<?(.*?)>?$", response, re.IGNORECASE | re.DOTALL)

    if label_match:
        label = int(label_match.group(1))
    else:
        # Print the response if the label is not found
        print(f"Label not found in response: {response}")
        label = None  # Assign None if label is missing

    # Extract explanation if found, else default to 'No explanation provided'
    explanation = (
        explanation_match.group(1).strip() if explanation_match else "No explanation provided"
    )

    return label, explanation

def process_file(input_csv, output_csv):
    # Read the input CSV file
    df = pd.read_csv(input_csv)

    # Apply the extraction function to the 'response' column
    df[['label', 'explanation']] = df['response'].apply(
        lambda x: pd.Series(extract_label_and_explanation(x))
    )

    # Count the number of rows with no label
    missing_label_count = df['label'].isna().sum()
    print(f"Number of rows with no label: {missing_label_count}")

    # Remove rows with missing labels
    df_clean = df.dropna(subset=['label'])

    # Count the number of rows where label is 3
    label_3_count = (df_clean['label'] == 3).sum()
    print(f"Number of rows with label 3: {label_3_count}")

    # Remove all rows where label is 3
    df_clean = df_clean[df_clean['label'] != 3]

    # Count the number of rows with "No explanation provided"
    no_explanation_count = (df_clean['explanation'] == "No explanation provided").sum()
    print(f"Number of rows with 'No explanation provided': {no_explanation_count}")

    # Select the required columns for the final DataFrame
    final_df = df_clean[['previous_sentence', 'current_sentence', 'label', 'explanation']]

    # Save the final DataFrame to a CSV file
    final_df.to_csv(output_csv, index=False)
    print(f"Data has been successfully saved to {output_csv}")
    print(f"Number of rows after removing missing labels and label 3s: {len(final_df)}")




# Example usage
if __name__ == "__main__":

    # Step 1: Make sure you have access and can see all the available models
    # List available models
    list_avail_models()
    

    # Step 2: Choosing a model to run the LLM commands
    # Choose a model ID from the list
    chosen_model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"


    # Test the model, you can uncomment this part to try it if you want.
    # if test_model(chosen_model_id):
    #     print("Model test successful!")
    # else:
    #     print("Model test failed. Please check your AWS Bedrock key and settings.")

    file_path = "10k_input.xlsx"
    output_file = "llm_output.csv"

    llm = load_model(chosen_model_id)
    

    # Load the Excel file using pandas
    try:
        df = pd.read_excel(file_path, engine='openpyxl')  
        print(df.head())  # Display the first few rows for verification
        print(f"Number of rows: {len(df)}")
        for index, row in df.iterrows():
            print(f"Index: {index}")
            if(index <= 1513):
                continue
            if(index > 11000):
                break
            ask_citation_question(llm, row['previous_sentence'], row["current_sentence"], output_file)
    except Exception as e:
        print(f"An error occurred while reading the Excel file: {e}")

    print("The program has ended")
    



    # Step 3: since the previous step is saved, then process all the results from LLM which is saved
    input_csv = "llm_output.csv"  # Replace with the path to your CSV file
    output_excel = "training_data.csv"  # Replace with the desired output path

    # Run the function to process the file
    process_file(input_csv, output_excel)