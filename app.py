# Phase 2: Data Preprocessing, Remapping, and Tokenization (from local CSV)

#
# In this script, we will:
# 1. Load the 'banking77' dataset from local CSV files ('train.csv', 'test.csv').
# 2. Convert the Pandas DataFrames into a Hugging Face DatasetDict.
# 3. Apply the category mapping from Phase 1.
# 4. Check the distribution of our new categories.
# 5. Load a tokenizer and tokenize the entire dataset.
# 6. Save the processed dataset to disk for the next phase.
#

# Make sure you have the required libraries installed:
# pip install transformers datasets pandas scikit-learn torch matplotlib

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

def preprocess_banking_dataset_from_csv():
    """
    Loads, remaps, analyzes, and tokenizes the Banking77 dataset from local CSV files.
    """
    print("--- 1. Loading and Remapping Dataset from local CSVs ---")

    # --- IMPORTANT ---
    # This script assumes you have downloaded 'train.csv' and 'test.csv' from the
    # Banking77 GitHub repository and they are in the same folder as this script.
    try:
        train_df = pd.read_csv("dataset/train.csv")
        test_df = pd.read_csv("dataset/test.csv")
    except FileNotFoundError:
        print("\nERROR: Could not find 'train.csv' or 'test.csv'.")
        print("Please make sure these files are in the same directory as your script.")
        return

    # Convert the Pandas DataFrames into Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Combine them into a DatasetDict, which is the standard format
    raw_datasets = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    print("Successfully loaded data from local CSVs and converted to a DatasetDict.")
    print(raw_datasets)

    # Define the same category map from Phase 1
    category_map = {
        'activate_my_card': 'Billing Question','card_payment_fee_charged': 'Billing Question','card_payment_not_recognised': 'Billing Question','card_payment_wrong_exchange_rate': 'Billing Question','cash_withdrawal_charge': 'Billing Question','cash_withdrawal_not_recognised': 'Billing Question','exchange_charge': 'Billing Question','exchange_rate': 'Billing Question','exchange_via_app': 'Billing Question','fee_applied': 'Billing Question','fiat_currency_support': 'Billing Question','pending_card_payment': 'Billing Question','pending_cash_withdrawal': 'Billing Question','pending_top_up': 'Billing Question','pending_transfer': 'Billing Question','top_up_by_bank_transfer_charge': 'Billing Question','top_up_by_card_charge': 'Billing Question','top_up_failed': 'Billing Question','top_up_limits': 'Billing Question','top_up_reverted': 'Billing Question','transaction_charged_twice': 'Billing Question','transfer_fee_charged': 'Billing Question','beneficiary_not_allowed': 'Billing Question','card_arrival': 'Billing Question','card_delivery_estimate': 'Billing Question','card_linking': 'Billing Question','card_not_working': 'Billing Question','decline_card_payment': 'Billing Question','declined_cash_withdrawal': 'Billing Question','declined_transfer': 'Billing Question','direct_debit_payment_not_recognised': 'Billing Question','failed_transfer': 'Billing Question','topping_up_by_card': 'Billing Question','transfer_not_received_by_recipient': 'Billing Question','transfer_timing': 'Billing Question','wrong_amount_of_cash_received': 'Billing Question','wrong_exchange_rate_for_cash_withdrawal': 'Billing Question','cancel_transfer': 'Billing Question','request_refund': 'Billing Question',
        'app_does_not_work': 'Technical Issue','face_id_not_working': 'Technical Issue','fingerprint_not_working': 'Technical Issue','passcode_forgotten': 'Technical Issue','pin_blocked': 'Technical Issue','unable_to_verify_identity': 'Technical Issue','verify_my_identity': 'Technical Issue','getting_physical_card': 'Technical Issue',
        'ATMs_support': 'General Inquiry','account_blocked': 'General Inquiry','age_limit': 'General Inquiry','apple_pay_or_google_pay': 'General Inquiry','atm_support': 'General Inquiry','automatic_top_up': 'General Inquiry','balance_not_updated_after_bank_transfer': 'General Inquiry','balance_not_updated_after_cheque_or_cash_deposit': 'General Inquiry','card_about_to_expire': 'General Inquiry','card_acceptance': 'General Inquiry','card_swallowed': 'General Inquiry','change_pin': 'General Inquiry','contactless_not_working': 'General Inquiry','country_support': 'General Inquiry','disposable_card_limits': 'General Inquiry','edit_personal_details': 'General Inquiry','get_disposable_virtual_card': 'General Inquiry','get_physical_card': 'General Inquiry','getting_spare_card': 'General Inquiry','how_do_I_report_fraud': 'General Inquiry','lost_or_stolen_card': 'General Inquiry','lost_or_stolen_phone': 'General Inquiry','order_physical_card': 'General Inquiry','supported_cards_and_currencies': 'General Inquiry','verify_source_of_funds': 'General Inquiry','verify_top_up': 'General Inquiry','virtual_card_not_working': 'General Inquiry','what_are_my_limits': 'General Inquiry','what_is_a_disposable_virtual_card': 'General Inquiry','what_is_my_pin': 'General Inquiry','why_verify_identity': 'General Inquiry','terminate_account': 'General Inquiry'
    }

    # Define our new labels and create mappings for label encoding
    new_labels = ['Billing Question', 'Technical Issue', 'General Inquiry']
    label2id = {label: i for i, label in enumerate(new_labels)}
    id2label = {i: label for i, label in enumerate(new_labels)}

    def remap_labels_from_csv(example):
        # The CSV has a 'category' column with text labels
        original_label_name = example['category']
        new_label_name = category_map.get(original_label_name, "Unmapped")
        example['label'] = label2id.get(new_label_name)
        return example

    # Apply the mapping function and remove the old 'category' column
    processed_datasets = raw_datasets.map(remap_labels_from_csv, remove_columns=['category'])
    print("\nLabels have been successfully remapped to our 3 categories.")


    print("\n--- 2. Analyzing Category Distribution ---")
    # This part remains the same
    train_df_processed = pd.DataFrame(processed_datasets['train'])
    label_counts = train_df_processed['label'].value_counts().sort_index()
    label_counts.index = label_counts.index.map(lambda i: id2label[i])
    print("Distribution of new categories in the training set:")
    print(label_counts)
    label_counts.plot(kind='bar', title='Category Distribution')
    plt.ylabel('Number of Tickets')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    print("\n--- 3. Tokenizing the Dataset ---")
    # This part also remains the same
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    tokenized_datasets = processed_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    print("\nDataset tokenized and formatted for training:")
    print(tokenized_datasets["train"][0])


    print("\n--- 4. Saving Processed Dataset ---")
    tokenized_datasets.save_to_disk("banking77-processed")
    print("\nProcessed dataset saved to the 'banking77-processed' directory.")
    print("\nPhase 2 is complete! We are now ready for model training.")


if __name__ == "__main__":
    preprocess_banking_dataset_from_csv()

