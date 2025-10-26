from transformers import pipeline
import os

def test_classifier():
    """
    Loads the fine-tuned model and tests it on sample customer support tickets.
    """
    model_path = "./my-ticket-classifier-final"

    print(f"--- 1. Loading Model from '{model_path}' ---")
    
    # Check if the model directory exists
    if not os.path.exists(model_path):
        print(f"\nERROR: Model directory not found at '{model_path}'")
        print("Please make sure you have successfully run the Phase 3 training script and the model was saved correctly.")
        return

    try:
        # Load the text classification pipeline with our fine-tuned model
        classifier = pipeline("text-classification", model=model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"\nERROR: Failed to load the model. Details: {e}")
        return

    print("\n--- 2. Preparing Sample Tickets for Testing ---")
    
    # Create a list of new, unseen tickets to test the model's performance
    sample_tickets = [
        "My credit card was charged twice for the same transaction, can you please reverse one?", # Expected: Billing Question
        "I can't log in to the mobile banking app, it keeps saying 'invalid credentials'.",     # Expected: Technical Issue
        "How do I apply for a home loan and what are the interest rates?",                      # Expected: General Inquiry
        "The ATM did not dispense any cash but my account was debited.",                        # Expected: Billing Question
        "Your website is down, I can't access my account details.",                             # Expected: Technical Issue
        "What are your branch opening hours on the weekend?",                                   # Expected: General Inquiry
    ]
    print(f"Found {len(sample_tickets)} sample tickets to classify.")

    print("\n--- 3. Classifying Tickets ---")
    
    # Run each ticket through the classifier and print the results
    for ticket in sample_tickets:
        # The pipeline returns a list containing a dictionary
        result = classifier(ticket)
        # The top prediction is the first element
        prediction = result[0]
        label = prediction['label']
        score = prediction['score']
        
        # Print in a clean, readable format
        print(f"\nTicket: '{ticket}'")
        print(f"Predicted Category: {label} (Confidence: {score:.2%})")
        print("-" * 30)

    #print("\nPhase 4 is complete. You have successfully built and tested a custom NLP classifier!")


if __name__ == "__main__":
    test_classifier()

