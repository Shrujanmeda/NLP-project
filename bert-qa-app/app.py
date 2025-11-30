# app.py
from flask import Flask, request, render_template
import torch
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
import os

# --- 1. Initialize Flask App and BERT Model ---
app = Flask(__name__)

# Use a lighter model (DistilBERT) for edge deployment
MODEL_NAME = "distilbert-base-uncased-distilled-squad"

# Check for model existence before loading (a common edge optimization)
# In a real edge scenario, you might pre-download and save the model locally
try:
    # Load tokenizer and model once on startup
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForQuestionAnswering.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit gracefully if model cannot be loaded
    tokenizer, model = None, None

# --- 2. Define the Question Answering Logic ---
def get_bert_answer(question, context):
    """
    Finds the answer span within the context for a given question.
    """
    if not model or not tokenizer:
        return "Model not loaded. Check model path or internet connection."

    # Tokenize the inputs
    inputs = tokenizer.encode_plus(
        question, 
        context, 
        add_special_tokens=True, 
        return_tensors="pt",
        max_length=512,  # BERT limit
        truncation='only_second'
    )
    
    # Get the token IDs and segment IDs
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    # Model inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

    # Find the tokens with the highest 'start' and 'end' scores
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits) + 1  # +1 to make it inclusive

    # Convert tokens back to a string
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
    answer_tokens = tokens[answer_start:answer_end]
    
    # Reconstruct the answer string
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    # Handle case where the model returns an empty or invalid answer
    if answer.startswith('[CLS]') or answer.startswith('[SEP]'):
        return "Sorry, I couldn't find a clear answer in the provided FAQ context."
        
    return answer

# --- 3. Define the Context/FAQ Database (Simulated) ---
# In a real government FAQ scenario, this would be loaded from a file/DB.
# Note: The BERT model finds the answer *within* this text.
GOVT_FAQ_CONTEXT = (
    "The Department of Public Services (DPS) handles all permit applications. "
    "To apply for a building permit, you must submit Form 42B and pay a fee of $150. "
    "The processing time for new applications is typically 10 to 15 business days. "
    "Payments are accepted only via credit card or bank transfer, not cash. "
    "Office hours are Monday to Friday, 9:00 AM to 4:00 PM."
)


# --- 4. Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles both displaying the form and processing the question.
    """
    question = ""
    answer = ""
    if request.method == 'POST':
        question = request.form['question']
        answer = get_bert_answer(question, GOVT_FAQ_CONTEXT)
    
    # Pass all variables to the template
    return render_template(
        'index.html', 
        faq_context=GOVT_FAQ_CONTEXT, 
        question=question, 
        answer=answer
    )

# --- 5. Run the App ---

if __name__ == '__main__':
    # Use a specific port for deployment, often 5000 is default
    # Set debug=False for production/edge deployment
    app.run(host='0.0.0.0', port=5000, debug=True)