from flask import Flask, request, render_template
import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering

app = Flask(__name__)

# Use a lighter model for edge deployment
MODEL_NAME = "distilbert-base-uncased-distilled-squad"

# Keep tokenizer/model as globals but don't force-download at import time.
# This lets the web server bind quickly; model will be loaded on first request.
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        try:
            tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
            model = DistilBertForQuestionAnswering.from_pretrained(MODEL_NAME)
            model.to("cpu")
        except Exception as e:
            # Log to stdout/stderr so Render shows the error in build/deploy logs
            print(f"[model load error] {e}")
            tokenizer, model = None, None

def get_bert_answer(question, context):
    """
    Finds the answer span within the context for a given question.
    Loads the model lazily on first call if not already loaded.
    """
    if tokenizer is None or model is None:
        load_model()
        if tokenizer is None or model is None:
            return "Model not loaded. Check logs or model availability."

    # Tokenize the inputs (make sure we stay within 512 tokens)
    inputs = tokenizer.encode_plus(
        question,
        context,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

    answer_start = torch.argmax(start_logits, dim=1).item()
    answer_end = torch.argmax(end_logits, dim=1).item() + 1

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    # Guard indices
    answer_start = max(0, min(answer_start, len(tokens) - 1))
    answer_end = max(answer_start, min(answer_end, len(tokens)))

    answer_tokens = tokens[answer_start:answer_end]
    answer = tokenizer.convert_tokens_to_string(answer_tokens).strip()

    if not answer or answer.startswith("[CLS]") or answer.startswith("[SEP]"):
        return "Sorry, I couldn't find a clear answer in the provided FAQ context."

    return answer

# Simulated FAQ/context (replace with your real data source)
GOVT_FAQ_CONTEXT = (
    "The Department of Public Services (DPS) handles all permit applications. "
    "To apply for a building permit, you must submit Form 42B and pay a fee of $150. "
    "The processing time for new applications is typically 10 to 15 business days. "
    "Payments are accepted only via credit card or bank transfer, not cash. "
    "Office hours are Monday to Friday, 9:00 AM to 4:00 PM."
)

@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    answer = ""
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            answer = get_bert_answer(question, GOVT_FAQ_CONTEXT)
    return render_template("index.html", faq_context=GOVT_FAQ_CONTEXT, question=question, answer=answer)

# Ensure the app binds to the port provided by Render via $PORT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    # debug=False for production; threaded can help handle multiple requests in simple deployments
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
