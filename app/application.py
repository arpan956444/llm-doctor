import os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask,render_template,request,redirect,session,url_for, jsonify
from app.components.retriever import create_qa_chain
from app.config.config import MODEL_COMBINATIONS
import webbrowser
from threading import Timer
from app.common.logger import get_logger # Import the logger
app=Flask(__name__)
app.secret_key= os.urandom(24)  #for handel session expair...24 min
from markupsafe import Markup #it load html safely

logger = get_logger(__name__) # Initialize logger

def nl2br(value):
    return Markup(value.replace('\n','<br>\n'))

app.jinja_env.filters['nl2br'] = nl2br

@app.route('/')
def index():
    if "messages" not in session:
        session["messages"] = []
    return render_template(
        "index.html", 
        messages=session.get("messages", []),
        configs=list(MODEL_COMBINATIONS.keys())
    )

@app.route('/chat', methods=['POST'])
def chat():
    if "messages" not in session:
        session["messages"] = []
    
    data = request.get_json()
    user_input = data.get("prompt") if data else None
    config_name = data.get("config", "Setup_Default") if data else "Setup_Default"
    answer_style = data.get("answer_style", "concise") # NEW: Get answer style from UI

    if not user_input:
        return jsonify({"error": "No prompt provided"}), 400

    # Retrieve the configuration for the current request
    config = MODEL_COMBINATIONS.get(config_name, {})

    messages = session["messages"]
    messages.append({"role": "user", "content": user_input})
    session["messages"] = messages
    
    try:
        qa_chain = create_qa_chain(config_name=config_name)
        if qa_chain is None:
            raise Exception("System Error: QA chain could not be initialized.")
        
        # Convert session messages to chat_history format for ConversationalRetrievalChain
        # This takes pairs of (user_msg, assistant_msg) excluding the current prompt
        chat_history = []
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages):
                chat_history.append((messages[i]["content"], messages[i+1]["content"]))

        response = qa_chain.invoke({
            "question": user_input,
            "chat_history": chat_history
        })
        
        result = response.get("answer", "I couldn't find a definitive answer in the provided medical documents for this question. Please provide more context or rephrase your query.")
        source_docs = response.get("source_documents", [])
        
        # Calculate average relevance score from source documents, if reranking was used
        confidence_score = None
        if source_docs and config.get("rerank"): # Use the 'config' variable that's now defined
            scores = [doc.metadata.get('relevance_score') for doc in source_docs if 'relevance_score' in doc.metadata]
            if scores:
                confidence_score = sum(scores) / len(scores)

        assistant_message_content = result
        if confidence_score is not None:
            # Append confidence score to the assistant's message, e.g., for UI display
            assistant_message_content += f"\n\n(Confidence: {confidence_score:.2f})"
            if confidence_score < 0.5: # Example threshold for low confidence
                assistant_message_content += "\n<small class='text-warning'>**Note:** This answer is based on potentially weaker evidence from the documents.</small>"


        messages.append({"role": "assistant", "content": assistant_message_content})
        session["messages"] = messages
        
        return jsonify({
            "role": "assistant",
            "content": assistant_message_content,
            "confidence_score": confidence_score
        })
    except Exception as e:
        logger.error(f"Error in chat processing: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred. Please try again."}), 500

@app.route('/clear')
def clear():
    session.pop("messages",None)

    return redirect(url_for("index"))

if __name__ == '__main__':
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000")

    # Start a timer to open the browser after 1.5 seconds to ensure the server is up
    Timer(1.5, open_browser).start()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )