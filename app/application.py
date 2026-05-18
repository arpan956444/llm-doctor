import os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask,render_template,request,redirect,session,url_for, jsonify
from app.components.retriever import create_qa_chain
from app.config.config import MODEL_COMBINATIONS
import webbrowser
from threading import Timer
app=Flask(__name__)
app.secret_key= os.urandom(24)  #for handel session expair...24 min
from markupsafe import Markup #it load html safely

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

    if not user_input:
        return jsonify({"error": "No prompt provided"}), 400

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
        
        result = response.get("answer", "I couldn't find an answer in the provided documents.")
        
        messages.append({"role": "assistant", "content": result})
        session["messages"] = messages
        
        return jsonify({
            "role": "assistant",
            "content": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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