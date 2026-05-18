from flask import Flask,render_template,request,redirect,session,url_for, jsonify
from app.components.retriever import create_qa_chain
from dotenv import load_dotenv
import os
load_dotenv()
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
    return render_template("index.html", messages=session.get("messages", []))

@app.route('/chat', methods=['POST'])
def chat():
    if "messages" not in session:
        session["messages"] = []
    
    data = request.get_json()
    user_input = data.get("prompt") if data else None

    if not user_input:
        return jsonify({"error": "No prompt provided"}), 400

    messages = session["messages"]
    messages.append({"role": "user", "content": user_input})
    session["messages"] = messages
    
    try:
        qa_chain = create_qa_chain()
        if qa_chain is None:
            raise Exception("System Error: QA chain could not be initialized.")
        
        response = qa_chain.invoke({"query": user_input})
        result = response.get("result", "I couldn't find an answer in the provided documents.")
        
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
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )