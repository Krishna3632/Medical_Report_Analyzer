import os
import pdfplumber
import uuid
import io
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
ALLOWED_EXTENSIONS = {'pdf'}
SESSION_TIMEOUT_MINUTES = 30

# In-memory session storage (for serverless deployment)
session_data = {}

# Initialize AI Agent
agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    tools=[DuckDuckGo()],
    description="You are a helpful medical assistant that explains lab test reports and terms in simple language. Never give diagnoses or medication advice.",
    instructions=[
        "When a user asks about a medical term, explain it simply.",
        "If lab values are given (e.g., Hemoglobin 9.2 g/dL), explain what the normal range is and if it's high/low.",
        "Be reassuring and educational. Suggest consulting a doctor for accurate health evaluation.",
        "Avoid giving prescriptions or critical decisions.",
        "If the user pastes an entire report, break down each test and explain its meaning and status.",
        "Always explain in a simple language.",
        "Use bullet points and clear formatting for better readability."
    ],
    markdown=True
)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf_bytes(pdf_bytes):
    """Extract text from PDF bytes using pdfplumber"""
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def cleanup_expired_sessions():
    """Remove expired sessions from memory"""
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, data in session_data.items():
        if current_time - data['timestamp'] > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del session_data[session_id]
    
    if expired_sessions:
        print(f"Cleaned up {len(expired_sessions)} expired sessions")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF upload"""
    try:
        # Clean up expired sessions before processing
        cleanup_expired_sessions()
        
        # Check if file is present
        if 'pdf' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['pdf']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({"error": "Only PDF files are allowed"}), 400
        
        # Read file content into memory
        try:
            file_content = file.read()
            if not file_content:
                return jsonify({"error": "File is empty"}), 400
        except Exception as e:
            return jsonify({"error": f"Failed to read file: {str(e)}"}), 400
        
        # Extract text from PDF bytes
        try:
            text = extract_text_from_pdf_bytes(file_content)
        except Exception as e:
            return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 400
        
        if not text:
            return jsonify({"error": "No text could be extracted from the PDF"}), 400
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Store session data in memory
        session_data[session_id] = {
            'text': text,
            'filename': secure_filename(file.filename),
            'timestamp': datetime.now()
        }
        
        return jsonify({
            "message": "PDF uploaded and processed successfully",
            "session_id": session_id,
            "filename": secure_filename(file.filename),
            "text_length": len(text)
        })
    
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle question about uploaded report"""
    try:
        # Clean up expired sessions
        cleanup_expired_sessions()
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        query = data.get("query", "").strip()
        session_id = data.get("session_id", "").strip()
        
        if not query:
            return jsonify({"error": "No question provided"}), 400
        
        if not session_id:
            return jsonify({"error": "No session ID provided"}), 400
        
        # Check if session exists
        if session_id not in session_data:
            return jsonify({"error": "Session not found or expired. Please upload your PDF again."}), 400
        
        session_info = session_data[session_id]
        # Update timestamp
        session_info['timestamp'] = datetime.now()
        
        # Prepare context for AI agent
        context = session_info['text']
        full_prompt = f"""Here is a medical lab report:

{context}

User's question: {query}

Please analyze this lab report and answer the user's question. Remember to:
- Explain medical terms in simple language
- Mention normal ranges when discussing lab values
- Be reassuring and educational
- Always recommend consulting with a healthcare provider
- Never provide specific medical diagnoses or treatment recommendations"""
        
        # Get response from AI agent
        try:
            response = ""
            for chunk in agent.run(full_prompt, stream=True):
                response += chunk.content
            
            if not response.strip():
                response = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        
        except Exception as e:
            print(f"Agent error: {e}")
            response = "I'm sorry, but I encountered an error while processing your question. Please try again."
        
        return jsonify({
            "response": response,
            "filename": session_info['filename']
        })
    
    except Exception as e:
        return jsonify({"error": f"Failed to process question: {str(e)}"}), 500

@app.route('/session/<session_id>')
def get_session_info(session_id):
    """Get session information"""
    cleanup_expired_sessions()
    
    if session_id not in session_data:
        return jsonify({"error": "Session not found"}), 404
    
    session_info = session_data[session_id]
    return jsonify({
        "filename": session_info['filename'],
        "text_length": len(session_info['text']),
        "upload_time": session_info['timestamp'].isoformat()
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    cleanup_expired_sessions()
    
    return jsonify({
        "status": "healthy",
        "active_sessions": len(session_data),
        "timestamp": datetime.now().isoformat()
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

# For Vercel deployment
if __name__ == '__main__':
    print("Starting Medical Lab Report Analyzer...")
    print(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024)}MB")
    print(f"Session timeout: {SESSION_TIMEOUT_MINUTES} minutes")
    app.run(debug=True, host='0.0.0.0', port=5000)