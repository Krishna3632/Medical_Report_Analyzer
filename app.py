import os
import pdfplumber
import uuid
import io
import re
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
import asyncio
from connection import fetch_doctors_with_user
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
ALLOWED_EXTENSIONS = {'pdf'}
SESSION_TIMEOUT_MINUTES = 30

# In-memory session storage (for serverless deployment)
session_data = {}

# Sample doctors database - Replace with your actual database
DOCTORS_DATABASE = {
    'cardiology': [
        {'name': 'Dr. Sarah Johnson', 'specialization': 'Cardiology', 'location': 'New York', 'rating': 4.8},
        {'name': 'Dr. Michael Chen', 'specialization': 'Cardiology', 'location': 'California', 'rating': 4.9},
    ],
    'neurology': [
        {'name': 'Dr. Emma Davis', 'specialization': 'Neurology', 'location': 'Texas', 'rating': 4.7},
        {'name': 'Dr. James Wilson', 'specialization': 'Neurology', 'location': 'Florida', 'rating': 4.6},
    ],
    'gastroenterology': [
        {'name': 'Dr. Lisa Rodriguez', 'specialization': 'Gastroenterology', 'location': 'Chicago', 'rating': 4.8},
        {'name': 'Dr. David Kim', 'specialization': 'Gastroenterology', 'location': 'Seattle', 'rating': 4.7},
    ],
    'dermatology': [
        {'name': 'Dr. Amanda Taylor', 'specialization': 'Dermatology', 'location': 'Miami', 'rating': 4.9},
        {'name': 'Dr. Robert Brown', 'specialization': 'Dermatology', 'location': 'Denver', 'rating': 4.6},
    ],
    'orthopedics': [
        {'name': 'Dr. Kevin Martinez', 'specialization': 'Orthopedics', 'location': 'Phoenix', 'rating': 4.8},
        {'name': 'Dr. Jennifer Lee', 'specialization': 'Orthopedics', 'location': 'Boston', 'rating': 4.7},
    ],
    'general': [
        {'name': 'Dr. Mary Anderson', 'specialization': 'General Medicine', 'location': 'Atlanta', 'rating': 4.5},
        {'name': 'Dr. Thomas Garcia', 'specialization': 'General Medicine', 'location': 'Portland', 'rating': 4.6},
    ]
}

# Symptom-to-specialization mapping
SYMPTOM_SPECIALIZATION_MAP = {
    # Cardiology
    'chest pain': 'Cardiology',
    'heart palpitations': 'Cardiology',
    'shortness of breath': 'Cardiology',
    'high blood pressure': 'Cardiology',
    'low blood pressure': 'Cardiology',
    'irregular heartbeat': 'Cardiology',
    'swelling in legs': 'Cardiology',

    # Neurology
    'headache': 'Neurology',
    'dizziness': 'Neurology',
    'seizure': 'Neurology',
    'memory loss': 'Neurology',
    'migraine': 'Neurology',
    'numbness': 'Neurology',
    'tingling sensation': 'Neurology',
    'difficulty speaking': 'Neurology',
    'vision problems': 'Neurology',
    'tremors': 'Neurology',

    # Gastroenterology
    'stomach pain': 'Gastroenterology',
    'nausea': 'Gastroenterology',
    'vomiting': 'Gastroenterology',
    'diarrhea': 'Gastroenterology',
    'constipation': 'Gastroenterology',
    'bloating': 'Gastroenterology',
    'acid reflux': 'Gastroenterology',
    'blood in stool': 'Gastroenterology',
    'loss of appetite': 'Gastroenterology',

    # Dermatology
    'skin rash': 'Dermatology',
    'acne': 'Dermatology',
    'itching': 'Dermatology',
    'skin lesion': 'Dermatology',
    'dry skin': 'Dermatology',
    'eczema': 'Dermatology',
    'psoriasis': 'Dermatology',
    'hair loss': 'Dermatology',
    'nail discoloration': 'Dermatology',

    # Orthopedics
    'joint pain': 'Orthopedics',
    'back pain': 'Orthopedics',
    'fracture': 'Orthopedics',
    'muscle pain': 'Orthopedics',
    'knee pain': 'Orthopedics',
    'shoulder pain': 'Orthopedics',
    'swollen joints': 'Orthopedics',
    'difficulty walking': 'Orthopedics',

    # Pulmonology
    'chronic cough': 'Pulmonology',
    'wheezing': 'Pulmonology',
    'difficulty breathing': 'Pulmonology',
    'cough with blood': 'Pulmonology',
    'asthma symptoms': 'Pulmonology',

    # Endocrinology
    'fatigue': 'Endocrinology',
    'weight gain': 'Endocrinology',
    'weight loss': 'Endocrinology',
    'excessive thirst': 'Endocrinology',
    'frequent urination': 'Endocrinology',
    'sweating': 'Endocrinology',
    'heat intolerance': 'Endocrinology',

    # Psychiatry
    'depression': 'Psychiatry',
    'anxiety': 'Psychiatry',
    'insomnia': 'Psychiatry',
    'mood swings': 'Psychiatry',
    'hallucinations': 'Psychiatry',
    'panic attacks': 'Psychiatry',

    # Ophthalmology
    'blurred vision': 'Ophthalmology',
    'eye pain': 'Ophthalmology',
    'red eyes': 'Ophthalmology',
    'watery eyes': 'Ophthalmology',
    'double vision': 'Ophthalmology',

    # ENT
    'ear pain': 'ENT',
    'hearing loss': 'ENT',
    'sore throat': 'ENT',
    'nasal congestion': 'ENT',
    'runny nose': 'ENT',
    'loss of smell': 'ENT',

    # Urology
    'painful urination': 'Urology',
    'blood in urine': 'Urology',
    'urinary incontinence': 'Urology',
    'frequent urination': 'Urology',
    'difficulty urinating': 'Urology',

    # Gynecology
    'irregular periods': 'Gynecology',
    'pelvic pain': 'Gynecology',
    'vaginal discharge': 'Gynecology',
    'pain during intercourse': 'Gynecology',
    'pregnancy-related issues': 'Gynecology',

    # Pediatrics
    'crying excessively': 'Pediatrics',
    'fever in child': 'Pediatrics',
    'rashes in child': 'Pediatrics',
    'delayed milestones': 'Pediatrics',
}



# Symptom keywords for detection
SYMPTOM_KEYWORDS = [
    # General symptom verbs and descriptors
    'pain', 'ache', 'hurt', 'sore', 'tender', 'burning', 'stabbing', 'throbbing',
    'cramping', 'sharp pain', 'dull pain', 'pressure', 'tingling', 'numbness',
    'swelling', 'inflammation', 'discomfort', 'bruising', 'sensitivity',

    # Head & neurological
    'headache', 'migraine', 'dizziness', 'lightheadedness', 'vertigo',
    'seizure', 'tremors', 'memory loss', 'confusion', 'blurred vision',
    'double vision', 'difficulty speaking', 'loss of balance', 'fainting',

    # Digestive
    'nausea', 'vomiting', 'stomach pain', 'abdominal pain', 'bloating',
    'diarrhea', 'constipation', 'gas', 'indigestion', 'acid reflux',
    'loss of appetite', 'heartburn', 'cramps', 'blood in stool',

    # Respiratory
    'cough', 'sneeze', 'runny nose', 'nasal congestion', 'shortness of breath',
    'wheezing', 'chest pain', 'tight chest', 'difficulty breathing',
    'coughing up blood', 'phlegm', 'hoarseness',

    # Cardiovascular
    'heart palpitations', 'irregular heartbeat', 'fast heartbeat', 'slow heartbeat',
    'chest tightness', 'high blood pressure', 'low blood pressure',
    'swelling in legs', 'cold hands and feet',

    # Skin & allergy
    'rash', 'itching', 'dry skin', 'flaky skin', 'redness', 'hives',
    'skin peeling', 'eczema', 'acne', 'psoriasis', 'skin lesion',
    'discoloration', 'lumps', 'boils', 'blisters',

    # Musculoskeletal
    'joint pain', 'back pain', 'neck pain', 'muscle pain', 'stiffness',
    'limited mobility', 'cramps', 'spasms', 'bone pain', 'fracture',

    # Mental health
    'fatigue', 'tired', 'weakness', 'exhaustion', 'anxiety', 'depression',
    'stress', 'mood swings', 'irritability', 'insomnia', 'sleep problems',
    'hallucinations', 'panic attacks', 'feeling down', 'low energy',

    # Urology & genital
    'frequent urination', 'painful urination', 'burning while urinating',
    'blood in urine', 'urinary incontinence', 'vaginal discharge',
    'irregular periods', 'pelvic pain', 'pain during intercourse',

    # Eye & ENT
    'eye pain', 'red eyes', 'watery eyes', 'blurred vision', 'itchy eyes',
    'ear pain', 'hearing loss', 'ringing in ears', 'sore throat',
    'difficulty swallowing', 'loss of smell', 'loss of taste',

    # Infection & immune response
    'fever', 'chills', 'night sweats', 'body aches', 'swollen glands',

    # Common phrases for NLP
    'experiencing', 'feeling', 'having', 'suffering from', 'showing symptoms of',
    'dealing with', 'complaining of', 'symptoms', 'signs', 'issues',
    'problem', 'trouble', 'unwell', 'sick', 'not feeling well'
]

# Initialize AI Agents
lab_agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    tools=[DuckDuckGo()],
    description="Medical assistant for lab report analysis",
    instructions=[
        "Explain lab results in simple language",
        "Mention normal ranges for lab values",
        "Be educational and reassuring",
        "Always recommend consulting healthcare professionals",
        "Never provide specific diagnoses or treatments",
        "Use bullet points for clarity"
    ],
    markdown=True
)

general_agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    tools=[DuckDuckGo()],
    description="Medical assistant for general health questions",
    instructions=[
        "Provide educational health information",
        "Explain medical concepts clearly",
        "Be reassuring and informative",
        "Always recommend professional consultation",
        "Stay within educational bounds",
        "Use clear formatting"
    ],
    markdown=True
)

symptoms_agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    tools=[DuckDuckGo()],
    description="Medical assistant for symptom analysis",
    instructions=[
        "Analyze symptoms educationally",
        "Suggest possible causes without diagnosing",
        "Recommend when to seek immediate care",
        "Provide general self-care tips",
        "Always emphasize professional evaluation",
        "Include medical disclaimers"
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

def detect_query_type(query):
    """Detect what type of query this is based on content"""
    query_lower = query.lower()
    
    # Check for file upload requests
    upload_keywords = ['upload', 'pdf', 'report', 'lab report', 'test results', 'file']
    if any(keyword in query_lower for keyword in upload_keywords):
        return 'upload_request'
    
    # Check for symptom descriptions
    symptom_count = sum(1 for keyword in SYMPTOM_KEYWORDS if keyword in query_lower)
    
    # Symptom patterns
    symptom_patterns = [
        r'i have|i am having|i feel|i am feeling|experiencing',
        r'my \w+ (hurt|pain|ache|sore)',
        r'(pain|ache|hurt) in my',
        r'i have been (sick|unwell|feeling)',
        r'symptoms include|symptoms are',
        r'for \d+ days?|for \d+ weeks?|since yesterday|since last week'
    ]
    
    pattern_matches = sum(1 for pattern in symptom_patterns if re.search(pattern, query_lower))
    
    # If high symptom keyword count or pattern matches, likely symptoms
    if symptom_count >= 2 or pattern_matches >= 1:
        return 'symptoms'
    
    # Default to general medical question
    return 'general'

def analyze_symptoms_for_specialization(symptoms_text):
    """Analyze symptoms to determine appropriate medical specialization"""
    symptoms_lower = symptoms_text.lower()
    
    # Count matches for each specialization
    specialization_scores = {}
    
    for symptom, specialization in SYMPTOM_SPECIALIZATION_MAP.items():
        if symptom in symptoms_lower:
            if specialization not in specialization_scores:
                specialization_scores[specialization] = 0
            specialization_scores[specialization] += 1
    
    # Return the specialization with highest score, or general if no matches
    if specialization_scores:
        return max(specialization_scores, key=specialization_scores.get)
    else:
        return 'general'

def get_doctors_by_specialization(specialization):
    data = asyncio.run(fetch_doctors_with_user())
    print(data)
    ans = []

    print(f"Looking for specialization: {specialization.lower()}")

    for doc in data:
        specializations = doc.get('Specialization', [])
        print(f"Doctor: {doc['name']} - Specializations: {specializations}")

        # Lowercase conversion for matching
        specializations_lower = [spec.lower() for spec in specializations]
        print(f"Converted Specializations: {specializations_lower}")

        if specialization.lower() in specializations_lower:
            print(f"Matched Doctor: {doc['name']}")
            ans.append({
                "Doctor Name": doc["name"],
                "Rating": doc["rating"]
            })

    print("Final matched doctors:", ans)
    return ans

#Storing data(work for Ankush)
def store_data_for_past_reports(request_type,data,any_file):
    # Store data for past reports
    return ""

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF upload"""
    try:
        cleanup_expired_sessions()
        
        if 'pdf' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['pdf']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Only PDF files are allowed"}), 400
        
        try:
            file_content = file.read()
            if not file_content:
                return jsonify({"error": "File is empty"}), 400
        except Exception as e:
            return jsonify({"error": f"Failed to read file: {str(e)}"}), 400
        
        try:
            text = extract_text_from_pdf_bytes(file_content)
        except Exception as e:
            return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 400
        
        if not text:
            return jsonify({"error": "No text could be extracted from the PDF"}), 400
        
        session_id = str(uuid.uuid4())
        
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

@app.route('/smart_query', methods=['POST'])
def smart_query():
    """Handle smart query that determines the type automatically"""
    try:
        cleanup_expired_sessions()
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        query = data.get("query", "").strip()
        session_id = data.get("session_id", "").strip()
        
        if not query:
            return jsonify({"error": "No question provided"}), 400
        
        # Detect query type
        query_type = detect_query_type(query)
        
        # Handle upload request
        if query_type == 'upload_request':
            return jsonify({
                "response": "To upload a lab report, please use the upload button above or drag and drop a PDF file. I'll be able to analyze your lab results once you upload the file.",
                "query_type": "upload_request",
                "action_needed": "upload_file"
            })
        
        # Handle lab report questions if session exists
        if session_id and session_id in session_data:
            session_info = session_data[session_id]
            session_info['timestamp'] = datetime.now()
            
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
            
            try:
                response = ""
                for chunk in lab_agent.run(full_prompt, stream=True):
                    response += chunk.content
                
                if not response.strip():
                    response = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
            except Exception as e:
                print(f"Agent error: {e}")
                response = "I'm sorry, but I encountered an error while processing your question. Please try again."
            
            return jsonify({
                "response": response,
                "query_type": "lab_report",
                "filename": session_info['filename']
            })
        
        # Handle symptoms
        elif query_type == 'symptoms':
            # Analyze symptoms to determine specialization
            specialization = analyze_symptoms_for_specialization(query)
            
            # Get doctors for the specialization
            recommended_doctors = get_doctors_by_specialization(specialization)
            
            # Prepare prompt for symptoms analysis
            full_prompt = f"""User is experiencing these symptoms: {query}

Please provide educational information about these symptoms. Remember to:
- Explain what these symptoms might indicate in general terms
- Mention possible common causes (without diagnosing)
- Suggest when to seek immediate medical attention (red flags)
- Provide general self-care tips where appropriate
- Always emphasize that symptoms require professional medical evaluation
- Never provide specific diagnoses or treatment recommendations
- Be reassuring while being informative
- Use bullet points and clear formatting for better readability
- Include disclaimer about not replacing professional medical advice"""
            
            try:
                response = ""
                for chunk in symptoms_agent.run(full_prompt, stream=True):
                    response += chunk.content
                
                if not response.strip():
                    response = "I apologize, but I couldn't generate a response. Please try describing your symptoms differently."
            
            except Exception as e:
                print(f"Symptoms agent error: {e}")
                response = "I'm sorry, but I encountered an error while analyzing your symptoms. Please try again."
            
            return jsonify({
                "response": response,
                "query_type": "symptoms",
                "symptoms": query,
                "specialization": specialization,
                "recommended_doctors": recommended_doctors[:1]  # Return top 3 doctors
            })
        
        # Handle general questions
        else:
            full_prompt = f"""User's general medical question: {query}

Please provide helpful, educational information about this health topic. Remember to:
- Explain medical terms and concepts in simple language
- Provide accurate, general health information
- Be reassuring and educational
- Always emphasize the importance of consulting healthcare professionals
- Never provide specific diagnoses, prescriptions, or critical medical decisions
- Stay within the bounds of general health education
- Use bullet points and clear formatting for better readability"""
            
            try:
                response = ""
                for chunk in general_agent.run(full_prompt, stream=True):
                    response += chunk.content
                
                if not response.strip():
                    response = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
            except Exception as e:
                print(f"General agent error: {e}")
                response = "I'm sorry, but I encountered an error while processing your question. Please try again."
            
            return jsonify({
                "response": response,
                "query_type": "general"
            })
    
    except Exception as e:
        return jsonify({"error": f"Failed to process question: {str(e)}"}), 500

# Keep existing endpoints for backward compatibility
@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle question about uploaded report"""
    try:
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
        
        if session_id not in session_data:
            return jsonify({"error": "Session not found or expired. Please upload your PDF again."}), 400
        
        session_info = session_data[session_id]
        session_info['timestamp'] = datetime.now()
        
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
        
        try:
            response = ""
            for chunk in lab_agent.run(full_prompt, stream=True):
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

@app.route('/ask_general', methods=['POST'])
def ask_general_question():
    """Handle general medical questions"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        query = data.get("query", "").strip()
        
        if not query:
            return jsonify({"error": "No question provided"}), 400
        
        full_prompt = f"""User's general medical question: {query}

Please provide helpful, educational information about this health topic. Remember to:
- Explain medical terms and concepts in simple language
- Provide accurate, general health information
- Be reassuring and educational
- Always emphasize the importance of consulting healthcare professionals
- Never provide specific diagnoses, prescriptions, or critical medical decisions
- Stay within the bounds of general health education
- Use bullet points and clear formatting for better readability"""
        
        try:
            response = ""
            for chunk in general_agent.run(full_prompt, stream=True):
                response += chunk.content
            
            if not response.strip():
                response = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        
        except Exception as e:
            print(f"General agent error: {e}")
            response = "I'm sorry, but I encountered an error while processing your question. Please try again."
        
        return jsonify({
            "response": response
        })
    
    except Exception as e:
        return jsonify({"error": f"Failed to process question: {str(e)}"}), 500

@app.route('/symptoms', methods=['POST'])
def analyze_symptoms():
    """Handle symptoms analysis and doctor recommendations"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        symptoms = data.get("symptoms", "").strip()
        
        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400
        
        # Analyze symptoms to determine specialization
        specialization = analyze_symptoms_for_specialization(symptoms)
        
        # Get doctors for the specialization
        recommended_doctors = get_doctors_by_specialization(specialization)
        
        # Prepare prompt for symptoms analysis
        full_prompt = f"""User is experiencing these symptoms: {symptoms}

Please provide educational information about these symptoms. Remember to:
- Explain what these symptoms might indicate in general terms
- Mention possible common causes (without diagnosing)
- Suggest when to seek immediate medical attention (red flags)
- Provide general self-care tips where appropriate
- Always emphasize that symptoms require professional medical evaluation
- Never provide specific diagnoses or treatment recommendations
- Be reassuring while being informative
- Use bullet points and clear formatting for better readability
- Include disclaimer about not replacing professional medical advice"""
        
        try:
            response = ""
            for chunk in symptoms_agent.run(full_prompt, stream=True):
                response += chunk.content
            
            if not response.strip():
                response = "I apologize, but I couldn't generate a response. Please try describing your symptoms differently."
        
        except Exception as e:
            print(f"Symptoms agent error: {e}")
            response = "I'm sorry, but I encountered an error while analyzing your symptoms. Please try again."
        
        return jsonify({
            "response": response,
            "symptoms": symptoms,
            "specialization": specialization,
            "recommended_doctors": recommended_doctors[:1]  # Return top 3 doctors
        })
    
    except Exception as e:
        return jsonify({"error": f"Failed to analyze symptoms: {str(e)}"}), 500

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

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🏥 Starting Enhanced Medical Assistant Server...")
    print(f"📁 Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024)}MB")
    print(f"⏰ Session timeout: {SESSION_TIMEOUT_MINUTES} minutes")
    print("🔗 Available endpoints:")
    print("   - POST /upload - Upload PDF lab reports")
    print("   - POST /smart_query - Smart query handler (auto-detects intent)")
    print("   - POST /ask - Ask questions about uploaded reports")
    print("   - POST /ask_general - Ask general medical questions")
    print("   - POST /symptoms - Analyze symptoms and get doctor recommendations")
    print("   - GET /health - Health check")
    print("🚀 Server starting on http://localhost:5000")

    app.run(debug=True, host='0.0.0.0', port=5000)
    # app.run(debug=True, host='0.0.0.0', port=5000)