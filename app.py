import os
import requests
from dotenv import load_dotenv
from flask import Flask, request, render_template, redirect, url_for, flash, send_file
import docx
import pdfplumber
import re
from werkzeug.utils import secure_filename
import json
import logging
from datetime import datetime
import io
import html

# -----------------------------
# Configuration and Setup
# -----------------------------

# Load environment variables from .env file
load_dotenv()

# Access the API key securely
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in environment variables. Please set it in the .env file.")

# Secret Key
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY not found in environment variables. Please set it in the .env file.")

# API Configuration
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "meta/llama-3.1-405b-instruct"  # Ensure this model is correct and accessible
ENDPOINT = f"{BASE_URL}/chat/completions"  # Verify this endpoint from NVIDIA's API documentation

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}

# Flask App Initialization
app = Flask(__name__)
app.secret_key = SECRET_KEY  # Use the secret key from environment variables
app.config['UPLOAD_FOLDER'] = 'uploads'  # Ensure this folder exists
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# Create uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Helper Functions
# -----------------------------

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_text(text):
    """Sanitize input text by escaping potentially harmful characters."""
    return html.escape(text)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {e}")
    return text

def extract_text_from_docx(docx_path):
    """Extract text from a Word (docx) file."""
    text = ""
    try:
        doc = docx.Document(docx_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        raise ValueError(f"Error reading Word file: {e}")
    return text

def get_text_from_file(file_path):
    """Determine file type and extract text accordingly."""
    _, file_ext = os.path.splitext(file_path)
    file_ext = file_ext.lower()
    if file_ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_ext in [".docx", ".doc"]:
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or Word document.")

def parse_analysis_results(message_content):
    """
    Parses the ATS Score, Job Chances, Missing Skills, Reasons, and Suggestions from the API response content.
    Expects a well-structured JSON response.
    """
    # Remove markdown code block if present
    message_content = message_content.strip()
    if message_content.startswith("```") and message_content.endswith("```"):
        message_content = message_content[3:-3].strip()

    logging.debug(f"Message content after cleaning: {message_content}")

    try:
        data = json.loads(message_content)
        logging.debug(f"Parsed JSON data: {data}")
        
        # Extract and validate fields
        ats_score = data.get('ats_score')
        job_chances = data.get('job_chances')
        missing_skills = data.get('missing_skills', [])
        ats_score_reason = data.get('ats_score_reason', 'No reason provided.')
        job_chances_reason = data.get('job_chances_reason', 'No reason provided.')
        suggestions = data.get('suggestions', [])

        # Validate numerical fields
        if not isinstance(ats_score, (int, float)):
            logging.error("ATS Score is not a number.")
            ats_score = None
        if not isinstance(job_chances, (int, float)):
            logging.error("Job Chances is not a number.")
            job_chances = None

        # Validate missing_skills and suggestions as lists
        if not isinstance(missing_skills, list):
            logging.error("Missing Skills is not a list.")
            missing_skills = []
        if not isinstance(suggestions, list):
            logging.error("Suggestions is not a list.")
            suggestions = []

        # Ensure reasons are strings
        ats_score_reason = str(ats_score_reason).strip()
        job_chances_reason = str(job_chances_reason).strip()

        logging.debug(f"Extracted ATS Score: {ats_score}")
        logging.debug(f"Extracted Job Chances: {job_chances}")
        logging.debug(f"Extracted Missing Skills: {missing_skills}")
        logging.debug(f"Extracted ATS Score Reason: {ats_score_reason}")
        logging.debug(f"Extracted Job Chances Reason: {job_chances_reason}")
        logging.debug(f"Extracted Suggestions: {suggestions}")

        return ats_score, job_chances, missing_skills, {
            "ats_score_reason": ats_score_reason,
            "job_chances_reason": job_chances_reason
        }, suggestions
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed: {e}")
        logging.error(f"Original message content: {message_content}")
        return None, None, None, None, "Invalid JSON response from API."
    except KeyError as e:
        logging.error(f"Missing key in JSON response: {e}")
        logging.error(f"Original JSON data: {data}")
        return None, None, None, None, f"Missing key in JSON response: {e}"
    except Exception as e:
        logging.error(f"Unexpected error during parsing: {e}")
        return None, None, None, None, f"Unexpected error during parsing: {e}"

def save_result_to_json(ats_score, job_chances, missing_skills, reasons, suggestions):
    """Append analysis result to a single JSON file."""
    # Define the directory and file path
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)  # Create directory if it doesn't exist
    file_path = os.path.join(data_dir, 'results.json')
    
    # Prepare the result data
    result = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',  # UTC timestamp
        'ats_score': ats_score,
        'job_chances': job_chances,
        'missing_skills': missing_skills,
        'reasons': reasons,
        'suggestions': suggestions
    }
    
    # Load existing data
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    # Append the new result
    data.append(result)
    
    # Write back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def save_result_to_separate_json(ats_score, job_chances, missing_skills, reasons, suggestions):
    """Save each analysis result as a separate JSON file."""
    # Define the directory and ensure it exists
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate a unique filename using UTC timestamp
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')  # Example: 20231009T123456Z
    filename = f'result_{timestamp}.json'
    file_path = os.path.join(data_dir, filename)
    
    # Prepare the result data
    result = {
        'timestamp': timestamp,
        'ats_score': ats_score,
        'job_chances': job_chances,
        'missing_skills': missing_skills,
        'reasons': reasons,
        'suggestions': suggestions
    }
    
    # Write the result to the JSON file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

def analyze_resume_job(resume_text, job_description):
    """Analyze the resume against the job description using NVIDIA's NIM API."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    combined_content = (
        "Analyze the following resume and job description in detail.\n\n"
        f"**Resume Skills:** {resume_text}\n\n"
        f"**Job Description Requirements:** {job_description}\n\n"
        "Provide the analysis with the following details in **ONLY** JSON format as demonstrated below. "
        "Ensure that the JSON is well-structured and includes all missing skills comprehensively. "
        "Provide **detailed**, **specific**, and **actionable** reasons for both the ATS Score and Job Chances. "
        "Avoid vague statements and ensure that each reason directly correlates with the resume and job description. "
        "Do not add any extra text or explanations outside the JSON.\n\n"
        "{\n"
        "  \"ats_score\": 85,\n"
        "  \"job_chances\": 75,\n"
        "  \"missing_skills\": [\"Java\", \"C#\"],\n"
        "  \"ats_score_reason\": \"Your resume lacks relevant keywords such as 'Agile Methodologies' and 'Full-Stack Development' which are emphasized in the job description.\",\n"
        "  \"job_chances_reason\": \"While your current skills are strong, acquiring proficiency in Python and gaining experience with cloud platforms like AWS could significantly enhance your job prospects.\",\n"
        "  \"suggestions\": [\n"
        "    \"Incorporate keywords like 'Agile Methodologies' and 'Full-Stack Development' into your resume to better match the job requirements.\",\n"
        "    \"Consider obtaining certifications in Python and AWS to increase your competitiveness for this role.\"\n"
        "  ]\n"
        "}"
    )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": combined_content
            }
        ],
        "temperature": 0.3,  # Slightly increased for more creativity
        "top_p": 0.9,         # Increased to allow for more diverse output
        "max_tokens": 2000,   # Further increased to accommodate detailed responses
        "stream": False
    }

    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        logging.debug(f"Response Status Code: {response.status_code}")
        logging.debug(f"Response Content: {response.text}")  # Log the raw response
        response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        logging.error(f"HTTP Error: {errh}")
        return None, None, None, None, f"HTTP Error: {errh}"
    except requests.exceptions.ConnectionError as errc:
        logging.error(f"Error Connecting: {errc}")
        return None, None, None, None, f"Error Connecting: {errc}"
    except requests.exceptions.Timeout as errt:
        logging.error(f"Timeout Error: {errt}")
        return None, None, None, None, f"Timeout Error: {errt}"
    except requests.exceptions.RequestException as err:
        logging.error(f"An error occurred: {err}")
        return None, None, None, None, f"An error occurred: {err}"

    try:
        response_json = response.json()
        message_content = response_json['choices'][0]['message']['content']
        logging.debug(f"Parsed message content: {message_content}")
        ats_score, job_chances, missing_skills, reasons, suggestions = parse_analysis_results(message_content)
        # Validate extracted data
        if ats_score is None or job_chances is None:
            logging.error("Essential fields are missing in the API response.")
            return None, None, None, None, "Essential fields are missing in the API response."
        return ats_score, job_chances, missing_skills, reasons, suggestions
    except (ValueError, KeyError) as e:
        logging.error(f"Failed to parse response: {e}")
        return None, None, None, None, f"Failed to parse response: {e}"

# -----------------------------
# Flask Routes
# -----------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        logging.debug("Received POST request.")
        # Check if the post request has the resume part
        if 'resume' not in request.files:
            logging.warning("No resume file part in the request.")
            flash('No resume file part')
            return redirect(request.url)
        resume_file = request.files['resume']
        if resume_file.filename == '':
            logging.warning("No selected resume file.")
            flash('No selected resume file')
            return redirect(request.url)
        if resume_file and allowed_file(resume_file.filename):
            resume_filename = secure_filename(resume_file.filename)
            resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_filename)
            logging.debug(f"Saving resume file to {resume_path}")
            resume_file.save(resume_path)
            # Extract resume text
            try:
                resume_text = get_text_from_file(resume_path)
                resume_text = sanitize_text(resume_text)
                logging.debug(f"Extracted resume text: {resume_text[:100]}...")  # Log first 100 chars
            except Exception as e:
                logging.error(f"Error extracting resume text: {e}")
                flash(str(e))
                return redirect(request.url)
        else:
            logging.warning("Unsupported resume file format.")
            flash('Unsupported resume file format. Please upload a PDF or Word document.')
            return redirect(request.url)

        # Check if the post request has the job description part
        if 'job_description_file' in request.files and request.files['job_description_file'].filename != '':
            job_desc_file = request.files['job_description_file']
            if job_desc_file and allowed_file(job_desc_file.filename):
                job_desc_filename = secure_filename(job_desc_file.filename)
                job_desc_path = os.path.join(app.config['UPLOAD_FOLDER'], job_desc_filename)
                logging.debug(f"Saving job description file to {job_desc_path}")
                job_desc_file.save(job_desc_path)
                # Extract job description text
                try:
                    job_desc_text = get_text_from_file(job_desc_path)
                    job_desc_text = sanitize_text(job_desc_text)
                    logging.debug(f"Extracted job description text: {job_desc_text[:100]}...")  # Log first 100 chars
                except Exception as e:
                    logging.error(f"Error extracting job description text: {e}")
                    flash(str(e))
                    return redirect(request.url)
            else:
                logging.warning("Unsupported job description file format.")
                flash('Unsupported job description file format. Please upload a PDF or Word document.')
                return redirect(request.url)
        else:
            # If no file uploaded, fallback to textarea input
            job_desc_text = request.form.get('job_description', '').strip()
            job_desc_text = sanitize_text(job_desc_text)
            logging.debug(f"Job description received from textarea: {job_desc_text[:100]}...")  # Log first 100 chars
            if not job_desc_text:
                logging.warning("Job description is empty.")
                flash('Job description cannot be empty.')
                return redirect(request.url)

        # Analyze resume against job description
        ats_score, job_chances, missing_skills, reasons, suggestions = analyze_resume_job(resume_text, job_desc_text)
        logging.debug(f"Analysis Results - ATS Score: {ats_score}, Job Chances: {job_chances}, Missing Skills: {missing_skills}, Reasons: {reasons}, Suggestions: {suggestions}")
        if ats_score is None:
            logging.error(f"Analysis Failed: {job_chances}")
            flash(f"Analysis Failed: {job_chances}")
            return redirect(request.url)

        # Remove the uploaded files after processing
        try:
            os.remove(resume_path)
            logging.debug(f"Removed uploaded resume file {resume_path}")
            if 'job_description_file' in locals():
                os.remove(job_desc_path)
                logging.debug(f"Removed uploaded job description file {job_desc_path}")
        except Exception as e:
            logging.warning(f"Could not remove uploaded file: {e}")

        # Save the result to JSON files
        try:
            # Approach 1: Append to a single JSON file
            save_result_to_json(ats_score, job_chances, missing_skills, reasons, suggestions)
            
            # Approach 2: Save as a separate JSON file
            save_result_to_separate_json(ats_score, job_chances, missing_skills, reasons, suggestions)
            
            logging.debug("Saved analysis results to JSON files.")
        except Exception as e:
            logging.error(f"Failed to save analysis results: {e}")
            flash(f"Failed to save analysis results: {e}")
            return redirect(request.url)

        # Render results
        return render_template(
            'result.html',
            ats_score=ats_score,
            job_chances=job_chances,
            missing_skills=missing_skills,
            reasons=reasons,
            suggestions=suggestions
        )
    logging.debug("Rendering index.html")
    return render_template('index.html')

@app.route('/download')
def download_analysis():
    """Download the latest analysis result as a JSON file."""
    # Assuming the latest result is the last item in 'results.json'
    try:
        file_path = os.path.join('data', 'results.json')
        if not os.path.exists(file_path):
            flash("No analysis results available for download.")
            return redirect(url_for('index'))
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not data:
                flash("No analysis results available for download.")
                return redirect(url_for('index'))
            latest_result = data[-1]
            # Convert JSON to bytes
            json_bytes = json.dumps(latest_result, indent=4).encode('utf-8')
            # Create a BytesIO object
            return send_file(
                io.BytesIO(json_bytes),
                mimetype='application/json',
                as_attachment=True,
                download_name='analysis_result.json'
            )
    except Exception as e:
        logging.error(f"Error downloading analysis result: {e}")
        flash(f"Error downloading analysis result: {e}")
        return redirect(url_for('index'))

# -----------------------------
# Run the Flask App
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)
