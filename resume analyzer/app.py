from flask import Flask, render_template, request, redirect, url_for, session, abort, flash  # Added flash here
import os
import docx2txt
import fitz  # PyMuPDF
import re
from werkzeug.utils import secure_filename
import nltk
from nltk.corpus import stopwords
from collections import Counter
import sqlite3
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your_super_secret_key' 

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'} 

# Database setup
DATABASE = 'resume_analyzer.db'

def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                ats_score REAL,
                upload_date TEXT,
                contact_info TEXT
            )
        ''')
        db.execute('''
            CREATE TABLE IF NOT EXISTS job_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_filename TEXT,
                jd_filename TEXT,
                match_score REAL,
                analysis_date TEXT
            )
        ''')
        db.execute('''
            CREATE TABLE IF NOT EXISTS app_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resumes_uploaded INTEGER DEFAULT 0,
                jds_uploaded INTEGER DEFAULT 0,
                ats_checks INTEGER DEFAULT 0,
                match_checks INTEGER DEFAULT 0
            )
        ''')
        db.commit()

init_db()

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# ================== UTILITY FUNCTIONS ==================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text(file):
    try:
        if file.filename.endswith('.pdf'):
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                return ''.join([page.get_text() for page in doc])
        elif file.filename.endswith('.docx'):
            return docx2txt.process(file)
        return ""
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return ""

# ================== ATS SCORING FUNCTIONS ==================
ATS_PARAMETERS = {
    'keyword_density': 0.3,
    'action_verbs': 0.15,
    'quantifiable_results': 0.2,
    'structure': 0.15,
    'contact_info': 0.1,
    'length': 0.1
}

ACTION_VERBS = [
    'achieved', 'managed', 'developed', 'led', 'implemented',
    'increased', 'reduced', 'optimized', 'created', 'designed'
]

def calculate_keyword_density(text):
    words = [word.lower() for word in re.findall(r'\w+', text) if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    return min(len(word_counts) / 50, 1)

def count_action_verbs(text):
    verbs_found = [verb for verb in ACTION_VERBS if verb in text.lower()]
    return min(len(verbs_found) / 5, 1)

def check_quantifiable_results(text):
    quantifiers = re.findall(r'\b(\d+%|\d+\+|\$?\d+\s?[km]?|\d+\s?(years|months))\b', text, re.I)
    return min(len(quantifiers) / 3, 1)

def evaluate_structure(text):
    structure_score = 0
    sections = ['experience', 'education', 'skills', 'summary', 'projects']
    for section in sections:
        if re.search(rf'\b{section}\b', text, re.I):
            structure_score += 0.2
    if re.search(r'(\d{4})\s*-\s*(present|\d{4})', text):
        structure_score += 0.2
    return min(structure_score, 1)

def check_contact_info(text):
    contact_items = 0
    if re.search(r'[\w\.-]+@[\w\.-]+', text): contact_items += 1
    if re.search(r'\+?\d[\d -]{8,}\d', text): contact_items += 1
    if re.search(r'linkedin\.com|github\.com', text): contact_items += 1
    return contact_items / 3

def check_length(text):
    word_count = len(re.findall(r'\w+', text))
    if 400 <= word_count <= 800: return 1
    elif 300 <= word_count < 400 or 800 < word_count <= 1000: return 0.5
    return 0

def calculate_ats_score(text):
    scores = {
        'keyword_density': calculate_keyword_density(text),
        'action_verbs': count_action_verbs(text),
        'quantifiable_results': check_quantifiable_results(text),
        'structure': evaluate_structure(text),
        'contact_info': check_contact_info(text),
        'length': check_length(text)
    }
    return min(sum(scores[param] * weight for param, weight in ATS_PARAMETERS.items()) * 100, 100)

def generate_feedback(score, text):
    feedback = []
    if score < 60:
        feedback.append("Your resume needs significant optimization to pass ATS screening.")
    elif score < 80:
        feedback.append("Your resume is decent but could be improved for better ATS performance.")
    else:
        feedback.append("Excellent! Your resume is well-optimized for ATS systems.")
    
    if calculate_keyword_density(text) < 0.5:
        feedback.append("🔍 Increase keyword density with more relevant skills.")
    if count_action_verbs(text) < 0.5:
        feedback.append("💪 Add more action verbs like 'achieved', 'managed', 'developed'.")
    if check_quantifiable_results(text) < 0.5:
        feedback.append("📊 Include quantifiable achievements (e.g., 'increased sales by 30%').")
    if evaluate_structure(text) < 0.7:
        feedback.append("📑 Improve structure with clear section headings.")
    if check_contact_info(text) < 1:
        feedback.append("📱 Ensure all contact info (email, phone, LinkedIn) is included.")
    
    word_count = len(re.findall(r'\w+', text))
    if word_count < 400:
        feedback.append("📏 Resume is too short - aim for 400-800 words.")
    elif word_count > 800:
        feedback.append("📏 Resume is too long - keep under 800 words.")
    
    return feedback

# ================== JOB MATCH FUNCTIONS ==================
def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else "Not Found"

def extract_phone(text):
    match = re.search(r'(\+?\d{10,13})', text)
    return match.group(0) if match else "Not Found"

def extract_name(text):
    name_pattern = re.compile(r'\b([A-Z][A-Z]+(?:\s+[A-Z][A-Z]+){1,3})\b')
    matches = name_pattern.findall(text)
    blacklist = {'RESUME', 'SUMMARY', 'SKILLS', 'PROJECTS', 'EDUCATION', 'EXPERIENCE', 'CERTIFICATIONS'}
    
    for match in matches:
        if all(word not in blacklist for word in match.split()):
            if not re.search(r'\d', match):
                return match.title()
    return "Not Found"

def extract_skills(text):
    keywords = ['python', 'c++', 'c', 'web development', 'database', 'canva', 
               'cyber security', 'design thinking', 'cisco', 'machine learning','Python', 'Java', 'SQL', 'JavaScript', 'HTML', 'CSS', 'React', 'Node.js', 'Git', 'Docker',
 'Kubernetes', 'REST API', 'AWS', 'Azure', 'TensorFlow', 'PyTorch', 'OpenAI API', 'Flask', 'Django', 'Bash',
 'Data Analysis', 'Data Visualization', 'Data Science', 'Machine Learning', 'Deep Learning',
 'NLP (Natural Language Processing)', 'Predictive Modeling', 'Data Mining', 'Big Data',
 'Excel', 'Power BI', 'Tableau', 'ETL (Extract, Transform, Load)',
 'GitHub', 'JIRA', 'VS Code', 'MySQL', 'PostgreSQL', 'MongoDB', 'Google Analytics',
 'Apache Spark', 'Hadoop', 'Figma', 'Canva',
 'Problem Solving', 'Team Collaboration', 'Critical Thinking', 'Communication',
 'Time Management', 'Leadership', 'Adaptability', 'Creativity',
 'Developed', 'Managed', 'Analyzed', 'Designed', 'Implemented',
 'Led', 'Created', 'Optimized', 'Researched', 'Coordinated']
]
    skills_found = [word for word in keywords if word.lower() in text.lower()]
    return ', '.join(skills_found)

def calculate_match_score(resume_text, job_desc_text):
    resume_words = set(resume_text.lower().split())
    job_desc_words = set(job_desc_text.lower().split())
    common_words = resume_words.intersection(job_desc_words)
    match_percentage = (len(common_words) / len(job_desc_words)) * 100
    return min(round(match_percentage, 2), 100)

def generate_match_feedback(score):
    feedback = f"Your resume matches {score}% of the job description keywords. "
    if score > 70:
        feedback += "Excellent match!"
    elif score > 40:
        feedback += "Good match but could be improved."
    else:
        feedback += "Low match. Consider tailoring your resume."
    return feedback

# ================== ADMIN FUNCTIONS ==================
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.args.get('admin_key') != 'secret123':
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

def update_analytics(field):
    try:
        db = get_db()
        db.execute(f'''
            UPDATE app_analytics 
            SET {field} = {field} + 1 
            WHERE id = 1
        ''')
        db.commit()
    except Exception as e:
        print(f"Error updating analytics: {str(e)}")

# ================== ROUTES ==================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ats-checker', methods=['GET', 'POST'])
def ats_checker():
    if request.method == 'POST':
        if 'resume' not in request.files:
            return redirect(request.url)
        
        file = request.files['resume']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                text = extract_text(file)
                score = calculate_ats_score(text)
                feedback = generate_feedback(score, text)
                
                # Store in database
                db = get_db()
                db.execute('''
                    INSERT INTO resumes (filename, ats_score, upload_date, contact_info)
                    VALUES (?, ?, ?, ?)
                ''', (file.filename, score, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                      extract_email(text) or extract_phone(text)))
                db.commit()
                
                update_analytics('ats_checks')
                
                return render_template('ats_results.html', 
                                    score=round(score, 1),
                                    feedback=feedback,
                                    filename=file.filename)
            except Exception as e:
                return f"Error processing file: {str(e)}", 500
        
        return "Invalid file format. Please upload .pdf or .docx files only."
    
    return render_template('ats_checker.html')

@app.route('/compatibility-test', methods=['GET', 'POST'])
def compatibility_test():
    if request.method == 'POST':
        if 'resume' not in request.files or 'jobdescription' not in request.files:
            return redirect(request.url)
        
        resume_file = request.files['resume']
        job_desc_file = request.files['jobdescription']

        if resume_file.filename == '' or job_desc_file.filename == '':
            return redirect(request.url)
        
        if (resume_file and allowed_file(resume_file.filename) and 
            job_desc_file and allowed_file(job_desc_file.filename)):
            
            try:
                resume_text = extract_text(resume_file)
                job_desc_text = extract_text(job_desc_file)
                
                score = calculate_match_score(resume_text, job_desc_text)
                feedback = generate_match_feedback(score)
                
                # Store in database
                db = get_db()
                db.execute('''
                    INSERT INTO job_matches (resume_filename, jd_filename, match_score, analysis_date)
                    VALUES (?, ?, ?, ?)
                ''', (resume_file.filename, job_desc_file.filename, score, 
                      datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                db.commit()
                
                update_analytics('match_checks')
                
                return render_template('compatibility_results.html', 
                                    score=round(score, 1),
                                    feedback=feedback,
                                    resume_filename=resume_file.filename,
                                    job_desc_filename=job_desc_file.filename)
            except Exception as e:
                return f"Error processing files: {str(e)}", 500
        
        return "Invalid file format. Please upload .pdf or .docx files only."
    
    return render_template('compatibility_test.html')

@app.route('/resume-ranking', methods=['GET', 'POST'])
def resume_ranking():
    if request.method == 'POST':
        if 'resume' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['resume']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                text = extract_text(file)
                score = calculate_ats_score(text)
                
                # Store in database
                db = get_db()
                db.execute('''
                    INSERT INTO resumes (filename, ats_score, upload_date, contact_info)
                    VALUES (?, ?, ?, ?)
                ''', (file.filename, score, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                      extract_email(text) or extract_phone(text)))
                db.commit()
                
                update_analytics('resumes_uploaded')
                
                return render_template('upload_success.html', 
                                    score=round(score, 1),
                                    filename=file.filename)
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(request.url)
        
        flash('Invalid file format. Please upload PDF or DOCX only.', 'error')
        return redirect(request.url)
    
    # GET request - show form
    return render_template('resume_ranking.html')

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    db = get_db()
    
    # Initialize analytics if not exists
    analytics = db.execute('SELECT * FROM app_analytics WHERE id = 1').fetchone()
    if not analytics:
        db.execute('INSERT INTO app_analytics (id) VALUES (1)')
        db.commit()
        analytics = db.execute('SELECT * FROM app_analytics WHERE id = 1').fetchone()
    
    # Get stats
    total_resumes = db.execute('SELECT COUNT(*) FROM resumes').fetchone()[0]
    total_matches = db.execute('SELECT COUNT(*) FROM job_matches').fetchone()[0]
    
    # Get top resumes
    top_resumes = db.execute('''
        SELECT * FROM resumes 
        ORDER BY ats_score DESC 
        LIMIT 10
    ''').fetchall()
    
    # Get recent matches
    recent_matches = db.execute('''
        SELECT * FROM job_matches 
        ORDER BY analysis_date DESC 
        LIMIT 5
    ''').fetchall()
    
    return render_template('admin_dashboard.html', 
                         analytics=analytics,
                         top_resumes=top_resumes,
                         recent_matches=recent_matches,
                         total_resumes=total_resumes,
                         total_matches=total_matches)

@app.route('/admin/resume-analytics')
@admin_required
def resume_analytics():
    db = get_db()
    resumes = db.execute('''
        SELECT * FROM resumes 
        ORDER BY ats_score DESC
    ''').fetchall()
    return render_template('resume_analytics.html', resumes=resumes)
@app.route('/resume')
def resume_analyzer():
    return render_template('resume.html')  # or whatever your resume analyzer page is called
if __name__ == '__main__':
    app.run(debug=True)

