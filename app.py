from flask import Flask, render_template, request, redirect
import mysql.connector
import joblib
import numpy as np
import os
from dotenv import load_dotenv
from google import genai 
app = Flask(__name__)


load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# CLEANED AI SECTION :
def get_ai_insight(emp):
    """Generates a cleaned professional performance summary."""
    model_name = 'gemini-2.5-flash' 

    prompt = (
        f"Analyze performance for {emp['name']}. "
        f"Attendance: {emp['attendance']}, Efficiency: {emp['task_efficiency']}, "
        f"Teamwork: {emp['teamwork']}, Initiative: {emp['initiative']}, Quality: {emp['project_quality']}. "
        "Provide a concise 2-sentence summary and one bulleted 'Growth Tip'. "
        "Do not use markdown bolding in the response."
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Log Error: {e}")
        return "Insight currently being generated. Please refresh shortly."

# Load AI model
model = joblib.load('model/performance_model.pkl')

# MySQL Connection
def get_db_connection():
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )
    return conn

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Add Employee & Predict Score   
@app.route('/add_employee', methods=['GET', 'POST'])
def add_employee():
    if request.method == 'POST':
        name = request.form['name']
        department = request.form['department']
        attendance = float(request.form['attendance'])
        task_efficiency = float(request.form['task_efficiency'])
        teamwork = float(request.form['teamwork'])
        initiative = float(request.form['initiative'])
        project_quality = float(request.form['project_quality'])

        # Predict performance
        features = np.array([[attendance, task_efficiency, teamwork, initiative, project_quality]])
        predicted_score = float(model.predict(features)[0])

        # Store in MySQL
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO employees 
            (name, department, attendance, task_efficiency, teamwork, initiative, project_quality, predicted_score)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (name, department, attendance, task_efficiency, teamwork, initiative, project_quality, predicted_score))
        conn.commit()
        cursor.close()
        conn.close()

        return redirect('/report')

    return render_template('add_employee.html')

# Evaluate Employee
@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name, department FROM employees")
    employees = cursor.fetchall()
    cursor.close()
    conn.close()

    if request.method == 'POST':
        emp_id = request.form['employee_id']
        task_efficiency = float(request.form['task_efficiency'])
        teamwork = float(request.form['teamwork'])
        project_quality = float(request.form['project_quality'])

        # Update employee scores
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE employees
            SET task_efficiency=%s, teamwork=%s, project_quality=%s
            WHERE id=%s
        """, (task_efficiency, teamwork, project_quality, emp_id))
        conn.commit()
        cursor.close()
        conn.close()

        # Re-predict overall score
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT attendance, task_efficiency, teamwork, initiative, project_quality FROM employees WHERE id=%s", (emp_id,))
        emp_data = cursor.fetchone()
        features = np.array([[emp_data['attendance'], emp_data['task_efficiency'], emp_data['teamwork'], emp_data['initiative'], emp_data['project_quality']]])
        new_score = float(model.predict(features)[0])

        cursor.execute("UPDATE employees SET predicted_score=%s WHERE id=%s", (new_score, emp_id))
        conn.commit()
        cursor.close()
        conn.close()

        return redirect('/report')

    return render_template('evaluate.html', employees=employees)

# View Reports
@app.route('/report')
def report():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM employees")
    employees = cursor.fetchall()
    
    # Add AI insights to each employee object before sending to template
    for emp in employees:
        emp['ai_insight'] = get_ai_insight(emp)
        
    cursor.close()
    conn.close()
    return render_template('report.html', employees=employees)

# Run App
if __name__ == '__main__':
    app.run(debug=True)
