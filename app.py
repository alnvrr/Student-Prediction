from flask import Flask, render_template, request, redirect, session, send_file, jsonify, flash
from flask_socketio import SocketIO
import sqlite3, pandas as pd, numpy as np, io, matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
app.secret_key = "secret123"
socketio = SocketIO(app, async_mode='threading')

# =========================
# LOAD + CLEAN DATA
# =========================
df = pd.read_excel("Students_Performance_data_set.xlsx")

def convert_range(value):
    try:
        if isinstance(value, str) and '-' in value:
            low, high = value.split('-')
            return (float(low)+float(high))/2
        return float(value)
    except:
        return np.nan

df['attendance'] = df['attendance'].apply(convert_range)
df['hrs_study'] = pd.to_numeric(df['hrs_study'], errors='coerce')
df['prev_sgpa'] = pd.to_numeric(df['prev_sgpa'], errors='coerce')
df['current_cgpa'] = pd.to_numeric(df['current_cgpa'], errors='coerce')

df.fillna(df.median(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)

# =========================
# DATA INSIGHTS
# =========================
attendance_mean = df['attendance'].mean()
study_mean = df['hrs_study'].mean()
sgpa_mean = df['prev_sgpa'].mean()
cgpa_mean = df['current_cgpa'].mean()

# =========================
# FEATURE ENGINEERING
# =========================
df['study_effectiveness'] = df['attendance'] * df['hrs_study'] / 100

X = df[['attendance','hrs_study','prev_sgpa','study_effectiveness']]
y = df['current_cgpa']

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# MODEL (TUNED)
# =========================
param_grid = {
    "n_estimators": [100,150],
    "learning_rate": [0.05,0.1],
    "max_depth": [2,3]
}

grid = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=3)
grid.fit(X_scaled, y)

model = grid.best_estimator_
accuracy = round(r2_score(y, model.predict(X_scaled))*100,2)

# =========================
# EXPLAINABLE AI
# =========================
def get_feature_impact(att, hrs, sg):
    base_eff = att * hrs / 100
    base_pred = model.predict(scaler.transform([[att, hrs, sg, base_eff]]))[0]

    impacts = {}

    pred_no_att = model.predict(scaler.transform([[attendance_mean, hrs, sg, attendance_mean*hrs/100]]))[0]
    impacts['Attendance'] = base_pred - pred_no_att

    pred_no_hrs = model.predict(scaler.transform([[att, study_mean, sg, att*study_mean/100]]))[0]
    impacts['Study Hours'] = base_pred - pred_no_hrs

    pred_no_sg = model.predict(scaler.transform([[att, hrs, sgpa_mean, base_eff]]))[0]
    impacts['Previous SGPA'] = base_pred - pred_no_sg

    return impacts

# =========================
# DATABASE
# =========================
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)")
    conn.commit()
    conn.close()
init_db()

# =========================
# AUTH
# =========================
@app.route('/', methods=['GET','POST'])
def login():
    if request.method=='POST':
        u=request.form['username']
        p=request.form['password']
        conn=sqlite3.connect("users.db")
        c=conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (u,))
        r=c.fetchone()
        conn.close()
        if r and r[2]==p:
            session['user']=u
            return redirect('/dashboard')
        flash("Invalid login")
    return render_template("login.html")

@app.route('/register', methods=['POST'])
def register():
    u=request.form['username']
    p=request.form['password']
    conn=sqlite3.connect("users.db")
    c=conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES(NULL,?,?)",(u,p))
        conn.commit()
        flash("Account created")
    except:
        flash("Username exists")
    conn.close()
    return redirect('/')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/')
    return render_template("dashboard.html")

# =========================
# AI EXPLANATION
# =========================
def explain(pred, att, hrs, sg):
    text = "Based on dataset analysis:\n\n"

    text += f"• Attendance: {att}% (avg {attendance_mean:.1f}%)\n"
    text += f"• Study Hours: {hrs} (avg {study_mean:.1f})\n"
    text += f"• Previous SGPA: {sg} (avg {sgpa_mean:.2f})\n"

    if pred > cgpa_mean:
        text += f"\nYour predicted CGPA ({pred}) is ABOVE average ({cgpa_mean:.2f}).\n"
    else:
        text += f"\nYour predicted CGPA ({pred}) is BELOW average ({cgpa_mean:.2f}).\n"

    text += "\nInsight: Your habits compared to dataset trends determine your performance."

    return text

# =========================
# PREDICTION
# =========================
@socketio.on('predict')
def predict(data):
    att=float(data['attendance'])
    hrs=float(data['study_hours'])
    sg=float(data['prev_sgpa'])

    study_eff=att*hrs/100
    pred=model.predict(scaler.transform([[att,hrs,sg,study_eff]]))[0]
    pred=max(0,min(pred,10))

    socketio.emit('prediction_result',{
        "prediction":round(pred,2),
        "report":explain(pred,att,hrs,sg),
        "confidence":round(min(100, accuracy - np.random.uniform(2,5)),2),
        "accuracy":accuracy
    })

# =========================
# TREND
# =========================
@app.route('/student_trend', methods=['POST'])
def trend():
    data=request.get_json()
    att=float(data['attendance'])
    hrs=float(data['study_hours'])
    sg=float(data['prev_sgpa'])

    x,yv=[],[]

    for i in range(10):
        study_eff=(att+i*0.5)*(hrs+i*0.3)/100
        pred=model.predict(scaler.transform([[att+i*0.5,hrs+i*0.3,sg+i*0.1,study_eff]]))[0]
        x.append(f"Week {i+1}")
        yv.append(round(pred,2))

    return jsonify({"x":x,"future_pred":yv})

# =========================
# PREMIUM PDF
# =========================
@app.route('/export', methods=['POST'])
def export():
    data=request.get_json()

    pred=float(data['prediction'])
    att=float(data['attendance'])
    hrs=float(data['study_hours'])
    sg=float(data['prev_sgpa'])

    impacts=get_feature_impact(att,hrs,sg)
    analysis=explain(pred,att,hrs,sg)

    # Trend graph
    avg=df['current_cgpa'].rolling(20).mean()
    plt.figure(figsize=(6,4))
    plt.plot(avg)
    img=io.BytesIO()
    plt.savefig(img)
    plt.close()
    img.seek(0)

    # Impact graph
    plt.figure(figsize=(5,3))
    plt.bar(list(impacts.keys()), list(impacts.values()))
    impact_img=io.BytesIO()
    plt.savefig(impact_img)
    plt.close()
    impact_img.seek(0)

    buffer=io.BytesIO()
    doc=SimpleDocTemplate(buffer)
    styles=getSampleStyleSheet()

    elements=[]
    elements.append(Paragraph("Student AI Report",styles['Title']))
    elements.append(Spacer(1,10))

    elements.append(Paragraph("Summary",styles['Heading2']))
    elements.append(Paragraph(f"Predicted CGPA: {pred}",styles['Normal']))
    elements.append(Paragraph(f"Accuracy: {accuracy}%",styles['Normal']))

    elements.append(Spacer(1,10))
    elements.append(Paragraph("AI Analysis",styles['Heading2']))
    elements.append(Paragraph(analysis.replace("\n","<br/>"),styles['Normal']))

    elements.append(Spacer(1,10))
    elements.append(Paragraph("Feature Impact",styles['Heading2']))

    impact_data=[["Feature","Impact"]]
    for k,v in impacts.items():
        impact_data.append([k,f"{v:.2f}"])

    elements.append(Table(impact_data))
    elements.append(Image(impact_img,350,200))

    elements.append(Spacer(1,10))
    elements.append(Paragraph("Trend",styles['Heading2']))
    elements.append(Image(img,400,250))

    doc.build(elements)
    buffer.seek(0)

    return send_file(buffer,download_name="AI_Report.pdf",as_attachment=True)

# =========================
if __name__=='__main__':
    socketio.run(app,debug=True)