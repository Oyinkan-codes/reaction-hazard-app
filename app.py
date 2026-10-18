import os
import io
import re
import math
import hashlib
import hmac
import time
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, date, timedelta, timezone

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, declarative_base

# ==========================================
# 1. PAGE CONFIGURATION & INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="Global Chemical Reaction Hazard Engine",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

def cfg(key, default=""):
    """Retrieve secret or environment variable gracefully."""
    if key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key, default)

# ==========================================
# 2. DATABASE ENGINE & MODELS (PG8000 DRIVER)
# ==========================================
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = sa.Column(sa.Integer, primary_key=True)
    email = sa.Column(sa.String(255), unique=True, nullable=False)
    password_hash = sa.Column(sa.String(255), nullable=False)
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    pro_expires_at = sa.Column(sa.DateTime, nullable=True)

class Assessment(Base):
    __tablename__ = 'assessments'
    id = sa.Column(sa.Integer, primary_key=True)
    user_email = sa.Column(sa.String(255), nullable=False)
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    reaction_name = sa.Column(sa.String(255), nullable=False)
    risk_level = sa.Column(sa.String(50), nullable=False)
    report_hash = sa.Column(sa.String(64), nullable=False)

class PaymentNotice(Base):
    __tablename__ = 'payment_notices'
    id = sa.Column(sa.Integer, primary_key=True)
    user_email = sa.Column(sa.String(255), nullable=False)
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    amount_paid = sa.Column(sa.String(50), nullable=False)
    reference = sa.Column(sa.String(255), nullable=False)
    currency = sa.Column(sa.String(10), nullable=False)
    approved = sa.Column(sa.Boolean, default=False)

@st.cache_resource(show_spinner=False)
def get_engine():
    """Returns SQLAlchemy engine using pure-Python pg8000 driver with clean SSL parameters."""
    raw_url = str(cfg("DATABASE_URL", "sqlite:///reaction_hazard.db"))
    
    # Remove query parameters like ?sslmode=... for pg8000 compatibility
    if "?" in raw_url:
        base_url = raw_url.split("?")[0]
    else:
        base_url = raw_url
    
    # Standardise driver prefix to pure-Python pg8000
    if base_url.startswith("postgres://"):
        url = "postgresql+pg8000://" + base_url[len("postgres://"):]
    elif base_url.startswith("postgresql://") and not base_url.startswith("postgresql+"):
        url = "postgresql+pg8000://" + base_url[len("postgresql://"):]
    elif "postgresql+psycopg" in base_url:
        url = base_url.replace("postgresql+psycopg2://", "postgresql+pg8000://").replace("postgresql+psycopg://", "postgresql+pg8000://")
    else:
        url = base_url
        
    kwargs = {"pool_pre_ping": True}
    
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # Pass native SSL context required by Neon.tech to pg8000
        kwargs["connect_args"] = {"ssl_context": True}
        
    engine = sa.create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    return engine

# Initialize engine and session globally
engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ==========================================
# 3. HELPER FUNCTIONS, EMAIL & AUTH
# ==========================================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

def generate_report_hash(reaction_name: str, risk_level: str, user_email: str) -> str:
    secret_key = cfg("REPORT_SIGNING_KEY", "default_secret_key")
    data = f"{reaction_name}:{risk_level}:{user_email}:{secret_key}"
    return hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]

def is_pro_user(email: str) -> bool:
    admin_emails = [e.strip().lower() for e in cfg("ADMIN_EMAILS", "").split(",") if e.strip()]
    if email.lower() in admin_emails:
        return True
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email.lower()).first()
        if user and user.pro_expires_at:
            return user.pro_expires_at > datetime.utcnow()
    finally:
        db.close()
    return False

def get_daily_usage(email: str) -> int:
    db = SessionLocal()
    try:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        count = db.query(Assessment).filter(
            Assessment.user_email == email.lower(),
            Assessment.created_at >= today_start
        ).count()
        return count
    finally:
        db.close()

def send_payment_email_notification(user_email, amount, currency, reference):
    """Sends an automated email notification to the admin when a payment notice is submitted."""
    sender = cfg("GMAIL_SENDER", "")
    password = cfg("GMAIL_APP_PASSWORD", "")
    receiver = cfg("CONTACT_EMAIL", "oyinkansolaosibodu@gmail.com")
    
    if not sender or not password:
        return  # Email notifications skipped if secrets aren't configured
        
    subject = f"🔔 New Pro Payment Notice: {currency} {amount} from {user_email}"
    body = f"""
    New payment notice submitted on Global Chemical Hazard Engine:
    
    User Email: {user_email}
    Amount Paid: {currency} {amount}
    Details/Reference: {reference}
    Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
    
    Log into your app's Admin Panel to review and approve this claim.
    """
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
    except Exception as e:
        print(f"Failed to send notification email: {e}")

# Initialize Session State
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
if 'admin_unlocked' not in st.session_state:
    st.session_state.admin_unlocked = False

# ==========================================
# 4. SIDEBAR & AUTHENTICATION UI
# ==========================================
with st.sidebar:
    st.title("🧪 Reaction Engine")
    st.markdown("---")
    
    if not st.session_state.authenticated:
        st.subheader("Sign In / Register")
        auth_mode = st.radio("Choose Action", ["Login", "Register"])
        email_input = st.text_input("Email Address").strip().lower()
        pass_input = st.text_input("Password", type="password")
        
        if auth_mode == "Register":
            if st.button("Create Account", use_container_width=True):
                if email_input and pass_input:
                    db = SessionLocal()
                    try:
                        existing = db.query(User).filter(User.email == email_input).first()
                        if existing:
                            st.error("Account already exists. Please log in.")
                        else:
                            new_user = User(email=email_input, password_hash=hash_password(pass_input))
                            db.add(new_user)
                            db.commit()
                            st.success("Account created! Please log in.")
                    finally:
                        db.close()
                else:
                    st.warning("Please fill in all fields.")
        else:
            if st.button("Log In", use_container_width=True):
                if email_input and pass_input:
                    db = SessionLocal()
                    try:
                        user = db.query(User).filter(User.email == email_input).first()
                        if user and verify_password(pass_input, user.password_hash):
                            st.session_state.authenticated = True
                            st.session_state.user_email = user.email
                            st.rerun()
                        else:
                            st.error("Invalid credentials.")
                    finally:
                        db.close()
    else:
        st.success(f"Logged in as: **{st.session_state.user_email}**")
        pro_status = is_pro_user(st.session_state.user_email)
        if pro_status:
            st.markdown("🎖️ **Plan:** PRO (Unlimited)")
        else:
            free_limit = int(cfg("FREE_DAILY_RUNS", "3"))
            st.markdown("⚡ **Plan:** Free Tier")
            runs = get_daily_usage(st.session_state.user_email)
            st.progress(min(runs / free_limit, 1.0))
            st.caption(f"Daily Free Runs Used: {runs} / {free_limit}")
            
        if st.button("Log Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_email = ""
            st.session_state.admin_unlocked = False
            st.rerun()

# ==========================================
# 5. MAIN APPLICATION & NAVIGATION
# ==========================================
st.title("Global Chemical Reaction Hazard & Runaway Screening Engine")
st.caption("HSE / COMAH / OSHA Aligned Process Safety Assessment Platform")

tabs = ["Hazard Screening", "My Assessments", "Upgrade to Pro", "Admin Panel"]
tab1, tab2, tab3, tab4 = st.tabs(tabs)

# --- TAB 1: HAZARD SCREENING ---
with tab1:
    st.header("Reaction Process Screening")
    
    if not st.session_state.authenticated:
        st.info("Please sign in or create an account via the sidebar to run assessments.")
    else:
        pro = is_pro_user(st.session_state.user_email)
        usage = get_daily_usage(st.session_state.user_email)
        free_limit = int(cfg("FREE_DAILY_RUNS", "3"))
        
        if not pro and usage >= free_limit:
            st.error(f"❌ Free limit reached ({usage}/{free_limit} runs used today). Upgrade to Pro for unlimited assessments.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                rxn_name = st.text_input("Reaction/Process Name", "Exothermic Nitration Process")
                delta_h = st.number_input("Enthalpy of Reaction ΔHr (kJ/kg)", value=-850.0)
                cp = st.number_input("Specific Heat Capacity Cp (kJ/kg·K)", value=2.1)
                t_oper = st.number_input("Operating Temperature T_oper (°C)", value=65.0)
            
            with col2:
                t_boil = st.number_input("Boiling Temperature T_boil (°C)", value=110.0)
                ea = st.number_input("Activation Energy Ea (kJ/mol)", value=95.0)
                mrt = st.number_input("Mean Residence Time (hours)", value=4.0)

            if st.button("Run Thermal Hazard Assessment", type="primary"):
                # Adiabatic Temperature Rise Calculation
                delta_t_ad = abs(delta_h) / cp
                t_max_rx = t_oper + delta_t_ad
                
                # Risk Classification Matrix
                if delta_t_ad > 200 or t_max_rx > t_boil:
                    risk = "CRITICAL / HIGH RUNAWAY RISK"
                    badge_color = "red"
                elif delta_t_ad > 50:
                    risk = "MODERATE HAZARD"
                    badge_color = "orange"
                else:
                    risk = "LOW RISK"
                    badge_color = "green"
                
                st.markdown(f"### Assessment Result: :{badge_color}[{risk}]")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Adiabatic Temp Rise (ΔT_ad)", f"{delta_t_ad:.1f} °C")
                res_col2.metric("Max Temperature (T_max)", f"{t_max_rx:.1f} °C")
                res_col3.metric("Boiling Margin", f"{(t_boil - t_max_rx):.1f} °C")
                
                # Plot Simulation Chart
                times = np.linspace(0, mrt, 100)
                temps = t_oper + (delta_t_ad * (1 - np.exp(-times / (mrt / 3))))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=times, y=temps, mode='lines', name='Temperature Curve', line=dict(color='firebrick', width=3)))
                fig.add_hline(y=t_boil, line_dash="dash", line_color="blue", annotation_text="Boiling Point Threshold")
                fig.update_layout(title="Predicted Temperature Profile vs Time", xaxis_title="Time (Hours)", yaxis_title="Temperature (°C)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Save Assessment Record to Neon DB
                report_hash = generate_report_hash(rxn_name, risk, st.session_state.user_email)
                db = SessionLocal()
                try:
                    record = Assessment(
                        user_email=st.session_state.user_email,
                        reaction_name=rxn_name,
                        risk_level=risk,
                        report_hash=report_hash
                    )
                    db.add(record)
                    db.commit()
                    st.success(f"Assessment saved permanently to database. Report Hash: `{report_hash}`")
                finally:
                    db.close()

# --- TAB 2: MY ASSESSMENTS ---
with tab2:
    st.header("Assessment History & Verification")
    if not st.session_state.authenticated:
        st.info("Log in to view saved reports.")
    else:
        db = SessionLocal()
        try:
            records = db.query(Assessment).filter(Assessment.user_email == st.session_state.user_email).order_by(Assessment.created_at.desc()).all()
            if records:
                df = pd.DataFrame([{
                    "ID": r.id,
                    "Date": r.created_at.strftime("%Y-%m-%d %H:%M"),
                    "Reaction": r.reaction_name,
                    "Risk Level": r.risk_level,
                    "Verification Hash": r.report_hash
                } for r in records])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No saved assessments found.")
        finally:
            db.close()

# --- TAB 3: UPGRADE TO PRO ---
with tab3:
    st.header("Select Subscription Plan")
    st.markdown("Choose a plan tailored to your laboratory and engineering compliance needs.")
    
    # Display 3-Tier Pricing Cards
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.subheader("⚡ Starter")
        st.markdown("### ₦13,999 / mo")
        st.caption("Ideal for individual researchers & lab technicians")
        st.markdown("""
        * Unlimited Thermal Screenings
        * Basic Simulation Curves
        * Saved Assessment History
        """)
        
    with col_p2:
        st.subheader("🎖️ Professional")
        st.markdown("### ₦28,999 / mo")
        st.caption("Best for process safety consultants & plant engineers")
        st.markdown("""
        * Everything in Starter
        * Cryptographic Verification Hashes
        * Downloadable PDF Compliance Reports
        * Priority Email Support
        """)
        
    with col_p3:
        st.subheader("🏢 Enterprise")
        st.markdown("### ₦48,000 / mo")
        st.caption("Designed for chemical manufacturing sites & R&D teams")
        st.markdown("""
        * Everything in Professional
        * Multi-seat Team Access
        * Custom Kinetic Parameter Modeling
        * COMAH / OSHA Alignment Certification
        """)

    st.markdown("---")
    st.subheader("Payment Information")
    st.markdown(cfg("BANK_DETAILS", "Contact owner for payment details."))
    
    st.subheader("Submit Payment Notice")
    with st.form("payment_form"):
        selected_plan = st.selectbox("Select Plan to Activate", [
            "Starter Tier (₦13,999 / $10 USD / £8 GBP)",
            "Professional Tier (₦28,999 / $22 USD / £16 GBP)",
            "Enterprise Tier (₦48,000 / $36 USD / £27 GBP)"
        ])
        p_amount = st.text_input("Amount Paid", placeholder="e.g., 13999")
        p_currency = st.selectbox("Currency Paid", ["NGN", "GBP", "USD"])
        p_ref = st.text_input("Bank Reference / Transfer Description")
        submitted = st.form_submit_button("Notify Admin of Payment")
        
        if submitted:
            if p_amount and p_ref and st.session_state.authenticated:
                db = SessionLocal()
                try:
                    full_ref = f"[{selected_plan}] Ref: {p_ref}"
                    notice = PaymentNotice(
                        user_email=st.session_state.user_email,
                        amount_paid=p_amount,
                        currency=p_currency,
                        reference=full_ref
                    )
                    db.add(notice)
                    db.commit()
                    
                    # Trigger instant email notification to admin inbox
                    send_payment_email_notification(
                        st.session_state.user_email, 
                        p_amount, 
                        p_currency, 
                        full_ref
                    )
                    
                    st.success("Payment notification submitted! Admin will verify and activate your chosen plan upon confirmation.")
                finally:
                    db.close()
            else:
                st.warning("Please sign in and complete all form fields.")

# --- TAB 4: ADMIN PANEL ---
with tab4:
    st.header("System Administration")
    if not st.session_state.authenticated:
        st.info("Please log in with an admin account.")
    else:
        admin_emails = [e.strip().lower() for e in cfg("ADMIN_EMAILS", "").split(",") if e.strip()]
        if st.session_state.user_email.lower() not in admin_emails:
            st.error("Access Restricted: Your account does not have administrative privileges.")
        else:
            if not st.session_state.admin_unlocked:
                pass_input = st.text_input("Enter Admin Passphrase", type="password")
                if st.button("Unlock Admin Dashboard"):
                    if pass_input == cfg("ADMIN_PASSPHRASE", ""):
                        st.session_state.admin_unlocked = True
                        st.success("Admin Panel Unlocked.")
                        st.rerun()
                    else:
                        st.error("Incorrect Passphrase.")
            else:
                st.success("🔓 Admin Access Granted")
                st.subheader("Pending Payment Claims")
                
                db = SessionLocal()
                try:
                    notices = db.query(PaymentNotice).filter(PaymentNotice.approved == False).all()
                    if notices:
                        for n in notices:
                            col_a, col_b = st.columns([3, 1])
                            col_a.write(f"**User:** {n.user_email} | **Amount:** {n.currency} {n.amount_paid} | **Ref:** {n.reference}")
                            if col_b.button(f"Approve #{n.id}"):
                                n.approved = True
                                # Grant 30 days Pro access
                                user = db.query(User).filter(User.email == n.user_email).first()
                                if user:
                                    user.pro_expires_at = datetime.utcnow() + timedelta(days=30)
                                db.commit()
                                st.success(f"Granted Pro status to {n.user_email}")
                                st.rerun()
                    else:
                        st.info("No pending payment approvals.")
                finally:
                    db.close()

# ==========================================
# 6. FOOTER & LEGAL DISCLAIMERS
# ==========================================
st.markdown("---")
st.caption("© 2026 Global Chemical Reaction Hazard Screening Engine. Proprietary Software. All Rights Reserved. Aligned with HSE / COMAH / OSHA Guidelines.")
