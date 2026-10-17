"""
Global Chemical Reaction Hazard & Runaway Screening Engine
==========================================================
Copyright (c) 2026 Oyinkansola Osibodu. All rights reserved.

PROPRIETARY AND CONFIDENTIAL.
This source code, the user interface, the report templates, the documentation and the
product name are the intellectual property of the copyright holder. No licence is granted
except under a written agreement or the Terms of Use shown inside the application.
Copying, modifying, redistributing, reselling, hosting for third parties or reverse
engineering this software without written permission is prohibited.

Screening-level adiabatic thermal risk tool. NOT a substitute for DSC / ARC / RC1
calorimetry, a formal process hazard analysis, or the judgement of a competent person.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import textwrap
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import sqlalchemy as sa
import streamlit as st

# =============================================================================
# 1. CONFIGURATION  (override any value in Streamlit secrets or environment)
# =============================================================================
APP_NAME = "Global Chemical Reaction Hazard & Runaway Screening Engine"
APP_SHORT = "Reaction Hazard Screening Engine"
APP_VERSION = "2.0.0"
TERMS_VERSION = "2026-10-17"

R_GAS = 8.314462618  # J/(mol*K)
KELVIN = 273.15
WINDOW_H = 24.0
WAT = timedelta(hours=1)  # West Africa Time (no daylight saving)

PBKDF2_ITERS = 600_000
MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 15
PRO_DAYS = 30


def cfg(key: str, default=None):
    """Read a setting from Streamlit secrets, then environment, then default."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


def _as_int(value, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


OWNER = str(cfg("OWNER_NAME", "Oyinkansola Osibodu"))
CONTACT_EMAIL = str(cfg("CONTACT_EMAIL", "oyinkansolaosibodu@gmail.com"))
FREE_DAILY_RUNS = _as_int(cfg("FREE_DAILY_RUNS", 5), 5)
PRO_PRICE_NGN = _as_int(cfg("PRO_PRICE_NGN", 5000), 5000)
COPYRIGHT_LINE = f"© 2026 {OWNER}. All rights reserved."


def admin_emails() -> set:
    raw = cfg("ADMIN_EMAILS", CONTACT_EMAIL)
    if isinstance(raw, str):
        raw = raw.split(",")
    return {str(e).strip().lower() for e in (raw or []) if str(e).strip()}


def utcnow() -> datetime:
    """Naive UTC timestamp (portable across SQLite and PostgreSQL)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def ngn(amount: int) -> str:
    return f"₦{amount:,}"


# =============================================================================
# 2. DATABASE LAYER  (SQLite locally, PostgreSQL in production)
# =============================================================================
metadata = sa.MetaData()

users_t = sa.Table(
    "users", metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("email", sa.String(255), unique=True, nullable=False),
    sa.Column("name", sa.String(120), nullable=False),
    sa.Column("pw_hash", sa.String(255), nullable=False),
    sa.Column("plan", sa.String(16), nullable=False, default="free"),
    sa.Column("plan_expires", sa.DateTime),
    sa.Column("created_at", sa.DateTime, nullable=False),
    sa.Column("last_login", sa.DateTime),
    sa.Column("failed_attempts", sa.Integer, nullable=False, default=0),
    sa.Column("locked_until", sa.DateTime),
    sa.Column("terms_version", sa.String(32)),
    sa.Column("terms_accepted_at", sa.DateTime),
)

assessments_t = sa.Table(
    "assessments", metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("user_id", sa.Integer, nullable=False, index=True),
    sa.Column("created_at", sa.DateTime, nullable=False, index=True),
    sa.Column("batch_name", sa.String(200), nullable=False),
    sa.Column("inputs_json", sa.Text, nullable=False),
    sa.Column("results_json", sa.Text, nullable=False),
    sa.Column("level", sa.String(16), nullable=False),
    sa.Column("report_id", sa.String(40), unique=True, nullable=False),
    sa.Column("report_hash", sa.String(64), nullable=False),
)

payments_t = sa.Table(
    "payments", metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("user_id", sa.Integer, nullable=False, index=True),
    sa.Column("reference", sa.String(64), unique=True, nullable=False),
    sa.Column("amount_kobo", sa.Integer, nullable=False),
    sa.Column("status", sa.String(16), nullable=False),
    sa.Column("created_at", sa.DateTime, nullable=False),
    sa.Column("paid_at", sa.DateTime),
)


transfers_t = sa.Table(
    "transfer_claims", metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("user_id", sa.Integer, nullable=False, index=True),
    sa.Column("created_at", sa.DateTime, nullable=False),
    sa.Column("payer_name", sa.String(120), nullable=False),
    sa.Column("transfer_ref", sa.String(160), nullable=False),
    sa.Column("note", sa.String(300)),
    sa.Column("status", sa.String(16), nullable=False),  # pending / approved / rejected
    sa.Column("decided_at", sa.DateTime),
)


@st.cache_resource(show_spinner=False)
def get_engine():
    url = str(cfg("DATABASE_URL", "sqlite:///reaction_hazard.db"))
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = sa.create_engine(url, **kwargs)
    metadata.create_all(engine)
    return engine


def using_ephemeral_db() -> bool:
    return str(cfg("DATABASE_URL", "sqlite")).startswith("sqlite")


# =============================================================================
# 3. AUTHENTICATION
# =============================================================================
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-']+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def hash_password(password: str, iterations: int = PBKDF2_ITERS) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iters, salt_hex, hash_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters)
        )
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def _dummy_hash() -> str:
    return hash_password("timing-equaliser")


def password_problem(password: str):
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return "Password must contain at least one letter and one number."
    if len(password) > 200:
        return "Password is too long."
    return None


def register_user(email: str, name: str, password: str, accepted_terms: bool):
    email = (email or "").strip().lower()
    name = " ".join((name or "").split())
    if not accepted_terms:
        return False, "You must accept the Terms of Use, Privacy Notice and Disclaimer to create an account."
    if not EMAIL_RE.match(email) or len(email) > 255:
        return False, "Please enter a valid email address."
    if len(name) < 2 or len(name) > 120:
        return False, "Please enter your full name (2-120 characters)."
    problem = password_problem(password or "")
    if problem:
        return False, problem
    now = utcnow()
    try:
        with get_engine().begin() as conn:
            if conn.execute(sa.select(users_t.c.id).where(users_t.c.email == email)).first():
                return False, "An account with this email already exists. Please log in."
            conn.execute(sa.insert(users_t).values(
                email=email, name=name, pw_hash=hash_password(password), plan="free",
                created_at=now, failed_attempts=0, terms_version=TERMS_VERSION,
                terms_accepted_at=now,
            ))
    except sa.exc.IntegrityError:
        return False, "An account with this email already exists. Please log in."
    return True, "Account created. You can now log in."


def authenticate(email: str, password: str):
    """Return (user_dict | None, message). Includes lockout after repeated failures."""
    email = (email or "").strip().lower()
    now = utcnow()
    with get_engine().begin() as conn:
        row = conn.execute(sa.select(users_t).where(users_t.c.email == email)).mappings().first()
        if row is None:
            verify_password(password or "", _dummy_hash())  # equalise timing
            return None, "Incorrect email or password."
        if row["locked_until"] and row["locked_until"] > now:
            mins = max(1, math.ceil((row["locked_until"] - now).total_seconds() / 60))
            return None, f"Too many failed attempts. Try again in about {mins} minute(s)."
        if verify_password(password or "", row["pw_hash"]):
            conn.execute(sa.update(users_t).where(users_t.c.id == row["id"]).values(
                failed_attempts=0, locked_until=None, last_login=now))
            return dict(row), ""
        attempts = (row["failed_attempts"] or 0) + 1
        values = {"failed_attempts": attempts}
        if attempts >= MAX_FAILED_LOGINS:
            values = {"failed_attempts": 0, "locked_until": now + timedelta(minutes=LOCKOUT_MINUTES)}
        conn.execute(sa.update(users_t).where(users_t.c.id == row["id"]).values(**values))
        return None, "Incorrect email or password."


def get_user(uid: int):
    with get_engine().connect() as conn:
        row = conn.execute(sa.select(users_t).where(users_t.c.id == uid)).mappings().first()
    return dict(row) if row else None


def plan_of(user: dict) -> str:
    """'pro' or 'free'. Owner/admin accounts always have full access."""
    if user["email"] in admin_emails():
        return "pro"
    if user["plan"] == "pro" and (user["plan_expires"] is None or user["plan_expires"] > utcnow()):
        return "pro"
    return "free"


def is_admin(user: dict) -> bool:
    return user["email"] in admin_emails()


# =============================================================================
# 4. USAGE, ASSESSMENT RECORDS AND INTEGRITY HASH
# =============================================================================
def day_start_utc() -> datetime:
    """Start of the current West Africa Time day, expressed in UTC."""
    local = utcnow() + WAT
    return local.replace(hour=0, minute=0, second=0, microsecond=0) - WAT


def runs_today(uid: int) -> int:
    with get_engine().connect() as conn:
        return conn.execute(
            sa.select(sa.func.count()).select_from(assessments_t).where(
                assessments_t.c.user_id == uid, assessments_t.c.created_at >= day_start_utc())
        ).scalar() or 0


def integrity_hash(payload: dict) -> str:
    """SHA-256 of the canonical record; HMAC-SHA256 when REPORT_SIGNING_KEY is configured."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    key = cfg("REPORT_SIGNING_KEY")
    if key:
        return hmac.new(str(key).encode("utf-8"), blob, hashlib.sha256).hexdigest()
    return hashlib.sha256(blob).hexdigest()


def _record_payload(report_id, created, email, params, res, level):
    return {
        "report_id": report_id,
        "generated_utc": created.strftime("%Y-%m-%d %H:%M:%S"),
        "user": email,
        "inputs": params,
        "results": res,
        "level": level,
    }


def save_assessment(user: dict, params: dict, res: dict, level: str):
    created = utcnow().replace(microsecond=0)
    report_id = f"RHS-{created:%Y%m%d}-{uuid.uuid4().hex[:8].upper()}"
    payload = _record_payload(report_id, created, user["email"], params, res, level)
    digest = integrity_hash(payload)
    with get_engine().begin() as conn:
        conn.execute(sa.insert(assessments_t).values(
            user_id=user["id"], created_at=created, batch_name=params["name"][:200],
            inputs_json=json.dumps(params), results_json=json.dumps(res, default=str),
            level=level, report_id=report_id, report_hash=digest))
    return report_id, digest, created


def list_assessments(uid: int, limit: int) -> pd.DataFrame:
    with get_engine().connect() as conn:
        rows = conn.execute(
            sa.select(assessments_t).where(assessments_t.c.user_id == uid)
            .order_by(assessments_t.c.created_at.desc(), assessments_t.c.id.desc()).limit(limit)
        ).mappings().all()
    out = []
    for r in rows:
        res = json.loads(r["results_json"])
        out.append({
            "Date (UTC)": r["created_at"].strftime("%Y-%m-%d %H:%M"),
            "Batch": r["batch_name"],
            "dT_ad (°C)": round(res["dt_ad"], 1),
            "T_max (°C)": round(res["t_max"], 1),
            "TMR_ad (h)": None if res["tmr_h"] in (None, float("inf")) else round(res["tmr_h"], 2),
            "Risk": r["level"].upper(),
            "Report ID": r["report_id"],
        })
    return pd.DataFrame(out)


def verify_report(uid: int, report_id: str):
    """Re-compute the stored record's hash to prove it was issued, unaltered, by this system."""
    with get_engine().connect() as conn:
        row = conn.execute(sa.select(assessments_t).where(
            assessments_t.c.report_id == report_id.strip().upper(),
            assessments_t.c.user_id == uid)).mappings().first()
        if not row:
            return None
        email = conn.execute(sa.select(users_t.c.email).where(users_t.c.id == uid)).scalar()
    payload = _record_payload(row["report_id"], row["created_at"], email,
                              json.loads(row["inputs_json"]), json.loads(row["results_json"]), row["level"])
    ok = hmac.compare_digest(integrity_hash(payload), row["report_hash"])
    return {"ok": ok, "batch": row["batch_name"], "created": row["created_at"], "hash": row["report_hash"]}


# =============================================================================
# 5. PAYMENTS (Paystack) AND ADMIN OPERATIONS
# =============================================================================
PAYSTACK_API = "https://api.paystack.co"


def paystack_enabled() -> bool:
    return bool(cfg("PAYSTACK_SECRET_KEY")) and bool(cfg("APP_URL"))


def _paystack_headers() -> dict:
    return {"Authorization": f"Bearer {cfg('PAYSTACK_SECRET_KEY')}", "Content-Type": "application/json"}


def start_payment(user: dict):
    """Create a pending payment and return (authorization_url | None, message)."""
    if not paystack_enabled():
        return None, "Online payment is not enabled yet."
    reference = f"RHS{user['id']}X{uuid.uuid4().hex[:14]}"
    amount_kobo = PRO_PRICE_NGN * 100
    try:
        r = requests.post(
            f"{PAYSTACK_API}/transaction/initialize", headers=_paystack_headers(), timeout=20,
            json={"email": user["email"], "amount": amount_kobo, "currency": "NGN",
                  "reference": reference, "callback_url": str(cfg("APP_URL")),
                  "metadata": {"user_id": user["id"], "product": "pro_30_days"}})
        data = r.json()
    except (requests.RequestException, ValueError):
        return None, "Could not reach the payment provider. Please try again."
    if not data.get("status"):
        return None, "The payment provider rejected the request. Please try again later."
    with get_engine().begin() as conn:
        conn.execute(sa.insert(payments_t).values(
            user_id=user["id"], reference=reference, amount_kobo=amount_kobo,
            status="pending", created_at=utcnow()))
    return data["data"]["authorization_url"], ""


def grant_pro(conn, uid: int, days: int):
    row = conn.execute(sa.select(users_t.c.plan_expires, users_t.c.plan).where(users_t.c.id == uid)).first()
    base = utcnow()
    if row and row[1] == "pro" and row[0] and row[0] > base:
        base = row[0]  # extend from current expiry
    conn.execute(sa.update(users_t).where(users_t.c.id == uid).values(
        plan="pro", plan_expires=base + timedelta(days=days)))


def confirm_payment(user: dict, reference: str):
    """Verify a Paystack transaction server-side and activate Pro exactly once."""
    reference = (reference or "").strip()
    if not reference or not paystack_enabled():
        return False, "Payment verification is unavailable."
    with get_engine().connect() as conn:
        pay = conn.execute(sa.select(payments_t).where(payments_t.c.reference == reference)).mappings().first()
    if not pay or pay["user_id"] != user["id"]:
        return False, "This payment reference does not belong to your account."
    if pay["status"] == "success":
        return True, "This payment was already confirmed. Pro is active."
    try:
        r = requests.get(f"{PAYSTACK_API}/transaction/verify/{reference}",
                         headers=_paystack_headers(), timeout=20)
        data = r.json()
    except (requests.RequestException, ValueError):
        return False, "Could not reach the payment provider. Please try again."
    d = data.get("data") or {}
    paid_ok = (
        data.get("status") is True and d.get("status") == "success"
        and d.get("amount") == pay["amount_kobo"] and d.get("currency", "NGN") == "NGN"
        and str((d.get("customer") or {}).get("email", "")).lower() == user["email"]
    )
    if not paid_ok:
        status = d.get("status", "unknown")
        return False, f"Payment not confirmed yet (status: {status}). If you were charged, wait a minute and press Verify again."
    with get_engine().begin() as conn:
        done = conn.execute(sa.update(payments_t).where(
            payments_t.c.reference == reference, payments_t.c.status != "success"
        ).values(status="success", paid_at=utcnow()))
        if done.rowcount == 1:
            grant_pro(conn, user["id"], PRO_DAYS)
    return True, f"Payment confirmed. Pro is active for {PRO_DAYS} days. Thank you!"


def latest_pending_reference(uid: int):
    with get_engine().connect() as conn:
        return conn.execute(
            sa.select(payments_t.c.reference).where(
                payments_t.c.user_id == uid, payments_t.c.status == "pending")
            .order_by(payments_t.c.id.desc()).limit(1)).scalar()


def bank_details_text() -> str:
    """Bank-transfer instructions, kept in secrets (never in the code)."""
    v = cfg("BANK_DETAILS")
    return str(v).strip() if v else ""


def submit_transfer_claim(user: dict, payer_name: str, transfer_ref: str, note: str):
    payer_name = " ".join((payer_name or "").split())
    transfer_ref = " ".join((transfer_ref or "").split())
    note = " ".join((note or "").split())[:300]
    if len(payer_name) < 2:
        return False, "Enter the name on the account you paid from."
    if len(transfer_ref) < 3:
        return False, "Enter the transfer reference, session ID or date/time and bank you paid from."
    with get_engine().begin() as conn:
        pending = conn.execute(sa.select(sa.func.count()).select_from(transfers_t).where(
            transfers_t.c.user_id == user["id"], transfers_t.c.status == "pending")).scalar() or 0
        if pending >= 3:
            return False, "You already have pending payment notices. Please wait for them to be reviewed."
        conn.execute(sa.insert(transfers_t).values(
            user_id=user["id"], created_at=utcnow(), payer_name=payer_name[:120],
            transfer_ref=transfer_ref[:160], note=note, status="pending"))
    return True, "Payment notice sent. Pro is activated once your transfer is confirmed (usually within 24 hours)."


def list_user_claims(uid: int) -> pd.DataFrame:
    with get_engine().connect() as conn:
        rows = conn.execute(sa.select(
            transfers_t.c.created_at, transfers_t.c.transfer_ref, transfers_t.c.status
        ).where(transfers_t.c.user_id == uid).order_by(transfers_t.c.id.desc()).limit(10)).mappings().all()
    return pd.DataFrame([{"Submitted (UTC)": r["created_at"].strftime("%Y-%m-%d %H:%M"),
                          "Reference": r["transfer_ref"], "Status": r["status"].title()} for r in rows])


def pending_claims() -> list:
    with get_engine().connect() as conn:
        rows = conn.execute(
            sa.select(transfers_t, users_t.c.email).join(users_t, users_t.c.id == transfers_t.c.user_id)
            .where(transfers_t.c.status == "pending").order_by(transfers_t.c.id)).mappings().all()
    return [dict(r) for r in rows]


def decide_claim(claim_id: int, approve: bool, days: int = PRO_DAYS):
    """Approve (grants Pro and records revenue) or reject a pending transfer claim, exactly once."""
    now = utcnow()
    with get_engine().begin() as conn:
        row = conn.execute(sa.select(transfers_t).where(transfers_t.c.id == claim_id)).mappings().first()
        if not row:
            return False, "Claim not found."
        done = conn.execute(sa.update(transfers_t).where(
            transfers_t.c.id == claim_id, transfers_t.c.status == "pending"
        ).values(status="approved" if approve else "rejected", decided_at=now))
        if done.rowcount != 1:
            return False, "This claim was already decided."
        if approve:
            grant_pro(conn, row["user_id"], days)
            conn.execute(sa.insert(payments_t).values(
                user_id=row["user_id"], reference=f"TRANSFER-{claim_id}",
                amount_kobo=PRO_PRICE_NGN * 100, status="success", created_at=row["created_at"], paid_at=now))
    return True, "Approved: Pro activated." if approve else "Rejected."


def admin_grant(email: str, days: int):
    email = email.strip().lower()
    with get_engine().begin() as conn:
        uid = conn.execute(sa.select(users_t.c.id).where(users_t.c.email == email)).scalar()
        if uid is None:
            return False, "No user with that email."
        grant_pro(conn, uid, days)
    return True, f"Granted {days} day(s) of Pro to {email}."


def admin_revoke(email: str):
    email = email.strip().lower()
    with get_engine().begin() as conn:
        n = conn.execute(sa.update(users_t).where(users_t.c.email == email).values(
            plan="free", plan_expires=None)).rowcount
    return (True, f"Pro revoked for {email}.") if n else (False, "No user with that email.")


def admin_reset_password(email: str, new_password: str):
    problem = password_problem(new_password)
    if problem:
        return False, problem
    email = email.strip().lower()
    with get_engine().begin() as conn:
        n = conn.execute(sa.update(users_t).where(users_t.c.email == email).values(
            pw_hash=hash_password(new_password), failed_attempts=0, locked_until=None)).rowcount
    return (True, f"Password reset for {email}. Ask the user to change it after logging in.") if n else (False, "No user with that email.")


def admin_stats() -> dict:
    now = utcnow()
    with get_engine().connect() as conn:
        users = conn.execute(sa.select(sa.func.count()).select_from(users_t)).scalar() or 0
        pro = conn.execute(sa.select(sa.func.count()).select_from(users_t).where(
            users_t.c.plan == "pro", sa.or_(users_t.c.plan_expires.is_(None), users_t.c.plan_expires > now))).scalar() or 0
        runs = conn.execute(sa.select(sa.func.count()).select_from(assessments_t)).scalar() or 0
        today = conn.execute(sa.select(sa.func.count()).select_from(assessments_t).where(
            assessments_t.c.created_at >= day_start_utc())).scalar() or 0
        revenue = conn.execute(sa.select(sa.func.coalesce(sa.func.sum(payments_t.c.amount_kobo), 0)).where(
            payments_t.c.status == "success")).scalar() or 0
    return {"users": users, "pro": pro, "runs": runs, "today": today, "revenue_ngn": revenue // 100}


def admin_users_df() -> pd.DataFrame:
    with get_engine().connect() as conn:
        rows = conn.execute(sa.select(
            users_t.c.email, users_t.c.name, users_t.c.plan, users_t.c.plan_expires,
            users_t.c.created_at, users_t.c.last_login, users_t.c.terms_version,
            users_t.c.terms_accepted_at).order_by(users_t.c.id.desc())).mappings().all()
    return pd.DataFrame([dict(r) for r in rows])


def admin_payments_df() -> pd.DataFrame:
    with get_engine().connect() as conn:
        rows = conn.execute(sa.select(
            payments_t.c.reference, payments_t.c.user_id, payments_t.c.amount_kobo,
            payments_t.c.status, payments_t.c.created_at, payments_t.c.paid_at
        ).order_by(payments_t.c.id.desc()).limit(500)).mappings().all()
    df = pd.DataFrame([dict(r) for r in rows])
    if not df.empty:
        df["amount_ngn"] = df["amount_kobo"] // 100
        df = df.drop(columns=["amount_kobo"])
    return df


# =============================================================================
# 6. CORE CALCULATION ENGINE
# =============================================================================
def c_to_k(t_c: float) -> float:
    return t_c + KELVIN


def parse_float(text):
    try:
        v = float(str(text).strip())
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def validate_inputs(p: dict) -> list:
    errors = []
    if not p["name"].strip():
        errors.append("Reaction / batch identifier must not be empty.")
    if len(p["name"]) > 200:
        errors.append("Reaction / batch identifier is too long (max 200 characters).")
    if p["cp"] <= 0:
        errors.append("Specific heat capacity (Cp) must be greater than zero.")
    if p["rho"] <= 0:
        errors.append("Mixture density must be greater than zero.")
    if p["c0"] <= 0:
        errors.append("Initial concentration (C0) must be greater than zero.")
    if p["ea"] <= 0:
        errors.append("Activation energy (Ea) must be greater than zero.")
    if p["a"] is None or p["a"] <= 0:
        errors.append("Pre-exponential factor (A) must be a positive number (e.g. 1e8).")
    if p["t_op"] <= -KELVIN:
        errors.append("Operating temperature must be above absolute zero (-273.15 °C).")
    if p["dh"] >= 0:
        errors.append("Reaction enthalpy must be negative (exothermic) for a runaway screen. "
                      "Endothermic or thermoneutral reactions do not self-heat adiabatically.")
    return errors


def sanity_warnings(p: dict, res: dict) -> list:
    w = []
    if res["t_max"] > 300:
        w.append("T_max exceeds 300 °C: secondary decomposition is likely and this single-reaction "
                 "model loses validity. Test the reaction mass by DSC/ARC.")
    if p["ea"] < 20 or p["ea"] > 300:
        w.append("Activation energy is outside the typical 20-300 kJ/mol range. Check the units.")
    if p["a"] > 1e25 or p["a"] < 1:
        w.append("Pre-exponential factor looks unusual (typical range about 1e3 to 1e20). Check the value.")
    if p["c0"] > 50:
        w.append("Initial concentration above 50 mol/L is unusual. Check the units.")
    if p["cp"] > 10 or p["cp"] < 0.5:
        w.append("Specific heat outside 0.5-10 J/(g·°C) is unusual for a liquid reaction mass.")
    if res["dt_ad"] > 50 and p["t_op"] + res["dt_ad"] > 100:
        w.append("T_max may exceed the boiling point of common solvents. Assess vapour pressure and "
                 "venting; this tool does not model boil-off or gas generation.")
    return w


def calculate(p: dict) -> dict:
    t_k = c_to_k(p["t_op"])
    dt_ad = (-p["dh"] * p["c0"] * 1000.0) / (p["rho"] * p["cp"])
    t_max = p["t_op"] + dt_ad
    exponent = -p["ea"] * 1000.0 / (R_GAS * t_k)
    k_top = p["a"] * math.exp(exponent) if exponent > -700 else 0.0
    rate = k_top * p["c0"] ** p["order"]               # mol/(L·s)
    q_vol = -p["dh"] * 1000.0 * rate                    # W/L, positive = heat released
    if q_vol > 0:
        tmr_h = (p["cp"] * p["rho"] * R_GAS * t_k ** 2) / (q_vol * p["ea"] * 1000.0) / 3600.0
    else:
        tmr_h = math.inf
    return {"t_k": t_k, "dt_ad": dt_ad, "t_max": t_max, "k_top": k_top,
            "rate": rate, "q_vol": q_vol, "tmr_h": tmr_h}


def classify(tmr_h: float, dt_ad: float):
    if tmr_h < 8 or dt_ad > 100:
        reasons = []
        if tmr_h < 8:
            reasons.append("TMR_ad < 8 h")
        if dt_ad > 100:
            reasons.append("dT_ad > 100 °C")
        return ("high", "HIGH RISK / CRITICAL",
                "Triggered by: " + " and ".join(reasons) + ". Do not scale up without calorimetric data "
                "(DSC/ARC), a defined cooling-failure scenario and an independent safety review.")
    if tmr_h <= 24:
        return ("moderate", "MODERATE RISK",
                "TMR_ad between 8 and 24 h. Confirm cooling capacity, temperature interlocks and "
                "emergency response times before scale-up.")
    return ("low", "LOW RISK (screening level)",
            "TMR_ad > 24 h and dT_ad <= 100 °C. Standard controls apply. This is a screening result "
            "only; verify with experimental data.")


def simulate_adiabatic(p: dict, res: dict, window_h: float = WINDOW_H) -> pd.DataFrame:
    """Adiabatic self-heating with reactant consumption (adaptive RK4, no SciPy)."""
    n = p["order"]
    t0_k, dt_ad, c0 = res["t_k"], res["dt_ad"], p["c0"]

    def rate_x(x: float) -> float:
        x = min(max(x, 0.0), 1.0)
        t = t0_k + dt_ad * x
        expo = -p["ea"] * 1000.0 / (R_GAS * t)
        if expo < -700:
            return 0.0
        return p["a"] * math.exp(expo) * c0 ** (n - 1) * (1.0 - x) ** n

    t_end = window_h * 3600.0
    t, x = 0.0, 0.0
    times, temps = [0.0], [p["t_op"]]
    it = 0
    while t < t_end and x < 0.9999 and it < 300000:
        it += 1
        r = rate_x(x)
        dt = 300.0
        if r > 0:
            dt = min(dt, 0.002 / r)
            if dt_ad > 0:
                dt = min(dt, 0.2 / (dt_ad * r))
        dt = max(min(dt, t_end - t), 1e-6)
        k1 = rate_x(x)
        k2 = rate_x(x + 0.5 * dt * k1)
        k3 = rate_x(x + 0.5 * dt * k2)
        k4 = rate_x(x + dt * k3)
        x = min(x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, 1.0)
        t += dt
        times.append(t)
        temps.append(p["t_op"] + dt_ad * x)
    if times[-1] < t_end:
        times.append(t_end)
        temps.append(temps[-1])
    df = pd.DataFrame({"time_h": np.array(times) / 3600.0, "temp_c": temps})
    if len(df) > 4000:
        idx = np.unique(np.linspace(0, len(df) - 1, 4000).astype(int))
        df = df.iloc[idx].reset_index(drop=True)
    return df


def peak_heating_time(df: pd.DataFrame):
    t = df["time_h"].to_numpy() * 3600.0
    temp = df["temp_c"].to_numpy()
    if len(t) < 3:
        return None
    dtdt = np.gradient(temp, t)
    i = int(np.argmax(dtdt))
    if i <= 1 or dtdt[i] <= 0 or i >= len(t) - 2:
        return None
    return float(t[i] / 3600.0)


# =============================================================================
# 7. REPORTING: CSV AND PDF (dependency-free)
# =============================================================================
def tmr_display(tmr_h) -> str:
    """Readable TMR_ad: very large values mean there is no practical self-heating."""
    if tmr_h is None or math.isinf(tmr_h) or tmr_h > 1e6:
        return "> 1,000,000"
    return f"{tmr_h:,.2f}"


def fmt_tmr(tmr_h) -> str:
    return tmr_display(tmr_h) + (" (no significant heat release)" if tmr_h is None or tmr_h > 1e6 else "")


def report_rows(rec: dict) -> list:
    p, res = rec["params"], rec["res"]
    peak = rec["peak_h"]
    return [
        ("Record", "Report ID", rec["report_id"], ""),
        ("Record", "Generated (UTC)", rec["created"].strftime("%Y-%m-%d %H:%M:%S"), ""),
        ("Record", "Reaction / batch ID", p["name"], ""),
        ("Record", "Prepared by", f"{rec['user_name']} <{rec['user_email']}>", ""),
        ("Record", "Tool / version", f"{APP_NAME} v{APP_VERSION}", ""),
        ("Inputs", "Reaction enthalpy (dH_rxn)", f"{p['dh']:.2f}", "kJ/mol"),
        ("Inputs", "Initial concentration (C0)", f"{p['c0']:.4g}", "mol/L"),
        ("Inputs", "Specific heat capacity (Cp)", f"{p['cp']:.4g}", "J/(g.degC)"),
        ("Inputs", "Mixture density (rho)", f"{p['rho']:.4g}", "kg/m3"),
        ("Inputs", "Activation energy (Ea)", f"{p['ea']:.4g}", "kJ/mol"),
        ("Inputs", "Pre-exponential factor (A)", f"{p['a']:.4e}", "1/s or L/(mol.s)"),
        ("Inputs", "Reaction order (n)", str(p["order"]), "-"),
        ("Inputs", "Operating temperature (T_op)", f"{p['t_op']:.2f}", "degC"),
        ("Results", "Operating temperature (T_op)", f"{res['t_k']:.2f}", "K"),
        ("Results", "Adiabatic temperature rise (dT_ad)", f"{res['dt_ad']:.2f}", "degC"),
        ("Results", "Maximum adiabatic temperature (T_max)", f"{res['t_max']:.2f}", "degC"),
        ("Results", "Rate constant at T_op (k)", f"{res['k_top']:.4e}", "1/s or L/(mol.s)"),
        ("Results", "Reaction rate at T_op", f"{res['rate']:.4e}", "mol/(L.s)"),
        ("Results", "Heat generation rate q(T_op)", f"{res['q_vol']:.4e}", "W/L"),
        ("Results", "Time to maximum rate (TMR_ad)", fmt_tmr(res["tmr_h"]), "h"),
        ("Results", "Simulated peak self-heating time (24 h window)",
         "n/a (rate decaying or no peak)" if peak is None else f"{peak:.2f}", "h"),
        ("Assessment", "Hazard classification", rec["level_title"], ""),
        ("Assessment", "Comment", rec["level_msg"], ""),
        ("Assessment", "Notes", p.get("notes") or "-", ""),
        ("Criteria", "High", "TMR_ad < 8 h OR dT_ad > 100 degC", ""),
        ("Criteria", "Moderate", "8 h <= TMR_ad <= 24 h", ""),
        ("Criteria", "Low", "TMR_ad > 24 h", ""),
        ("Integrity", "Record hash", rec["hash"], "SHA-256 / HMAC"),
        ("Notice", "Copyright", COPYRIGHT_LINE, ""),
        ("Notice", "Disclaimer", "Screening-level estimate only. Not a substitute for calorimetric testing, "
         "a process hazard analysis or competent professional judgement. Does not certify HSE/COMAH compliance.", ""),
    ]


def build_csv(rec: dict) -> bytes:
    df = pd.DataFrame(report_rows(rec), columns=["Section", "Parameter", "Value", "Unit"])
    return df.to_csv(index=False).encode("utf-8-sig")


_PDF_MAP = {"—": "-", "–": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "•": "-", "Δ": "d",
            "ρ": "rho", "≤": "<=", "≥": ">=", "→": "->", "₦": "NGN ", "…": "...", " ": " "}


def pdf_safe(s) -> str:
    s = str(s)
    for k, v in _PDF_MAP.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap(s: str, width_pt: float, size: float, bold: bool = False) -> list:
    chars = max(8, int(width_pt / (size * (0.57 if bold else 0.54))))
    return textwrap.wrap(pdf_safe(s), chars) or [""]


class PdfDoc:
    """Tiny PDF writer: pages of vector graphics and Helvetica text (A4)."""
    W, H = 595, 842

    def __init__(self):
        self.pages: list = []
        self.ops: list = []

    def new_page(self):
        self.ops = []
        self.pages.append(self.ops)

    @staticmethod
    def _c(rgb):
        return "%.3f %.3f %.3f" % tuple(v / 255.0 for v in rgb)

    def rect(self, x, y, w, h, fill=None, stroke=None, lw=0.5):
        if fill:
            self.ops.append(f"{self._c(fill)} rg {x:.2f} {y:.2f} {w:.2f} {h:.2f} re f")
        if stroke:
            self.ops.append(f"{self._c(stroke)} RG {lw} w {x:.2f} {y:.2f} {w:.2f} {h:.2f} re S")

    def line(self, x1, y1, x2, y2, color=(0, 0, 0), lw=0.5, dash=None):
        d = f"[{dash[0]} {dash[1]}] 0 d" if dash else "[] 0 d"
        self.ops.append(f"{d} {self._c(color)} RG {lw} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S [] 0 d")

    def poly(self, pts, color=(0, 0, 0), lw=1.2):
        if len(pts) < 2:
            return
        path = f"{pts[0][0]:.2f} {pts[0][1]:.2f} m " + " ".join(f"{x:.2f} {y:.2f} l" for x, y in pts[1:])
        self.ops.append(f"[] 0 d {self._c(color)} RG {lw} w 1 j 1 J {path} S")

    def text(self, x, y, s, size=9, bold=False, color=(0, 0, 0)):
        font = "F2" if bold else "F1"
        self.ops.append(f"BT /{font} {size} Tf {self._c(color)} rg {x:.2f} {y:.2f} Td ({_esc(pdf_safe(s))}) Tj ET")

    def paragraph(self, x, y, s, width, size=9, bold=False, color=(0, 0, 0), leading=None) -> float:
        leading = leading or size * 1.35
        for ln in _wrap(s, width, size, bold):
            self.text(x, y, ln, size, bold, color)
            y -= leading
        return y

    def render(self, title: str, author: str) -> bytes:
        n = len(self.pages)
        objs = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            ("<< /Type /Pages /Kids [" + " ".join(f"{7 + 2 * i} 0 R" for i in range(n)) +
             f"] /Count {n} >>").encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
            f"<< /Title ({_esc(pdf_safe(title))}) /Author ({_esc(pdf_safe(author))}) "
            f"/Producer ({_esc(pdf_safe(APP_NAME))}) >>".encode("latin-1", "replace"),
        ]
        for i, ops in enumerate(self.pages):
            stream = "\n".join(ops).encode("latin-1", "replace")
            objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
            objs.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.W} {self.H}] "
                         f"/Contents {6 + 2 * i} 0 R /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> >>").encode())
        pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        offsets = []
        for num, body in enumerate(objs, start=1):
            offsets.append(len(pdf))
            pdf += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
        xref = len(pdf)
        pdf += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
        pdf += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
        pdf += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R /Info 5 0 R >>\n"
                f"startxref\n{xref}\n%%EOF").encode()
        return pdf


NAVY = (16, 42, 67)
GREY = (90, 98, 108)
LIGHT = (243, 245, 248)
LEVEL_COLORS = {"high": (192, 38, 38), "moderate": (202, 138, 4), "low": (22, 128, 61)}
LEVEL_TINT = {"high": (253, 236, 236), "moderate": (254, 247, 224), "low": (230, 246, 235)}


def build_pdf(rec: dict) -> bytes:
    p, res, rows = rec["params"], rec["res"], report_rows(rec)
    doc = PdfDoc()
    LM, RM = 40, 555
    CW = RM - LM
    color = LEVEL_COLORS[rec["level"]]

    def header(first: bool):
        doc.rect(0, 772 if first else 800, 595, 70 if first else 42, fill=NAVY)
        if first:
            doc.text(LM, 812, "REACTION HAZARD & RUNAWAY SCREENING RECORD", 15, True, (255, 255, 255))
            doc.text(LM, 794, "Adiabatic thermal risk screening  |  Prepared for HSE / COMAH laboratory safety records",
                     8.5, False, (200, 215, 230))
        else:
            doc.text(LM, 815, "REACTION HAZARD & RUNAWAY SCREENING RECORD (continued)", 11, True, (255, 255, 255))

    def table(y, title, section):
        doc.text(LM, y, title, 10, True, NAVY)
        doc.line(LM, y - 4, RM, y - 4, NAVY, 1)
        y -= 17
        for i, (sec, param, value, unit) in enumerate(rows):
            if sec != section:
                continue
            if i % 2 == 0:
                doc.rect(LM, y - 3.5, CW, 13.5, fill=LIGHT)
            doc.text(LM + 4, y, param, 8.5)
            doc.text(LM + 262, y, textwrap.shorten(pdf_safe(value), 42, placeholder="..."), 8.5, True)
            doc.text(LM + 430, y, unit, 8, False, GREY)
            y -= 13.5
        return y - 8

    # ---- Page 1 --------------------------------------------------------------
    doc.new_page()
    header(True)
    y = 748
    for label, val in (("Report ID", rec["report_id"]),
                       ("Generated (UTC)", rec["created"].strftime("%Y-%m-%d %H:%M:%S")),
                       ("Reaction / batch", p["name"]),
                       ("Prepared by", f"{rec['user_name']} ({rec['user_email']})"),
                       ("Tool", f"{APP_SHORT} v{APP_VERSION}")):
        doc.text(LM, y, label, 8.5, True, GREY)
        doc.text(LM + 100, y, textwrap.shorten(pdf_safe(val), 85, placeholder="..."), 9, False)
        y -= 13.5

    y -= 6
    box_h = 74
    doc.rect(LM, y - box_h, CW, box_h, fill=LEVEL_TINT[rec["level"]])
    doc.rect(LM, y - box_h, 6, box_h, fill=color)
    doc.text(LM + 16, y - 20, "HAZARD CLASSIFICATION: " + rec["level_title"], 12.5, True, color)
    doc.paragraph(LM + 16, y - 37, rec["level_msg"], CW - 30, 8.5, False, (40, 40, 40))
    y -= box_h + 14

    card_w = (CW - 20) / 3
    cards = [("dT_ad (deg C)", f"{res['dt_ad']:.1f}"),
             ("T_max (deg C)", f"{res['t_max']:.1f}"),
             ("TMR_ad (hours)", tmr_display(res["tmr_h"]))]
    for i, (lab, val) in enumerate(cards):
        x = LM + i * (card_w + 10)
        doc.rect(x, y - 52, card_w, 52, fill=(255, 255, 255), stroke=(200, 205, 212), lw=0.8)
        doc.text(x + 10, y - 17, lab, 8.5, False, GREY)
        doc.text(x + 10, y - 42, val, 20, True, NAVY)
    y -= 70

    y = table(y, "INPUT PARAMETERS", "Inputs")
    y = table(y, "CALCULATED RESULTS", "Results")
    if p.get("notes"):
        doc.text(LM, y, "NOTES", 10, True, NAVY)
        y = doc.paragraph(LM + 4, y - 15, p["notes"], CW - 8, 8.5)

    # ---- Page 2 --------------------------------------------------------------
    doc.new_page()
    header(False)
    y = 775
    doc.text(LM, y, "ADIABATIC SELF-HEATING PROFILE (24 h)", 10, True, NAVY)
    doc.line(LM, y - 4, RM, y - 4, NAVY, 1)
    px, py, pw, ph = LM + 40, y - 270, CW - 55, 245
    traj = rec["traj"]
    ymin = math.floor((min(p["t_op"], traj["temp_c"].min()) - 5) / 10.0) * 10
    ymax = math.ceil((max(res["t_max"], traj["temp_c"].max()) + 5) / 10.0) * 10
    if ymax - ymin < 20:
        ymax = ymin + 20

    def mx(t):
        return px + pw * (t / WINDOW_H)

    def my(v):
        return py + ph * ((v - ymin) / (ymax - ymin))

    doc.rect(px, py, pw, ph, fill=(255, 255, 255), stroke=(120, 126, 134), lw=0.8)
    for hr in range(0, 25, 4):
        doc.line(mx(hr), py, mx(hr), py + ph, (225, 228, 232), 0.4)
        doc.text(mx(hr) - 4, py - 11, str(hr), 7.5, False, GREY)
    step = (ymax - ymin) / 5.0
    for k in range(6):
        v = ymin + k * step
        doc.line(px, my(v), px + pw, my(v), (225, 228, 232), 0.4)
        doc.text(px - 30, my(v) - 3, f"{v:.0f}", 7.5, False, GREY)
    doc.text(px + pw / 2 - 25, py - 25, "Time (h)", 8.5, True, GREY)
    doc.text(LM - 2, py + ph + 8, "Temperature (deg C)", 8.5, True, GREY)
    doc.line(px, my(p["t_op"]), px + pw, my(p["t_op"]), (31, 119, 180), 0.9, (4, 3))
    doc.line(px, my(res["t_max"]), px + pw, my(res["t_max"]), (127, 127, 127), 0.9, (2, 3))
    if not math.isinf(res["tmr_h"]) and res["tmr_h"] <= WINDOW_H:
        doc.line(mx(res["tmr_h"]), py, mx(res["tmr_h"]), py + ph, (255, 127, 14), 0.9, (6, 3))
    idx = np.unique(np.linspace(0, len(traj) - 1, min(len(traj), 500)).astype(int))
    pts = [(mx(float(traj["time_h"].iloc[i])), my(float(traj["temp_c"].iloc[i]))) for i in idx]
    doc.poly(pts, (214, 39, 40), 1.6)
    ly = py - 42
    doc.line(px, ly + 3, px + 18, ly + 3, (214, 39, 40), 1.6)
    doc.text(px + 22, ly, "Adiabatic temperature", 7.5)
    doc.line(px + 125, ly + 3, px + 143, ly + 3, (31, 119, 180), 0.9, (4, 3))
    doc.text(px + 147, ly, f"T_op = {p['t_op']:.1f} C", 7.5)
    doc.line(px + 225, ly + 3, px + 243, ly + 3, (127, 127, 127), 0.9, (2, 3))
    doc.text(px + 247, ly, f"T_max = {res['t_max']:.1f} C", 7.5)
    if not math.isinf(res["tmr_h"]) and res["tmr_h"] <= WINDOW_H:
        doc.line(px + 325, ly + 3, px + 343, ly + 3, (255, 127, 14), 0.9, (6, 3))
        doc.text(px + 347, ly, f"TMR_ad = {res['tmr_h']:.1f} h", 7.5)
    y = ly - 26

    doc.text(LM, y, "CLASSIFICATION CRITERIA", 10, True, NAVY)
    doc.line(LM, y - 4, RM, y - 4, NAVY, 1)
    y -= 18
    for lab, txt, c in (("HIGH", "TMR_ad < 8 h  OR  dT_ad > 100 deg C", LEVEL_COLORS["high"]),
                        ("MODERATE", "8 h <= TMR_ad <= 24 h", LEVEL_COLORS["moderate"]),
                        ("LOW", "TMR_ad > 24 h (and dT_ad <= 100 deg C)", LEVEL_COLORS["low"])):
        doc.rect(LM, y - 2, 8, 8, fill=c)
        doc.text(LM + 14, y, lab, 8.5, True)
        doc.text(LM + 80, y, txt, 8.5)
        y -= 13
    y -= 10

    doc.text(LM, y, "ASSUMPTIONS & LIMITATIONS", 10, True, NAVY)
    doc.line(LM, y - 4, RM, y - 4, NAVY, 1)
    y -= 18
    for item in rec["warnings"] + [
        "Adiabatic conditions (no heat loss), single exothermic reaction, no decomposition or secondary reactions, "
        "perfectly mixed batch, constant Cp and density.",
        "TMR_ad uses the Semenov-type initial-rate approximation evaluated at the operating temperature. The plotted "
        "curve integrates reactant consumption and may differ from the headline TMR_ad.",
        "Gas generation, vapour pressure, boil-off, accumulation of unreacted feed and mixing or cooling failure "
        "scenarios are not modelled.",
        "This record is a screening-level estimate. It must be confirmed by DSC / ARC / RC1 calorimetry and a formal "
        "process hazard analysis before any scale-up decision, and reviewed by a competent person. It does not by "
        "itself demonstrate compliance with HSE, COMAH or any other regulation.",
    ]:
        doc.text(LM + 1, y, "-", 8.2)
        y = doc.paragraph(LM + 10, y, item, CW - 14, 8.2, leading=10.8)
        y -= 3
    y -= 8

    doc.text(LM, y, "SIGN-OFF", 10, True, NAVY)
    doc.line(LM, y - 4, RM, y - 4, NAVY, 1)
    y -= 26
    for role in ("Assessed by", "Reviewed by", "Approved by"):
        doc.text(LM, y, role + ":", 8.5, True)
        doc.line(LM + 70, y - 2, LM + 250, y - 2, (80, 80, 80), 0.6)
        doc.text(LM + 270, y, "Date:", 8.5, True)
        doc.line(LM + 300, y - 2, LM + 400, y - 2, (80, 80, 80), 0.6)
        y -= 24

    total = len(doc.pages)
    for i, ops in enumerate(doc.pages, start=1):
        doc.ops = ops
        doc.line(LM, 50, RM, 50, (190, 195, 201), 0.5)
        doc.text(LM, 38, f"{COPYRIGHT_LINE}  {APP_SHORT} v{APP_VERSION}. Screening only - see limitations.", 6.8, False, GREY)
        doc.text(LM, 28, f"Report ID {rec['report_id']}   |   Record hash {rec['hash']}", 6.2, False, GREY)
        doc.text(RM - 42, 38, f"Page {i} of {total}", 7, False, GREY)
    return doc.render(f"{APP_SHORT} - {rec['report_id']}", OWNER)


# =============================================================================
# 8. LEGAL TEXT  (templates - have a Nigerian lawyer review before relying on them)
# =============================================================================
def legal_terms() -> str:
    return f"""
**Terms of Use** · version {TERMS_VERSION}

These Terms are a binding agreement between you and **{OWNER}** ("we", "us", the "Owner") for the use of the
**{APP_NAME}** (the "Service"). By creating an account or using the Service you accept them.

**1. Eligibility.** You must be at least 18 and use the Service for professional, business, research or educational
purposes. If you use it on behalf of an organisation you confirm you have authority to bind it.

**2. Licence.** Subject to these Terms and payment of any applicable fees, we grant you a limited, revocable,
non-exclusive, non-transferable, non-sublicensable licence to access and use the Service for your own internal purposes.

**3. Ownership and intellectual property.** The Service, including its source code, calculation implementation,
user interface, report templates, text, graphics, documentation and name, is owned by the Owner and protected by
copyright and other laws. All rights are reserved. Except for the limited licence above, nothing in these Terms
transfers any right to you. Data you enter remains yours; you permit us to process it to provide the Service, keep
audit records of your assessments and, in anonymised form, to maintain and improve the Service.

**4. Restrictions.** You must not: (a) copy, modify, translate or create derivative works of the Service;
(b) reverse engineer, decompile or attempt to extract source code or algorithms, except to the extent the law
forbids this restriction; (c) scrape, crawl or access the Service by automated means or through an API we have not
provided; (d) resell, sublicense, white-label, host for third parties or otherwise commercialise the Service or its
outputs as a competing product; (e) share your account or credentials; (f) remove or alter copyright notices,
report identifiers or integrity hashes, or present an altered report as one issued by the Service; (g) circumvent
usage limits, plan restrictions or payment; (h) use the Service unlawfully or to harm others.

**5. Accounts.** Keep your credentials confidential. You are responsible for activity under your account and must
tell us promptly of any unauthorised use. Accounts that repeatedly fail to log in are temporarily locked.

**6. Plans and payment.** The Free plan is limited to {FREE_DAILY_RUNS} assessments per day and CSV export. The Pro
plan costs {ngn(PRO_PRICE_NGN)} for {PRO_DAYS} days of access (a one-off payment that does not renew automatically)
and removes the daily limit, adds PDF compliance reports and full history. Payment is made by bank transfer or, where enabled,
through Paystack; we never see or store your card details. Because the Service is delivered digitally and immediately, fees are
non-refundable except where the law requires otherwise or where we failed to provide the Service. We may change
prices for future purchases.

**7. Safety-critical use: screening estimates only.** The Service produces simplified, screening-level estimates
based on the inputs you provide and a single-reaction adiabatic model. It is **not** a substitute for calorimetric
testing (DSC, ARC, RC1 or equivalent), a formal process hazard analysis, or the judgement of a competent person. It
does not model gas generation, vapour pressure, secondary decomposition, accumulation, mixing or cooling failure.
Output quality depends entirely on your inputs. You are solely responsible for every safety, engineering,
scale-up and regulatory decision you make, and for obtaining independent professional review. Reports are formatted
to help with record-keeping; they do not certify, and are not evidence of, compliance with HSE, COMAH or any other
regulation.

**8. No warranty.** The Service is provided "as is" and "as available". To the maximum extent permitted by law we
disclaim all warranties, express or implied, including accuracy, fitness for a particular purpose, merchantability,
non-infringement and uninterrupted or error-free operation.

**9. Limitation of liability.** To the maximum extent permitted by law, the Owner is not liable for any indirect,
incidental, special, consequential or punitive loss, or for loss of profit, revenue, data, goodwill, or for injury,
death, property damage, fire, explosion or environmental harm, arising from or relating to your use of or reliance
on the Service. Our total aggregate liability for any claim shall not exceed the fees you paid us in the 12 months
before the claim arose (or {ngn(1000)} if you paid nothing). Nothing in these Terms excludes or limits liability that
cannot lawfully be excluded or limited.

**10. Indemnity.** You will indemnify and hold the Owner harmless from claims, losses and expenses (including
reasonable legal fees) arising from your breach of these Terms, your misuse of the Service, or decisions or actions
you or your organisation take in reliance on its output.

**11. Suspension and termination.** We may suspend or terminate access immediately if you breach these Terms or
misuse the Service. You may stop using the Service at any time. Sections on ownership, restrictions, disclaimers,
liability, indemnity and governing law survive termination.

**12. Changes.** We may update the Service and these Terms. Material changes will take effect when posted in the
Service; continued use after that means you accept them.

**13. Governing law.** These Terms are governed by the laws of the Federal Republic of Nigeria. The courts of
Lagos State have exclusive jurisdiction, subject to any mandatory rights you have under the law of your country.

**14. Contact.** {OWNER} · {CONTACT_EMAIL}
"""


def legal_privacy() -> str:
    return f"""
**Privacy Notice** · version {TERMS_VERSION}

**Data controller:** {OWNER}, {CONTACT_EMAIL}. We handle personal data in line with the Nigeria Data Protection Act 2023.

**What we collect.** Your name and email; a salted, hashed password (we cannot read your password); the inputs,
results and report identifiers of each assessment you run; payment references and the bank-transfer details you submit (card details, if you
pay online, are handled only by Paystack); account timestamps; and basic technical logs kept by our hosting providers.

**Why we use it.** To create and secure your account, provide the Service, enforce plan limits, keep an audit trail
of your assessments, process payments, prevent abuse, meet legal obligations and support you.
Our legal bases are performance of our contract with you, our legitimate interests in running and protecting the
Service, legal obligation and, where needed, consent.

**Who receives it.** Service providers who help us run the Service: the hosting platform, the database provider and, if
online payment is enabled, Paystack. We do not sell your personal data. We may disclose data where the law requires it.

**Transfers.** Providers may process data outside Nigeria; we require appropriate safeguards.

**Retention.** Account and assessment records are kept while your account is active and for a reasonable period
afterwards for security, accounting and legal purposes, then deleted or anonymised.

**Your rights.** You may ask to access, correct, delete or port your data, to object to or restrict processing, and
to withdraw consent. Email {CONTACT_EMAIL}. You may also complain to the Nigeria Data Protection Commission.

**Security.** Passwords are hashed with PBKDF2-SHA256, repeated failed logins lock the account temporarily, and
payments run through a PCI-compliant processor. No system is perfectly secure.

**Children.** The Service is not intended for anyone under 18.
"""


def legal_disclaimer() -> str:
    return (
        "**Important safety disclaimer.** This tool gives **screening-level estimates** from simplified kinetics and "
        "an adiabatic model. It is not a substitute for DSC / ARC / RC1 calorimetry, a process hazard analysis or "
        "competent professional judgement, and it does not certify HSE, COMAH or other regulatory compliance. "
        "You remain solely responsible for decisions made using its output. See the Terms of Use."
    )


def legal_licence() -> str:
    return f"""
**{COPYRIGHT_LINE}**

**{APP_NAME}**, its source code, calculation implementation, interface, report templates, documentation and name are
the proprietary intellectual property of {OWNER}. Use is permitted only under the Terms of Use. Unauthorised copying,
modification, redistribution, resale, hosting for third parties, scraping or reverse engineering is prohibited and
may result in account termination and legal action.

Every report carries a unique Report ID and a record hash so that its origin and integrity can be verified.

To request a commercial, white-label or enterprise licence, contact {CONTACT_EMAIL}.
"""


# =============================================================================
# 9. USER INTERFACE
# =============================================================================
CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1200px;}
.plan-badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem;
             font-weight:600; letter-spacing:.03em;}
.plan-pro {background:#d1fae5; color:#065f46;}
.plan-free {background:#e5e7eb; color:#374151;}
.site-footer {text-align:center; color:#6b7280; font-size:.8rem; margin-top:2.5rem;
              padding-top:1rem; border-top:1px solid rgba(128,128,128,.25);}
</style>
"""


def stretch(fn, *args, **kwargs):
    """Call a Streamlit widget full-width, across Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def footer():
    st.markdown(
        f"<div class='site-footer'>{COPYRIGHT_LINE} · {APP_NAME} v{APP_VERSION}<br>"
        "Screening-level estimates only; not a substitute for calorimetric testing or professional judgement.</div>",
        unsafe_allow_html=True)


def legal_tabs():
    t1, t2, t3, t4 = st.tabs(["Terms of Use", "Privacy Notice", "Disclaimer", "Copyright & Licence"])
    with t1:
        st.markdown(legal_terms())
    with t2:
        st.markdown(legal_privacy())
    with t3:
        st.markdown(legal_disclaimer())
    with t4:
        st.markdown(legal_licence())


def render_auth():
    st.title("⚠️ " + APP_NAME)
    st.caption("Real-time adiabatic thermal risk screening for batch reaction scale-up, "
               "built for laboratories, plants and process development engineers.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            tab_login, tab_signup = st.tabs(["Log in", "Create account"])
            with tab_login:
                with st.form("login_form"):
                    email = st.text_input("Email", key="li_email")
                    pw = st.text_input("Password", type="password", key="li_pw")
                    go_login = st.form_submit_button("Log in", type="primary")
                if go_login:
                    user, msg = authenticate(email, pw)
                    if user:
                        st.session_state["uid"] = user["id"]
                        st.rerun()
                    else:
                        st.error(msg)
                st.caption("Forgot your password? Email " + CONTACT_EMAIL + " from your registered address.")
            with tab_signup:
                with st.form("signup_form"):
                    name = st.text_input("Full name", key="su_name")
                    email2 = st.text_input("Email", key="su_email")
                    pw1 = st.text_input("Password (min 8 characters, letters and numbers)", type="password", key="su_pw1")
                    pw2 = st.text_input("Confirm password", type="password", key="su_pw2")
                    agree = st.checkbox("I am 18 or older and I accept the Terms of Use, Privacy Notice and "
                                        "Safety Disclaimer below.", key="su_agree")
                    go_signup = st.form_submit_button("Create free account", type="primary")
                if go_signup:
                    if pw1 != pw2:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = register_user(email2, name, pw1, agree)
                        (st.success if ok else st.error)(msg)
    with right:
        with st.container(border=True):
            st.subheader("Plans")
            st.markdown(
                f"**Free** · {ngn(0)}\n- {FREE_DAILY_RUNS} assessments per day\n- Hazard classification and chart\n"
                f"- CSV record export\n\n**Pro** · {ngn(PRO_PRICE_NGN)} / {PRO_DAYS} days\n- Unlimited assessments\n"
                "- Professional PDF compliance report\n- Full history and verification of past reports")
    st.info(legal_disclaimer())
    with st.expander("Terms of Use, Privacy Notice and Copyright"):
        legal_tabs()
    footer()


def sidebar_account(container, user: dict):
    plan = plan_of(user)
    with container:
        badge = f"<span class='plan-badge plan-{plan}'>{plan.upper()}</span>"
        st.markdown(f"**{user['name']}**  \n{user['email']}  \n{badge}", unsafe_allow_html=True)
        if plan == "pro":
            if user["email"] in admin_emails():
                st.caption("Owner account: full access.")
            elif user["plan_expires"]:
                st.caption(f"Pro active until {user['plan_expires']:%d %b %Y}.")
        else:
            used = runs_today(user["id"])
            left = max(0, FREE_DAILY_RUNS - used)
            st.caption(f"{left} of {FREE_DAILY_RUNS} free assessments left today (resets at midnight WAT).")
        if st.button("Log out", key="logout"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


def sidebar_form():
    st.header("Reaction Inputs")
    with st.form("inputs"):
        name = st.text_input("Reaction Name / Batch ID", value="Batch-001", max_chars=200)
        st.subheader("Thermodynamics")
        dh = st.number_input("Reaction enthalpy ΔH_rxn (kJ/mol)", value=-120.0, step=5.0, format="%.2f")
        c0 = st.number_input("Initial concentration C0 (mol/L)", value=2.0, step=0.1, format="%.3f")
        cp = st.number_input("Specific heat capacity Cp (J/g·°C)", value=3.5, step=0.1, format="%.3f")
        rho = st.number_input("Mixture density ρ (kg/m³)", value=1000.0, step=10.0, format="%.1f")
        st.subheader("Kinetics")
        ea = st.number_input("Activation energy Ea (kJ/mol)", value=85.0, step=1.0, format="%.2f")
        a_text = st.text_input("Pre-exponential factor A (1/s or L/mol·s)", value="1e8",
                               help="Scientific notation accepted, e.g. 1e8 or 2.5E10")
        order = st.radio("Reaction order (n)", [1, 2], horizontal=True,
                         help="1: rate = k·C (A in 1/s). 2: rate = k·C² (A in L/mol·s).")
        st.subheader("Process")
        t_op = st.number_input("Operating temperature T_op (°C)", value=50.0, step=1.0, format="%.2f")
        notes = st.text_area("Notes (optional, appears on the report)", max_chars=300, height=70)
        submitted = st.form_submit_button("Run screening", type="primary")
    params = {"name": name.strip(), "dh": dh, "c0": c0, "cp": cp, "rho": rho, "ea": ea,
              "a": parse_float(a_text), "order": order, "t_op": t_op, "notes": notes.strip()}
    return submitted, params


def handle_run(user: dict, params: dict):
    st.session_state.pop("run_errors", None)
    st.session_state.pop("quota_hit", None)
    errors = validate_inputs(params)
    if errors:
        st.session_state["run_errors"] = errors
        return
    if plan_of(user) == "free" and runs_today(user["id"]) >= FREE_DAILY_RUNS:
        st.session_state["quota_hit"] = True
        return
    res = calculate(params)
    level, title, msg = classify(res["tmr_h"], res["dt_ad"])
    traj = simulate_adiabatic(params, res)
    report_id, digest, created = save_assessment(user, params, res, level)
    st.session_state["last"] = {
        "params": params, "res": res, "level": level, "level_title": title, "level_msg": msg,
        "warnings": sanity_warnings(params, res), "traj": traj, "peak_h": peak_heating_time(traj),
        "report_id": report_id, "hash": digest, "created": created,
        "user_name": user["name"], "user_email": user["email"],
    }


def tab_screening(user: dict):
    plan = plan_of(user)
    if st.session_state.get("run_errors"):
        with st.container(border=True):
            st.error("**Input validation failed. Please correct the following:**")
            for e in st.session_state["run_errors"]:
                st.markdown(f"- {e}")
        return
    if st.session_state.get("quota_hit"):
        st.warning(f"You have used all {FREE_DAILY_RUNS} free assessments for today. "
                   "Upgrade to Pro for unlimited assessments and PDF reports (see the Upgrade tab), "
                   "or come back after midnight WAT.")
        return
    rec = st.session_state.get("last")
    if not rec:
        with st.container(border=True):
            st.markdown("### Start a screening")
            st.markdown("Enter your reaction data in the **sidebar** and press **Run screening**. "
                        "Each run is saved as a numbered, verifiable record for your safety files.")
            st.info(legal_disclaimer())
        return

    res, p = rec["res"], rec["params"]
    with st.container(border=True):
        st.subheader(f"Batch: {p['name']}")
        box = {"high": st.error, "moderate": st.warning, "low": st.success}[rec["level"]]
        icon = {"high": "🔴", "moderate": "🟡", "low": "🟢"}[rec["level"]]
        box(f"{icon} **{rec['level_title']}**\n\n{rec['level_msg']}")
        for w in rec["warnings"]:
            st.warning(w, icon="⚠️")
        st.caption(f"Report ID {rec['report_id']} · {rec['created']:%Y-%m-%d %H:%M} UTC")

    m1, m2, m3 = st.columns(3)
    with m1, st.container(border=True):
        st.metric("ΔT_ad (°C)", f"{res['dt_ad']:.1f}")
    with m2, st.container(border=True):
        st.metric("T_max (°C)", f"{res['t_max']:.1f}")
    with m3, st.container(border=True):
        st.metric("TMR_ad (hours)", tmr_display(res["tmr_h"]))

    with st.expander("Intermediate kinetic parameters"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("T_op (K)", f"{res['t_k']:.2f}")
        c2.metric("k(T_op)", f"{res['k_top']:.3e}")
        c3.metric("Rate (mol/L·s)", f"{res['rate']:.3e}")
        c4.metric("q(T_op) (W/L)", f"{res['q_vol']:.3e}")

    traj = rec["traj"]
    with st.container(border=True):
        st.subheader("Adiabatic Self-Heating Profile (24 h window)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=traj["time_h"], y=traj["temp_c"], mode="lines", name="Adiabatic temperature",
                                 line=dict(width=3, color="#d62728"),
                                 hovertemplate="t = %{x:.2f} h<br>T = %{y:.1f} °C<extra></extra>"))
        fig.add_hline(y=p["t_op"], line_dash="dash", line_color="#1f77b4",
                      annotation_text="T_op", annotation_position="bottom right")
        fig.add_hline(y=res["t_max"], line_dash="dot", line_color="#7f7f7f",
                      annotation_text="T_max (adiabatic)", annotation_position="top right")
        if not math.isinf(res["tmr_h"]) and res["tmr_h"] <= WINDOW_H:
            fig.add_vline(x=res["tmr_h"], line_dash="dashdot", line_color="#ff7f0e",
                          annotation_text=f"TMR_ad = {res['tmr_h']:.1f} h")
        fig.update_layout(xaxis_title="Time (h)", yaxis_title="Temperature (°C)", height=460,
                          margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified", template="plotly_white")
        fig.update_xaxes(range=[0, WINDOW_H])
        stretch(st.plotly_chart, fig)
        if rec["peak_h"] is not None:
            st.caption(f"Simulated peak self-heating rate (with reactant consumption) at about {rec['peak_h']:.2f} h. "
                       "The headline TMR_ad uses the initial-rate approximation, so the two can differ.")
        else:
            st.caption("No self-heating acceleration peak within 24 h (rate is decaying or negligible at T_op).")

    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", p["name"]) or "batch"
    with st.container(border=True):
        st.subheader("Compliance Report Export")
        d1, d2 = st.columns(2)
        with d1:
            if plan == "pro":
                stretch(st.download_button, "📄 Download PDF report", data=build_pdf(rec),
                        file_name=f"{rec['report_id']}_{safe_id}.pdf", mime="application/pdf")
            else:
                stretch(st.button, "🔒 PDF report (Pro)", disabled=True)
        with d2:
            stretch(st.download_button, "📊 Download CSV report", data=build_csv(rec),
                    file_name=f"{rec['report_id']}_{safe_id}.csv", mime="text/csv")
        if plan != "pro":
            st.caption("PDF compliance reports are included in Pro. See the Upgrade tab.")
        st.caption("Reports carry a unique Report ID and record hash. They are formatted for HSE / COMAH laboratory "
                   "safety files but do not themselves demonstrate compliance.")


def tab_history(user: dict):
    plan = plan_of(user)
    limit = 500 if plan == "pro" else 5
    df = list_assessments(user["id"], limit)
    with st.container(border=True):
        st.subheader("My assessments")
        if df.empty:
            st.info("No assessments yet. Run your first screening from the sidebar.")
        else:
            stretch(st.dataframe, df, hide_index=True)
            if plan == "pro":
                stretch(st.download_button, "Download history (CSV)", data=df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="assessment_history.csv", mime="text/csv")
            else:
                st.caption(f"Free plan shows your latest {limit} records. Pro keeps full history.")
    with st.container(border=True):
        st.subheader("Verify a report")
        st.caption("Enter a Report ID to confirm it was issued by this system and has not been altered.")
        rid = st.text_input("Report ID", placeholder="RHS-20261017-ABCD1234", key="verify_id")
        if rid.strip():
            out = verify_report(user["id"], rid)
            if out is None:
                st.error("No record with that ID was found on your account.")
            elif out["ok"]:
                st.success(f"Verified: issued {out['created']:%Y-%m-%d %H:%M} UTC for batch '{out['batch']}'. "
                           f"Record hash {out['hash'][:16]}…")
            else:
                st.error("Record failed integrity verification.")


def tab_upgrade(user: dict):
    plan = plan_of(user)
    with st.container(border=True):
        st.subheader("Your plan")
        if plan == "pro":
            if user["email"] in admin_emails():
                st.success("Owner account: full Pro access.")
            else:
                st.success(f"Pro is active until {user['plan_expires']:%d %b %Y}. "
                           "You can extend it at any time; days are added to your current expiry.")
        else:
            st.info("You are on the Free plan.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Free**\n- {FREE_DAILY_RUNS} assessments/day\n- CSV export\n- Last 5 records")
        with c2:
            st.markdown(f"**Pro · {ngn(PRO_PRICE_NGN)} for {PRO_DAYS} days**\n- Unlimited assessments\n"
                        "- PDF compliance reports\n- Full history + CSV")
    if bank_details_text():
        with st.container(border=True):
            st.subheader("Pay by bank transfer")
            st.markdown(f"Transfer **{ngn(PRO_PRICE_NGN)}** (or the equivalent for international accounts) to:")
            st.info(bank_details_text())
            st.markdown(f"Use **{user['email']}** as the payment reference where possible. After paying, tell us "
                        "below so we can confirm it and activate Pro.")
            with st.form("transfer_form", clear_on_submit=True):
                payer = st.text_input("Name on the account you paid from", max_chars=120)
                tref = st.text_input("Transfer reference / session ID, or date, time and bank", max_chars=160)
                tnote = st.text_input("Note (optional)", max_chars=300)
                sent = st.form_submit_button("I have paid: notify", type="primary")
            if sent:
                ok, msg = submit_transfer_claim(user, payer, tref, tnote)
                (st.success if ok else st.error)(msg)
            claims = list_user_claims(user["id"])
            if not claims.empty:
                st.caption("Your payment notices")
                stretch(st.dataframe, claims, hide_index=True)
    if paystack_enabled():
        with st.container(border=True):
            if st.button(f"Pay {ngn(PRO_PRICE_NGN)} securely with Paystack", type="primary", key="pay"):
                url, msg = start_payment(user)
                if url:
                    st.session_state["pay_url"] = url
                else:
                    st.error(msg)
            if st.session_state.get("pay_url"):
                stretch(st.link_button, "Continue to Paystack ➜", st.session_state["pay_url"])
                st.caption("After paying you will return here. If Pro is not active yet, press Verify below.")
            pending = latest_pending_reference(user["id"])
            if pending and st.button("I have paid: verify my payment", key="verify_pay"):
                ok, msg = confirm_payment(user, pending)
                (st.success if ok else st.warning)(msg)
                if ok:
                    st.rerun()
    elif not bank_details_text():
        with st.container(border=True):
            st.markdown(f"Online payment is not enabled yet. To upgrade, email **{CONTACT_EMAIL}** from your "
                        "registered address and Pro will be activated after payment.")
    st.caption("Pro is a one-off 30-day pass and does not renew automatically.")


def tab_admin(user: dict):
    if not is_admin(user):
        return
    passphrase = cfg("ADMIN_PASSPHRASE")
    if not passphrase:
        st.warning("Set ADMIN_PASSPHRASE in your secrets to enable the admin panel.")
        return
    if not st.session_state.get("admin_ok"):
        with st.form("admin_unlock"):
            entered = st.text_input("Admin passphrase", type="password")
            if st.form_submit_button("Unlock admin panel"):
                if hmac.compare_digest(entered.encode(), str(passphrase).encode()):
                    st.session_state["admin_ok"] = True
                    st.rerun()
                else:
                    st.error("Incorrect passphrase.")
        return
    s = admin_stats()
    c = st.columns(5)
    c[0].metric("Users", s["users"])
    c[1].metric("Active Pro", s["pro"])
    c[2].metric("Assessments", s["runs"])
    c[3].metric("Today", s["today"])
    c[4].metric("Revenue (NGN)", f"{s['revenue_ngn']:,}")
    with st.container(border=True):
        st.subheader("Manage a user")
        email = st.text_input("User email", key="adm_email")
        a, b, d = st.columns(3)
        days = a.number_input("Days of Pro", min_value=1, max_value=3650, value=30, key="adm_days")
        if a.button("Grant Pro"):
            (st.success if (r := admin_grant(email, int(days)))[0] else st.error)(r[1])
        if b.button("Revoke Pro"):
            (st.success if (r := admin_revoke(email))[0] else st.error)(r[1])
        newpw = d.text_input("Temporary password", type="password", key="adm_pw")
        if d.button("Reset password"):
            (st.success if (r := admin_reset_password(email, newpw))[0] else st.error)(r[1])
    with st.container(border=True):
        st.subheader("Bank-transfer payment notices")
        st.caption("Check your bank app first. Approve only when the money has actually arrived.")
        claims = pending_claims()
        if not claims:
            st.info("No pending notices.")
        for c in claims:
            with st.container(border=True):
                st.markdown(f"**{c['email']}** · payer: {c['payer_name']} · ref: `{c['transfer_ref']}` · "
                            f"{c['created_at']:%Y-%m-%d %H:%M} UTC" + (f"  \nNote: {c['note']}" if c["note"] else ""))
                a1, a2, _ = st.columns([1, 1, 4])
                if a1.button("Approve", key=f"appr_{c['id']}", type="primary"):
                    (st.success if (r := decide_claim(c["id"], True))[0] else st.error)(r[1])
                    st.rerun()
                if a2.button("Reject", key=f"rej_{c['id']}"):
                    (st.success if (r := decide_claim(c["id"], False))[0] else st.error)(r[1])
                    st.rerun()
    with st.container(border=True):
        st.subheader("Users")
        udf = admin_users_df()
        stretch(st.dataframe, udf, hide_index=True)
        if not udf.empty:
            st.download_button("Export users (CSV)", udf.to_csv(index=False).encode("utf-8-sig"), "users.csv", "text/csv")
    with st.container(border=True):
        st.subheader("Payments")
        stretch(st.dataframe, admin_payments_df(), hide_index=True)


def process_payment_callback(user: dict):
    """Paystack redirects back with ?reference=...; confirm it for the logged-in owner of the payment."""
    ref = st.session_state.pop("pending_reference", None)
    qp_ref = st.query_params.get("reference") or st.query_params.get("trxref")
    ref = qp_ref or ref
    if not ref:
        return
    ok, msg = confirm_payment(user, str(ref))
    try:
        st.query_params.clear()
    except Exception:
        pass
    (st.success if ok else st.warning)(msg)


def render_app(user: dict):
    process_payment_callback(user)
    acct = st.sidebar.container()
    with st.sidebar:
        st.divider()
        submitted, params = sidebar_form()
    if submitted:
        handle_run(user, params)
    user = get_user(user["id"]) or user
    sidebar_account(acct, user)

    st.title("⚠️ " + APP_NAME)
    st.caption("Real-time adiabatic thermal risk screening for batch reaction scale-up.")
    names = ["Screening", "My assessments", "Upgrade", "Legal"] + (["Admin"] if is_admin(user) else [])
    tabs = st.tabs(names)
    with tabs[0]:
        tab_screening(user)
    with tabs[1]:
        tab_history(user)
    with tabs[2]:
        tab_upgrade(user)
    with tabs[3]:
        legal_tabs()
    if is_admin(user):
        with tabs[4]:
            tab_admin(user)
    footer()


def main():
    st.set_page_config(page_title=APP_SHORT, page_icon="⚠️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    if using_ephemeral_db() and cfg("APP_URL"):
        st.warning("Server configuration: DATABASE_URL is not set, so accounts are stored in a temporary local "
                   "file that will be lost on restart. Configure a PostgreSQL database before taking real users.")
    uid = st.session_state.get("uid")
    user = get_user(uid) if uid else None
    if user is None:
        # Keep a payment reference from the Paystack redirect so it can be confirmed after login.
        ref = st.query_params.get("reference") or st.query_params.get("trxref")
        if ref:
            st.session_state["pending_reference"] = ref
            st.info("Log in to confirm your payment.")
        st.session_state.pop("uid", None)
        render_auth()
    else:
        render_app(user)


if __name__ == "__main__":
    main()