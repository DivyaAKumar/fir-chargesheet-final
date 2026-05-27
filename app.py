"""
app.py  —  FIR Chargesheet Generator  (with Blockchain Audit Trail)
====================================================================
All FIR / evidence / chargesheet / status-change events are
immutably recorded on the local SHA-256 blockchain (blockchain.py).
"""

from flask import (Flask, render_template, request, redirect,
                   url_for, flash, session, jsonify)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user,
                         login_required, logout_user, current_user)
from datetime import datetime
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from chargesheet import ChargeSheet
from blockchain import get_blockchain            # ← NEW

app = Flask(__name__)
app.config["SECRET_KEY"]             = os.environ.get("FLASK_SECRET_KEY") or os.urandom(24).hex()
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///fir_system.db"
app.config["UPLOAD_FOLDER"]           = "static/evidence"

db           = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# DB Models
# ──────────────────────────────────────────────────────────────────────────────

class User(UserMixin, db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    username       = db.Column(db.String(80), unique=True, nullable=False)
    password       = db.Column(db.String(120), nullable=False)
    police_station = db.Column(db.String(100), nullable=False)
    designation    = db.Column(db.String(50), nullable=False)


class FIR(db.Model):
    id                  = db.Column(db.Integer, primary_key=True)
    fir_number          = db.Column(db.String(20), unique=True, nullable=False)
    complainant_name    = db.Column(db.String(100), nullable=False)
    complainant_contact = db.Column(db.String(20), nullable=False)
    complainant_address = db.Column(db.String(200), nullable=False)
    incident_date       = db.Column(db.DateTime, nullable=False)
    incident_location   = db.Column(db.String(200), nullable=False)
    description         = db.Column(db.Text, nullable=False)
    status              = db.Column(db.String(20), default="pending")
    is_cognizable       = db.Column(db.Boolean, nullable=True)
    evidence_items      = db.relationship("Evidence", backref="fir", lazy=True)
    created_by          = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    creation_date       = db.Column(db.DateTime, default=datetime.utcnow)
    # blockchain block hash when FIR was created
    blockchain_hash     = db.Column(db.String(64), nullable=True)


class Evidence(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    fir_id      = db.Column(db.Integer, db.ForeignKey("fir.id"), nullable=False)
    description = db.Column(db.Text, nullable=False)
    image_path  = db.Column(db.String(200))
    uploaded_by = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)


class Chargesheet(db.Model):
    id               = db.Column(db.Integer, primary_key=True)
    fir_id           = db.Column(db.Integer, db.ForeignKey("fir.id"), nullable=False)
    ipc_sections     = db.Column(db.String(500), nullable=False)
    summary          = db.Column(db.Text, nullable=False)
    generated_date   = db.Column(db.DateTime, default=datetime.utcnow)
    generated_by     = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    # blockchain block hash when chargesheet was generated
    blockchain_hash  = db.Column(db.String(64), nullable=True)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ──────────────────────────────────────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and check_password_hash(user.password, request.form["password"]):
            login_user(user)
            flash(f"Welcome, {user.username}!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid username or password!", "error")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    firs        = FIR.query.all()
    bc          = get_blockchain()
    chain_stats = bc.chain_stats()
    return render_template("dashboard.html", firs=firs, chain_stats=chain_stats)


# ──────────────────────────────────────────────────────────────────────────────
# Create FIR  — blockchain: register_fir
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/create_fir", methods=["GET", "POST"])
@login_required
def create_fir():
    if request.method == "POST":
        year     = datetime.now().year
        last_fir = (FIR.query
                    .filter(FIR.fir_number.like(f"FIR/{year}/%"))
                    .order_by(FIR.id.desc()).first())
        new_num  = (int(last_fir.fir_number.split("/")[-1]) + 1) if last_fir else 1
        fir_num  = f"FIR/{year}/{new_num:04d}"

        fir = FIR(
            fir_number          = fir_num,
            complainant_name    = request.form["complainant_name"],
            complainant_contact = request.form["complainant_contact"],
            complainant_address = request.form["complainant_address"],
            incident_date       = datetime.strptime(request.form["incident_date"], "%Y-%m-%d"),
            incident_location   = request.form["incident_location"],
            description         = request.form["description"],
            created_by          = current_user.id,
        )
        db.session.add(fir)
        db.session.flush()   # get fir.id before commit

        # ── Blockchain: register FIR ──────────────────────────────────────────
        bc    = get_blockchain()
        block = bc.register_fir(
            fir_id      = fir.id,
            fir_number  = fir_num,
            complainant = fir.complainant_name,
            description = fir.description,
            officer     = current_user.username,
        )
        fir.blockchain_hash = block.hash
        # ─────────────────────────────────────────────────────────────────────

        db.session.commit()
        flash(f"FIR {fir_num} created and registered on blockchain! 🔗 Block #{block.index}", "success")
        return redirect(url_for("dashboard"))

    return render_template("create_fir.html")


# ──────────────────────────────────────────────────────────────────────────────
# View FIR
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/fir/<int:fir_id>")
@login_required
def view_fir(fir_id):
    fir          = FIR.query.get_or_404(fir_id)
    evidence     = Evidence.query.filter_by(fir_id=fir_id).all()
    chargesheet  = Chargesheet.query.filter_by(fir_id=fir_id).first()
    cs_data      = session.get("chargesheet_data") if chargesheet else None

    # Blockchain audit trail for this FIR
    bc           = get_blockchain()
    audit_trail  = bc.get_fir_audit_trail(fir_id)

    return render_template(
        "view_fir.html",
        fir             = fir,
        evidence        = evidence,
        chargesheet     = chargesheet,
        chargesheet_data = cs_data,
        audit_trail     = audit_trail,
        chain_valid     = bc.is_chain_valid(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Update Status  — blockchain: register_status_change
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/fir/<int:fir_id>/update_status", methods=["POST"])
@login_required
def update_fir_status(fir_id):
    fir        = FIR.query.get_or_404(fir_id)
    new_status = request.form["status"]
    old_status = fir.status

    if new_status == "accepted":
        if "is_cognizable" not in request.form:
            flash("Please specify cognizable / non-cognizable!", "error")
            return redirect(url_for("view_fir", fir_id=fir_id))
        fir.is_cognizable = request.form["is_cognizable"] == "true"

    fir.status = new_status

    # ── Blockchain ────────────────────────────────────────────────────────────
    bc = get_blockchain()
    bc.register_status_change(
        fir_id     = fir.id,
        fir_number = fir.fir_number,
        old_status = old_status,
        new_status = new_status,
        changed_by = current_user.username,
    )
    # ─────────────────────────────────────────────────────────────────────────

    db.session.commit()
    flash(f"FIR status updated to '{new_status}' and recorded on blockchain! 🔗", "success")
    return redirect(url_for("view_fir", fir_id=fir_id))


# ──────────────────────────────────────────────────────────────────────────────
# Add Evidence  — blockchain: register_evidence
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/fir/<int:fir_id>/add_evidence", methods=["POST"])
@login_required
def add_evidence(fir_id):
    fir = FIR.query.get_or_404(fir_id)

    if "evidence_image" in request.files:
        file = request.files["evidence_image"]
        if file and file.filename:
            filename  = secure_filename(f"{datetime.utcnow().timestamp()}_{file.filename}")
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            file.save(file_path)

            ev = Evidence(
                fir_id      = fir_id,
                description = request.form.get("evidence_description", "Evidence item"),
                image_path  = filename,
                uploaded_by = current_user.id,
            )
            db.session.add(ev)
            db.session.flush()

            # ── Blockchain ────────────────────────────────────────────────────
            bc = get_blockchain()
            bc.register_evidence(
                fir_id       = fir_id,
                fir_number   = fir.fir_number,
                evidence_desc = ev.description,
                uploaded_by  = current_user.username,
            )
            # ─────────────────────────────────────────────────────────────────

            db.session.commit()
            flash("Evidence uploaded and recorded on blockchain! 🔗", "success")
            return redirect(url_for("view_fir", fir_id=fir_id))

    # text-only evidence
    desc = request.form.get("evidence_description", "").strip()
    if desc:
        ev = Evidence(
            fir_id      = fir_id,
            description = desc,
            uploaded_by = current_user.id,
        )
        db.session.add(ev)
        db.session.flush()

        bc = get_blockchain()
        bc.register_evidence(
            fir_id       = fir_id,
            fir_number   = fir.fir_number,
            evidence_desc = desc,
            uploaded_by  = current_user.username,
        )
        db.session.commit()
        flash("Evidence added and recorded on blockchain! 🔗", "success")
    else:
        flash("No evidence provided.", "error")

    return redirect(url_for("view_fir", fir_id=fir_id))


# ──────────────────────────────────────────────────────────────────────────────
# Generate Chargesheet  — blockchain: register_chargesheet
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/fir/<int:fir_id>/generate_chargesheet", methods=["POST"])
@login_required
def generate_chargesheet(fir_id):
    fir            = FIR.query.get_or_404(fir_id)
    evidence_items = Evidence.query.filter_by(fir_id=fir_id).all()

    try:
        from chargesheet_rl import (load_trained_model,
                                     generate_chargesheet_from_description,
                                     rule_based_filter)

        agent, env = load_trained_model()
        if not agent or not env:
            flash("Model not loaded. Please train the model first.", "error")
            return redirect(url_for("view_fir", fir_id=fir_id))

        raw_sections = generate_chargesheet_from_description(
            fir.description, agent, env
        )
        ipc_sections = rule_based_filter(raw_sections, fir.description)

        if not ipc_sections:
            flash("No IPC sections generated. Please try again.", "error")
            return redirect(url_for("view_fir", fir_id=fir_id))

        # ── Build chargesheet dict ────────────────────────────────────────────
        csg = ChargeSheet()
        csg.set_fir_details(
            fir_number           = fir.fir_number,
            date_filed           = fir.creation_date,
            police_station       = current_user.police_station,
            complaint_description = fir.description,
        )
        csg.set_case_number(f"CASE/{fir.id}")
        csg.set_investigating_officer(
            f"{current_user.username} ({current_user.designation})"
        )
        csg.add_complainant(
            name    = fir.complainant_name,
            contact = fir.complainant_contact,
            address = fir.complainant_address,
        )
        csg.set_incident_details(fir.description)

        for ev in evidence_items:
            ev_type = "Digital" if ev.image_path else "Documentary"
            uploader = User.query.get(ev.uploaded_by)
            csg.add_evidence(
                type          = ev_type,
                description   = ev.description,
                collection_date = ev.upload_date,
                collected_by  = uploader.username,
                custody_chain = [uploader.username, "Evidence Room"],
            )
            if ev.image_path:
                csg.add_seized_item(
                    item_name    = f"Evidence #{ev.id}",
                    description  = ev.description,
                    seizure_date = ev.upload_date,
                    location     = "Evidence Room",
                )

        findings = (
            f"Based on investigation of {fir.fir_number} "
            f"(filed {fir.creation_date.strftime('%Y-%m-%d')}), "
            f"incident at {fir.incident_location} on "
            f"{fir.incident_date.strftime('%Y-%m-%d')}.\n"
            f"Applicable IPC sections: {', '.join(ipc_sections)}.\n"
            "All evidence has been properly documented."
        )
        csg.set_investigation_findings(findings)
        cs_data = csg.generate_chargesheet()

        # ── Blockchain: register chargesheet ──────────────────────────────────
        bc    = get_blockchain()
        block = bc.register_chargesheet(
            fir_id          = fir_id,
            fir_number      = fir.fir_number,
            ipc_sections    = ", ".join(ipc_sections),
            generated_by    = current_user.username,
            chargesheet_data = cs_data,
        )
        # ─────────────────────────────────────────────────────────────────────

        db_cs = Chargesheet(
            fir_id          = fir_id,
            ipc_sections    = ", ".join(ipc_sections),
            summary         = findings,
            generated_by    = current_user.id,
            blockchain_hash = block.hash,
        )
        db.session.add(db_cs)
        fir.status = "completed"
        db.session.commit()

        session["chargesheet_data"] = cs_data
        flash(
            f"Chargesheet generated & recorded on blockchain! 🔗 Block #{block.index}",
            "success",
        )
        return redirect(url_for("view_fir", fir_id=fir_id))

    except Exception as e:
        import traceback; traceback.print_exc()
        flash(f"Error: {e}", "error")
        return redirect(url_for("view_fir", fir_id=fir_id))


# ──────────────────────────────────────────────────────────────────────────────
# View Chargesheet
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/chargesheet/<int:fir_id>")
@login_required
def view_chargesheet(fir_id):
    fir         = FIR.query.get_or_404(fir_id)
    chargesheet = Chargesheet.query.filter_by(fir_id=fir_id).first_or_404()
    cs_data     = session.get("chargesheet_data")

    bc          = get_blockchain()
    audit_trail = bc.get_fir_audit_trail(fir_id)

    if not cs_data:
        # Regenerate on-the-fly
        try:
            from chargesheet_rl import (load_trained_model,
                                         generate_chargesheet_from_description,
                                         rule_based_filter)
            agent, env = load_trained_model()
            if agent and env:
                raw     = generate_chargesheet_from_description(fir.description, agent, env)
                ipcs    = rule_based_filter(raw, fir.description)
                csg     = ChargeSheet()
                csg.set_fir_details(fir.fir_number, fir.creation_date,
                                    current_user.police_station, fir.description)
                csg.set_case_number(f"CASE/{fir.id}")
                csg.set_investigating_officer(
                    f"{current_user.username} ({current_user.designation})"
                )
                csg.add_complainant(fir.complainant_name,
                                    fir.complainant_contact,
                                    fir.complainant_address)
                csg.set_incident_details(fir.description)
                csg.set_investigation_findings(
                    f"Applicable IPC sections: {', '.join(ipcs)}."
                )
                cs_data = csg.generate_chargesheet()
                session["chargesheet_data"] = cs_data
        except Exception as e:
            flash(f"Error regenerating chargesheet: {e}", "error")
            return redirect(url_for("view_fir", fir_id=fir_id))

    return render_template(
        "view_chargesheet.html",
        fir             = fir,
        chargesheet     = chargesheet,
        chargesheet_data = cs_data,
        audit_trail     = audit_trail,
        chain_valid     = bc.is_chain_valid(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Blockchain viewer API
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/blockchain")
@login_required
def blockchain_explorer():
    bc     = get_blockchain()
    blocks = bc.get_all_blocks()
    stats  = bc.chain_stats()
    return render_template("blockchain_explorer.html",
                           blocks=blocks, stats=stats)


@app.route("/api/blockchain/stats")
@login_required
def api_blockchain_stats():
    bc = get_blockchain()
    return jsonify(bc.chain_stats())


@app.route("/api/blockchain/fir/<int:fir_id>")
@login_required
def api_fir_audit(fir_id):
    bc = get_blockchain()
    return jsonify(bc.get_fir_audit_trail(fir_id))


# ──────────────────────────────────────────────────────────────────────────────
# DB init
# ──────────────────────────────────────────────────────────────────────────────

def init_db():
    with app.app_context():
        db.drop_all()
        db.create_all()
        if not User.query.filter_by(username="police").first():
            db.session.add(User(
                username       = "police",
                password       = generate_password_hash("police123"),
                police_station = "Central Police Station",
                designation    = "Inspector",
            ))
            db.session.commit()
            print("Default user created. Login: police / police123")


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
