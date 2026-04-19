"""
PowerGuard AI — Flask API Server
All ML logic lives in ml_engine.py — this file only handles HTTP routing.
"""

import datetime
import random

import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from ml_engine import PowerGuardMLEngine

# ─────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────
app    = Flask(__name__)
CORS(app)

engine = PowerGuardMLEngine()
print("=" * 55)
print("  PowerGuard AI — ML Backend Starting")
print("=" * 55)
engine.train()
print("=" * 55)
print(f"  ✅ All models trained | Ensemble: {engine.ensemble_accuracy():.1f}%")
print("=" * 55)


# ─────────────────────────────────────────
# STATIC FIXTURES
# ─────────────────────────────────────────

ANOMALY_FEED = [
    {"type": "theft", "icon": "⚡", "title": "Direct line tap — Sector 14B, Nagpur",
     "desc": "No meter record. Est. 2.3 kW continuous draw.", "time": "2m ago",  "risk": 92},
    {"type": "theft", "icon": "⚡", "title": "Meter seal broken — MH-4421",
     "desc": "Physical tamper detected. Usage dropped 74% overnight.", "time": "11m ago", "risk": 87},
    {"type": "fault", "icon": "⚠",  "title": "Meter MH-2891 sending null reads",
     "desc": "7 consecutive null values. Auto-replacement ticket raised.", "time": "28m ago", "risk": 78},
    {"type": "theft", "icon": "⚡", "title": "Hook connection — Zone 3, Plot 7",
     "desc": "Field team alerted. Est. ₹12 000/month revenue loss.", "time": "45m ago", "risk": 74},
    {"type": "fault", "icon": "⚠",  "title": "Voltage spike — UP-1134",
     "desc": "ML flagged 3σ deviation from 90-day baseline.", "time": "1h ago",  "risk": 68},
    {"type": "theft", "icon": "⚡", "title": "Jumper wire — Zone 5, Building C",
     "desc": "Smart meter magnetic interference detected.", "time": "2h ago",  "risk": 63},
]

SUSPICIOUS_USERS = [
    {"meter_id": "MH-4421", "name": "Ramesh Yadav",   "zone": "Zone 3, Nagpur",
     "type": "theft", "risk": 94, "reason": "Meter bypass suspected — usage 4.2× bill",
     "monthly_units": 420, "bill_amount": 680, "usage_bill_ratio": 4.2,
     "night_usage_pct": 0.52, "voltage_std": 14, "null_reads": 3,
     "meter_jump_count": 5, "avg_daily_units": 14, "weekend_spike": 0.92,
     "consecutive_zeros": 3,
     "complaint_text": "meter bypass direct wire hook connection seal broken"},

    {"meter_id": "DL-7723", "name": "Prakash Kumar",  "zone": "Zone 5, Mumbai",
     "type": "theft", "risk": 87, "reason": "Usage-bill mismatch ×3.2 for 6 months",
     "monthly_units": 380, "bill_amount": 890, "usage_bill_ratio": 3.2,
     "night_usage_pct": 0.41, "voltage_std": 11, "null_reads": 2,
     "meter_jump_count": 4, "avg_daily_units": 12.7, "weekend_spike": 0.88,
     "consecutive_zeros": 2,
     "complaint_text": "usage mismatch bill discrepancy slow meter bypass"},

    {"meter_id": "MH-2891", "name": "Sunita Devi",    "zone": "Zone 1, Pune",
     "type": "fault", "risk": 78, "reason": "Erratic voltage — 7 null reads today",
     "monthly_units": 195, "bill_amount": 1290, "usage_bill_ratio": 1.1,
     "night_usage_pct": 0.13, "voltage_std": 48, "null_reads": 7,
     "meter_jump_count": 1, "avg_daily_units": 6.5, "weekend_spike": 0.98,
     "consecutive_zeros": 7,
     "complaint_text": "null reads erratic voltage spike meter fault dead"},

    {"meter_id": "UP-1134", "name": "Anjali Singh",   "zone": "Zone 2, Nashik",
     "type": "theft", "risk": 71, "reason": "Night usage spike 2–4 AM consistently",
     "monthly_units": 340, "bill_amount": 720, "usage_bill_ratio": 2.8,
     "night_usage_pct": 0.63, "voltage_std": 13, "null_reads": 1,
     "meter_jump_count": 3, "avg_daily_units": 11.3, "weekend_spike": 0.91,
     "consecutive_zeros": 1,
     "complaint_text": "night spike late hours illegal tap usage mismatch"},

    {"meter_id": "MH-5501", "name": "Vijay Patil",    "zone": "Zone 4, Aurangabad",
     "type": "fault", "risk": 65, "reason": "Dead meter intervals detected",
     "monthly_units": 180, "bill_amount": 1100, "usage_bill_ratio": 1.0,
     "night_usage_pct": 0.11, "voltage_std": 52, "null_reads": 5,
     "meter_jump_count": 0, "avg_daily_units": 6.0, "weekend_spike": 1.02,
     "consecutive_zeros": 5,
     "complaint_text": "dead meter intervals no reading erratic fault"},

    {"meter_id": "MH-3301", "name": "Deepak Sharma",  "zone": "Zone 6, Thane",
     "type": "theft", "risk": 61, "reason": "Hook connection field report received",
     "monthly_units": 290, "bill_amount": 540, "usage_bill_ratio": 3.1,
     "night_usage_pct": 0.38, "voltage_std": 9, "null_reads": 1,
     "meter_jump_count": 2, "avg_daily_units": 9.7, "weekend_spike": 0.95,
     "consecutive_zeros": 1,
     "complaint_text": "hook wire jumper connection bypass field report theft"},
]


def _live_stats() -> dict:
    """Return live dashboard numbers with slight randomness for realism."""
    base_theft  = 47
    base_fault  = 23
    base_meters = 8420
    theft   = base_theft  + random.randint(-2, 3)
    fault   = base_fault  + random.randint(-1, 2)
    meters  = base_meters + random.randint(-10, 20)
    revenue = 240_000 + random.randint(-5_000, 8_000)
    return {
        "theft_detected": theft,
        "faulty_meters":  fault,
        "revenue_saved":  revenue,
        "meters_online":  meters,
        "last_updated":   datetime.datetime.now().isoformat(),
    }


# ─────────────────────────────────────────
# ROUTES — HEALTH
# ─────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status":   "ok",
        "service":  "PowerGuard AI Backend",
        "version":  "2.5",
        "accuracy": engine.ensemble_accuracy(),
    })


# ─────────────────────────────────────────
# ROUTES — PREDICTION
# ─────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON payload"}), 400
    try:
        return jsonify(engine.predict(data))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)
    if not data or "records" not in data:
        return jsonify({"error": "Provide a 'records' array"}), 400
    try:
        results = engine.batch_predict(data["records"])
        return jsonify({"results": results, "count": len(results)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────
# ROUTES — DASHBOARD
# ─────────────────────────────────────────

@app.route("/api/dashboard/stats", methods=["GET"])
def dashboard_stats():
    stats = _live_stats()

    # Consumption trend (30 days)
    np.random.seed(int(datetime.datetime.now().timestamp() / 3600))
    base = 420
    act  = [int(base + np.random.normal(0, 12) - i * 2.5) for i in range(30)][::-1]
    pred = [int(v + np.random.normal(2, 4)) for v in act]

    # Status distribution from trained data proportion
    if engine.is_trained and engine.live_meters:
        labels = [m['label'] for m in engine.live_meters]
        total = stats["meters_online"]
        c_theft = labels.count('theft')
        c_fault = labels.count('fault')
        c_norm = len(labels) - c_theft - c_fault
        
        status_dist = [
            int(total * (c_norm / len(labels))),
            int(total * (c_theft / len(labels))),
            int(total * (c_fault / len(labels)))
        ]
        # Adjust for remainder
        status_dist[0] = total - status_dist[1] - status_dist[2]
    else:
        status_dist = [stats["meters_online"] - 50, 35, 15]

    stats.update({
        "act_consumption":     act,
        "pred_consumption":    pred,
        "status_distribution": status_dist,
        "model_accuracy":      engine.ensemble_accuracy(),
        "model_scores":        engine.model_scores,
        "weekly_theft":        [14, 12, 18, 11, 22, 19, status_dist[1]//30],
        "weekly_faulty":       [8, 10, 7, 12, 9, 11, status_dist[2]//30],
    })
    return jsonify(stats)


@app.route("/api/ml/pca", methods=["GET"])
def get_pca():
    """Returns PCA coordinates for a sample of meters for the scatter plot."""
    if not engine.is_trained:
        return jsonify({"error": "Model not trained"}), 400
    
    # Take a sample of 200 meters
    sample_df = pd.DataFrame(random.sample(engine.live_meters, min(200, len(engine.live_meters))))
    X = engine._features(sample_df, fit=False)
    # Get first 2 PCA components (using the already fitted PCA)
    X_pca = engine.pca.transform(engine.scaler.transform(sample_df[engine.NUMERIC_COLS].values))
    
    results = []
    for i, row in sample_df.iterrows():
        results.append({
            "x": round(float(X_pca[i][0]), 3),
            "y": round(float(X_pca[i][1]), 3),
            "label": row['label']
        })
    return jsonify(results)


@app.route("/api/ml/anomaly-timeline/<meter_id>", methods=["GET"])
def anomaly_timeline(meter_id):
    """Returns 7-day risk score trend for a specific meter."""
    # Mocking a realistic trend based on meter type
    is_theft = "4421" in meter_id or "7723" in meter_id
    is_fault = "2891" in meter_id or "5501" in meter_id
    
    base = 85 if is_theft else (65 if is_fault else 15)
    labels = ["Apr 10", "Apr 11", "Apr 12", "Apr 13", "Apr 14", "Apr 15", "Apr 16"]
    data = [int(base + random.randint(-5, 10)) for _ in range(7)]
    
    return jsonify({
        "labels": labels,
        "data": data,
        "meter_id": meter_id
    })


@app.route("/api/anomalies", methods=["GET"])
def anomalies():
    return jsonify({"anomalies": ANOMALY_FEED})


@app.route("/api/meters", methods=["GET"])
def meters():
    try:
        sample = random.sample(engine.live_meters, min(24, len(engine.live_meters)))
    except Exception:
        sample = []

    result = []
    for row in sample:
        lbl = row.get("label", "normal")
        if lbl == "normal":
            st, ri = "Normal", random.randint(5, 20)
        elif lbl == "theft":
            st, ri = "Theft", random.randint(60, 95)
        else:
            st, ri = "Faulty", random.randint(50, 85)
        result.append({
            "id": row.get("meter_id", f"MTR-{random.randint(10000, 99999)}"),
            "lo": f"Barelly Zone {random.randint(1, 6)}",
            "kw": f"{int(row.get('monthly_units', random.randint(100, 500)))} kWh",
            "st": st,
            "ri": ri,
        })
    return jsonify(result)


@app.route("/api/dashboard/zones", methods=["GET"])
def zones():
    data = [
        {"zone": "Zone 3", "city": "Nagpur",     "incidents": 18 + random.randint(-1, 2), "status": "hot"},
        {"zone": "Zone 1", "city": "Pune",        "incidents": 12 + random.randint(-1, 2), "status": "warm"},
        {"zone": "Zone 5", "city": "Mumbai",      "incidents":  9 + random.randint(-1, 1), "status": "warm"},
        {"zone": "Zone 2", "city": "Nashik",      "incidents":  7 + random.randint(-1, 1), "status": "normal"},
        {"zone": "Zone 4", "city": "Aurangabad",  "incidents":  6 + random.randint(0,  1), "status": "normal"},
        {"zone": "Zone 6", "city": "Thane",       "incidents":  5 + random.randint(0,  1), "status": "normal"},
    ]
    return jsonify({"zones": data})


# ─────────────────────────────────────────
# ROUTES — ML METRICS
# ─────────────────────────────────────────

@app.route("/api/ml/metrics", methods=["GET"])
def ml_metrics():
    return jsonify(engine.get_metrics())


@app.route("/api/ml/trend", methods=["GET"])
def ml_trend():
    theft_base    = [38, 42, 35, 51, 47, 44, 47]
    fault_base    = [19, 21, 18, 25, 23, 20, 23]
    resolved_base = [12, 16, 20, 18, 25, 22, 19]
    return jsonify({
        "labels":   ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "theft":    [v + random.randint(-2, 2) for v in theft_base],
        "fault":    [v + random.randint(-1, 1) for v in fault_base],
        "resolved": [v + random.randint(-1, 2) for v in resolved_base],
    })


@app.route("/api/pca/components", methods=["GET"])
def pca_components():
    evr = engine.pca.explained_variance_ratio_.tolist()
    return jsonify({
        "explained_variance_ratio": [round(v * 100, 2) for v in evr],
        "cumulative_variance":      [round(sum(evr[: i + 1]) * 100, 2) for i in range(len(evr))],
        "n_components":             engine.pca.n_components_,
        "total_variance_explained": round(sum(evr) * 100, 2),
    })


# ─────────────────────────────────────────
# ROUTES — SUSPICIOUS USERS
# ─────────────────────────────────────────

@app.route("/api/suspicious-users", methods=["GET"])
def suspicious_users():
    NON_FEATURE = {"meter_id", "name", "zone", "type", "risk", "reason"}
    result      = []
    for u in SUSPICIOUS_USERS:
        features = {k: v for k, v in u.items() if k not in NON_FEATURE}
        try:
            pred   = engine.predict(features)
            u_copy = dict(u)
            u_copy.update({
                "ml_prediction":    pred["prediction"],
                "ml_risk_score":    pred["risk_score"],
                "ml_confidence":    pred["confidence"],
                "ml_probabilities": pred["probabilities"],
                "is_anomaly":       pred["is_anomaly"],
            })
            result.append(u_copy)
        except Exception:
            result.append(u)
    return jsonify({"users": result})


# ─────────────────────────────────────────
# ROUTES — TEXT ANALYSIS
# ─────────────────────────────────────────

@app.route("/api/analyze/text", methods=["POST"])
def analyze_text():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Provide 'text' field"}), 400

    cv_vec   = engine.count_vec.transform([text]).toarray()[0]
    vocab    = engine.count_vec.get_feature_names_out()
    top10    = sorted(zip(vocab, cv_vec), key=lambda x: -x[1])[:10]

    theft_kws = ["bypass", "hook", "wire", "seal", "tamper", "magnet", "tap", "jumper", "mismatch"]
    fault_kws = ["null", "erratic", "voltage", "spike", "dead", "fault", "malfunction", "intervals"]
    t_hits    = sum(1 for kw in theft_kws if kw in text.lower())
    f_hits    = sum(1 for kw in fault_kws  if kw in text.lower())

    if t_hits > f_hits:
        cat, conf = "theft", min(95, 60 + t_hits * 8)
    elif f_hits > 0:
        cat, conf = "fault", min(90, 55 + f_hits * 7)
    else:
        cat, conf = "normal", 70

    return jsonify({
        "text":                    text,
        "category":                cat,
        "confidence":              conf,
        "top_tokens":              [{"word": w, "count": int(c)} for w, c in top10 if c > 0],
        "theft_keywords_found":    t_hits,
        "fault_keywords_found":    f_hits,
    })


# ─────────────────────────────────────────
# ROUTES — EXPORT
# ─────────────────────────────────────────

@app.route("/api/export/reports", methods=["GET"])
def export_reports():
    rows = ["Meter ID,Customer,Location,Status,Risk Score,Date"]
    for u in SUSPICIOUS_USERS:
        rows.append(
            f"{u['meter_id']},{u['name']},{u['zone']},"
            f"{u['type'].title()},{u['risk']},2026-04-16"
        )
    csv_data = "\n".join(rows)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=powerguard_report.csv"},
    )


# ─────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, port=5001, host="0.0.0.0")
