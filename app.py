import datetime
import time
import random
import threading
import traceback
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from pymongo import MongoClient
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'enterprise_fraud_shield_secret_2026'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ==========================================
# 1. MODEL ARTIFACTS LOADING (SAFE GUARDED)
# ==========================================
try:
    artifacts = joblib.load('fraud_model.pkl')
    scaler = artifacts.get('scaler')
    rf_model = artifacts.get('rf_classifier')
    iso_forest = artifacts.get('isolation_forest')
    print("✓ ML Artifacts Loaded Successfully.")
except Exception as e:
    print(f"⚠️ Warning: Model load issue ({e}). Running on rule safeguard engine.")
    scaler, rf_model, iso_forest = None, None, None

# ==========================================
# 2. DATABASE & MEMORY CACHE
# ==========================================
LOCAL_TXN_CACHE = []
mongo_connected = False

try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    client.server_info()
    db = client["professional_ecommerce_db"]
    transactions_col = db["transactions"]
    mongo_connected = True
    print("✓ MongoDB Connected Successfully.")
except Exception as e:
    print(f"⚠️ MongoDB Offline ({e}). Running on in-memory storage.")
    mongo_connected = False
    transactions_col = None

QUARANTINE_REGISTRY = {}
simulator_active = False
simulator_thread = None

GEO_NODES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462},
    "London": {"lat": 51.5074, "lon": -0.1278},
    "New York": {"lat": 40.7128, "lon": -74.0060},
    "Moscow": {"lat": 55.7558, "lon": 37.6173}
}

CUSTOMER_NAME_POOL = [
    "Rahul Sharma", "Priya Verma", "Aditya Roy", "Sneha Gupta", 
    "Vikram Malhotra", "Ananya Iyer", "Karan Mehra", "Pooja Patel", 
    "Rohan Deshmukh", "Neha Choudhury"
]

# ==========================================
# 3. EXPLAINABLE AI (XAI) LOGIC
# ==========================================
def explain_decision(amount, hour, is_new_device, fails, velocity, geo_dist, is_anomaly, is_bot):
    reasons = []
    if is_bot:
        reasons.append("Biometric Anomaly: Synthetic / Script Injection (<30ms)")
    if amount > 50000:
        reasons.append(f"Monetary Spike: INR {amount:,.0f} exposure")
    if hour in [0, 1, 2, 3, 4, 23]:
        reasons.append("Temporal Anomaly: Midnight off-hours window")
    if is_new_device == 1:
        reasons.append("Hardware Fingerprint: Unregistered hardware/Proxy node")
    if fails >= 2:
        reasons.append(f"Credential Abuse: {fails} consecutive failed auths")
    if velocity >= 4:
        reasons.append(f"Velocity Surge: {velocity} txns / 10m (Carding/Testing)")
    if geo_dist >= 200:
        reasons.append(f"Geo-Velocity: Impossible Travel Leap ({geo_dist:,.0f} km jump)")
    if is_anomaly:
        reasons.append("Isolation Forest: Zero-day behavioral outlier")

    if not reasons:
        reasons.append("Standard legitimate transactional pattern")
    return reasons


def get_aggregated_stats():
    if mongo_connected and transactions_col is not None:
        try:
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_transactions": {"$sum": 1},
                        "total_fraud": {
                            "$sum": {
                                "$cond": [
                                    {"$or": [{"$eq": ["$status", "Blocked"]}, {"$eq": ["$is_fraud", True]}]},
                                    1, 0
                                ]
                            }
                        },
                        "total_flagged": {
                            "$sum": {
                                "$cond": [{"$eq": ["$status", "Flagged"]}, 1, 0]
                            }
                        },
                        "total_revenue": {
                            "$sum": {
                                "$cond": [{"$ne": ["$status", "Blocked"]}, "$amount", 0]
                            }
                        },
                        "avg_risk": {"$avg": "$risk_score"}
                    }
                }
            ]
            res = list(transactions_col.aggregate(pipeline))
            if res:
                stats = res[0]
                stats['avg_risk'] = round(stats.get('avg_risk', 0), 2)
                stats['total_revenue'] = round(stats.get('total_revenue', 0), 2)
                return stats
        except Exception:
            pass

    total_tx = len(LOCAL_TXN_CACHE)
    total_fraud = sum(1 for t in LOCAL_TXN_CACHE if t.get('status') == 'Blocked' or t.get('is_fraud') is True)
    total_flagged = sum(1 for t in LOCAL_TXN_CACHE if t.get('status') == 'Flagged')
    total_revenue = sum(t.get('amount', 0) for t in LOCAL_TXN_CACHE if t.get('status') != 'Blocked')
    avg_risk = round(sum(t.get('risk_score', 0) for t in LOCAL_TXN_CACHE) / max(1, total_tx), 2) if total_tx > 0 else 0.0

    return {
        "total_transactions": total_tx,
        "total_fraud": total_fraud,
        "total_flagged": total_flagged,
        "total_revenue": round(total_revenue, 2),
        "avg_risk": avg_risk
    }


def get_chart_payload(stats, recent_list):
    chart_txns = list(reversed(recent_list[:8]))
    total_tx = stats.get('total_transactions', 0)
    total_fraud = stats.get('total_fraud', 0)
    total_flagged = stats.get('total_flagged', 0)
    clean_approved = max(0, total_tx - (total_fraud + total_flagged))

    return {
        "labels": [str(t.get('user_id', 'Unknown')) for t in chart_txns],
        "scores": [float(t.get('risk_score', 0)) for t in chart_txns],
        "amounts": [float(t.get('amount', 0)) for t in chart_txns],
        "total_approved": int(clean_approved),
        "total_flagged": int(total_flagged),
        "total_blocked": int(total_fraud)
    }


# ==========================================
# 4. CORE ENGINE & TELEMETRY DISPATCH
# ==========================================
def process_transaction_logic(data):
    raw_user = str(data.get('user_id') or '').strip()
    
    if not raw_user or raw_user.upper() in ['UNKNOWN', 'CUSTOMER', 'AYUSH SINGH']:
        user_id = f"{random.choice(CUSTOMER_NAME_POOL)} #{random.randint(100, 999)}"
    else:
        user_id = raw_user

    # Quarantine Interception
    if user_id in QUARANTINE_REGISTRY:
        return {
            "user_id": user_id,
            "status": "Blocked",
            "action_code": "QUARANTINE_REJECT",
            "risk_score": 100.0,
            "risk_factors": ["Firewall Block: Identity quarantined in dynamic blacklist"],
            "quarantined": True
        }

    try:
        amount = float(data.get('amount') or 0.0)
        hour_of_day = int(data.get('hour_of_day') if data.get('hour_of_day') is not None else datetime.datetime.now().hour)
        is_new_device = int(data.get('is_new_device') or 0)
        failed_attempts = int(data.get('failed_attempts') or 0)
        txn_velocity = int(data.get('txn_velocity') or 1)
        geo_distance_km = float(data.get('geo_distance_km') or 5.0)
        avg_typing_speed_ms = float(data.get('avg_typing_speed_ms') or 140.0)
    except Exception:
        amount, hour_of_day, is_new_device, failed_attempts, txn_velocity, geo_distance_km, avg_typing_speed_ms = (
            2500.0, datetime.datetime.now().hour, 0, 0, 1, 5.0, 140.0
        )

    is_bot = avg_typing_speed_ms < 30
    feature_names = ['amount', 'hour_of_day', 'is_new_device', 'failed_attempts', 'txn_velocity', 'geo_distance_km']
    feature_vals = [amount, hour_of_day, is_new_device, failed_attempts, txn_velocity, geo_distance_km]

    is_anomaly = False
    rf_prob = 0.12

    try:
        if scaler and rf_model:
            df_features = pd.DataFrame([feature_vals], columns=feature_names)
            try:
                scaled = scaler.transform(df_features)
            except Exception:
                scaled = scaler.transform(np.array([feature_vals]))

            if hasattr(rf_model, 'predict_proba'):
                rf_prob = float(rf_model.predict_proba(scaled)[0][1])
            else:
                rf_prob = 0.90 if rf_model.predict(scaled)[0] == 1 else 0.15

            if iso_forest:
                iso_pred = iso_forest.predict(scaled)[0]
                is_anomaly = bool(iso_pred == -1)
    except Exception:
        calc = (amount / 2500.0) + (txn_velocity * 8.0) + (failed_attempts * 12.0) + (geo_distance_km / 50.0)
        rf_prob = min(0.95, max(0.12, calc / 100.0))

    base_score = rf_prob * 100.0
    if is_bot:
        base_score = min(base_score + 40.0, 99.0)
    if is_anomaly and base_score < 75:
        risk_score = round(min(base_score + 20.0, 99.0), 2)
    else:
        risk_score = round(base_score, 2)

    if geo_distance_km > 2000 or failed_attempts >= 3 or is_bot or amount >= 140000:
        risk_score = max(risk_score, 88.0)

    if risk_score >= 70:
        status = "Blocked"
        is_fraud = True
        action_code = "BLOCK"
        QUARANTINE_REGISTRY[user_id] = {
            "blocked_at": datetime.datetime.now().strftime('%H:%M:%S'),
            "risk": risk_score
        }
    elif risk_score >= 40:
        status = "Flagged"
        is_fraud = False
        action_code = "CHALLENGE_OTP"
    else:
        status = "Approved"
        is_fraud = False
        action_code = "APPROVE"

    risk_factors = explain_decision(
        amount, hour_of_day, is_new_device, failed_attempts, txn_velocity, geo_distance_km, is_anomaly, is_bot
    )

    origin_city = "Delhi"
    dest_city = "London" if geo_distance_km > 2000 else ("Mumbai" if geo_distance_km > 500 else "Lucknow")

    now_iso = datetime.datetime.utcnow().isoformat()
    formatted_time = datetime.datetime.now().strftime('%H:%M:%S')

    doc = {
        "user_id": user_id,
        "amount": amount,
        "hour_of_day": hour_of_day,
        "is_new_device": is_new_device,
        "failed_attempts": failed_attempts,
        "txn_velocity": txn_velocity,
        "geo_distance_km": geo_distance_km,
        "origin": {"city": origin_city, **GEO_NODES.get(origin_city, GEO_NODES["Delhi"])},
        "destination": {"city": dest_city, **GEO_NODES.get(dest_city, GEO_NODES["Lucknow"])},
        "risk_score": risk_score,
        "is_fraud": is_fraud,
        "status": status,
        "action_code": action_code,
        "risk_factors": risk_factors,
        "timestamp": now_iso,
        "formatted_time": formatted_time
    }

    if mongo_connected and transactions_col is not None:
        try:
            inserted = transactions_col.insert_one(doc.copy())
            doc['_id'] = str(inserted.inserted_id)
        except Exception:
            doc['_id'] = str(random.randint(100000, 999999))
    else:
        doc['_id'] = str(random.randint(100000, 999999))

    LOCAL_TXN_CACHE.insert(0, doc)

    try:
        stats = get_aggregated_stats()
        socketio.emit('telemetry_stream_event', {
            'transaction': doc,
            'stats': stats,
            'quarantine_list': QUARANTINE_REGISTRY
        })
    except Exception:
        pass

    return {
        "user_id": user_id,
        "status": status,
        "action_code": action_code,
        "risk_score": risk_score,
        "risk_factors": risk_factors,
        "quarantined": (status == "Blocked")
    }


# ==========================================
# 5. SIMULATED TRAFFIC BACKGROUND THREAD
# ==========================================
def background_traffic_simulator():
    global simulator_active
    while simulator_active:
        time.sleep(random.uniform(3.0, 5.0))
        if not simulator_active:
            break

        is_attack = random.random() < 0.25
        if is_attack:
            scenario = random.choice(['burst', 'geo', 'bot'])
            if scenario == 'burst':
                data = {
                    "user_id": f"BOT-NODE-{random.randint(100, 999)}", "amount": round(random.uniform(8000, 25000), 2),
                    "hour_of_day": random.choice([2, 3, 23]), "is_new_device": 1, "failed_attempts": 3,
                    "txn_velocity": 8, "geo_distance_km": 15, "avg_typing_speed_ms": 12
                }
            elif scenario == 'geo':
                data = {
                    "user_id": f"GEO-SPOOF-{random.randint(100, 999)}", "amount": round(random.uniform(45000, 95000), 2),
                    "hour_of_day": 14, "is_new_device": 1, "failed_attempts": 1,
                    "txn_velocity": 2, "geo_distance_km": 4800, "avg_typing_speed_ms": 110
                }
            else:
                data = {
                    "user_id": f"ZERO-DAY-{random.randint(100, 999)}", "amount": round(random.uniform(110000, 180000), 2),
                    "hour_of_day": 4, "is_new_device": 1, "failed_attempts": 5,
                    "txn_velocity": 7, "geo_distance_km": 3500, "avg_typing_speed_ms": 10
                }
        else:
            data = {
                "user_id": f"{random.choice(CUSTOMER_NAME_POOL)} #{random.randint(100, 999)}",
                "amount": round(random.uniform(450, 4800), 2),
                "hour_of_day": random.randint(9, 21), "is_new_device": 0, "failed_attempts": 0,
                "txn_velocity": 1, "geo_distance_km": random.uniform(1, 15), "avg_typing_speed_ms": random.uniform(95, 210)
            }

        with app.app_context():
            process_transaction_logic(data)


# ==========================================
# 6. ROUTE DEFINITIONS
# ==========================================
@app.route('/')
def checkout_page():
    stats = get_aggregated_stats()
    chart_data = get_chart_payload(stats, LOCAL_TXN_CACHE)
    return render_template(
        'checkout.html',
        stats=stats,
        chart_data=chart_data,
        quarantine=QUARANTINE_REGISTRY,
        transactions=LOCAL_TXN_CACHE[:20]
    )


@app.route('/admin/dashboard')
def dashboard_page():
    recent = []
    if mongo_connected and transactions_col is not None:
        try:
            raw_recent = list(transactions_col.find().sort("timestamp", -1).limit(20))
            for t in raw_recent:
                t['_id'] = str(t['_id'])
                if isinstance(t.get('timestamp'), datetime.datetime):
                    t['formatted_time'] = t['timestamp'].strftime('%H:%M:%S')
                recent.append(t)
        except Exception:
            recent = LOCAL_TXN_CACHE[:20]
    else:
        recent = LOCAL_TXN_CACHE[:20]

    stats = get_aggregated_stats()
    chart_data = get_chart_payload(stats, recent)

    return render_template(
        'dashboard.html',
        transactions=recent,
        stats=stats,
        chart_data=chart_data,
        quarantine=QUARANTINE_REGISTRY
    )


@app.route('/api/analyze-transaction', methods=['POST'])
def analyze_transaction():
    try:
        data = request.get_json(force=True, silent=True) or {}
        res = process_transaction_logic(data)

        # Ensure purely primitive values for JSON compliance (No serialization errors)
        safe_response = {
            "user_id": str(res.get("user_id", "Customer")),
            "status": str(res.get("status", "Approved")),
            "action_code": str(res.get("action_code", "APPROVE")),
            "risk_score": float(res.get("risk_score", 14.0)),
            "risk_factors": [str(f) for f in res.get("risk_factors", [])],
            "quarantined": bool(res.get("quarantined", False))
        }
        return jsonify(safe_response), 200

    except Exception as e:
        print("EXCEPTION IN ANALYZE ENDPOINT:\n", traceback.format_exc())
        return jsonify({
            "user_id": "Customer Fallback",
            "status": "Approved",
            "action_code": "APPROVE",
            "risk_score": 14.0,
            "risk_factors": ["Standard legitimate transactional pattern"],
            "quarantined": False
        }), 200


@app.route('/api/toggle-simulator', methods=['POST'])
def toggle_simulator():
    global simulator_active, simulator_thread
    data = request.get_json(force=True, silent=True) or {}
    simulator_active = data.get('enable', False)
    if simulator_active and (simulator_thread is None or not simulator_thread.is_alive()):
        simulator_thread = threading.Thread(target=background_traffic_simulator, daemon=True)
        simulator_thread.start()
    return jsonify({"simulator_active": simulator_active})


@app.route('/api/revoke-quarantine', methods=['POST'])
def revoke_quarantine():
    data = request.get_json(force=True, silent=True) or {}
    uid = data.get('user_id')
    if uid in QUARANTINE_REGISTRY:
        del QUARANTINE_REGISTRY[uid]
    socketio.emit('quarantine_updated', QUARANTINE_REGISTRY)
    return jsonify({"status": "success", "quarantine": QUARANTINE_REGISTRY})


@app.route('/api/clear-telemetry', methods=['POST'])
def clear_telemetry():
    if mongo_connected and transactions_col is not None:
        try:
            transactions_col.delete_many({})
        except Exception:
            pass
    LOCAL_TXN_CACHE.clear()
    QUARANTINE_REGISTRY.clear()
    socketio.emit('telemetry_reset_event')
    return jsonify({"status": "success"})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='127.0.0.1', port=5000, use_reloader=False, allow_unsafe_werkzeug=True)