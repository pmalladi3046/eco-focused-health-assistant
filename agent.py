"""
eco_agent_tf.py

Eco-Friendly Lifestyle Agent using TensorFlow / Keras.

- Terminal-based interactive agent.
- Estimates daily CO2 footprint for commute, food, electricity.
- Suggests small changes ranked by impact * personalized acceptance probability.
- Uses a tiny Keras personalizer model and a tiny Keras sentiment model.
- Persists user history and the personalizer model between runs.

Dependencies:
    pip install tensorflow numpy
"""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf

# ----------------------------
# File paths
# ----------------------------
MODEL_DIR = Path("eco_model_tf")         # Keras saves a folder
HISTORY_PATH = Path("user_history.json")

# ----------------------------
# Emission factors (kg CO2)
# ----------------------------
EMISSION = {
    "commute": {"car": 0.271, "bus": 0.105, "train": 0.041, "bike": 0.0, "walk": 0.0},
    "food": {"meat_heavy": 5.0, "meat_moderate": 2.5, "vegetarian": 1.5, "vegan": 1.2},
    "electricity_per_kwh": 0.475,
}

# ----------------------------
# Suggestions
# ----------------------------
SUGGESTIONS = [
    {"id": "commute_switch_public", "label": "Try public transit / carpool instead of driving alone", "type": "commute"},
    {"id": "commute_bike", "label": "Bike or walk instead of driving (if feasible)", "type": "commute"},
    {"id": "diet_reduce_meat", "label": "Replace one meat-heavy meal with a vegetarian/vegan meal", "type": "food"},
    {"id": "electricity_reduce", "label": "Reduce electricity usage by 10% (LEDs, lower thermostat, unplug devices)", "type": "electricity"},
]

# ----------------------------
# Tiny sentiment dataset (embedded)
# ----------------------------
SENTIMENT_DATA = [
    ("I feel great today!", 1),
    ("I am so happy with my progress.", 1),
    ("This is terrible, I failed again.", 0),
    ("I am frustrated with the traffic.", 0),
    ("Feeling neutral about everything.", 0),
    ("Excited to try new things!", 1),
]

# ----------------------------
# Utilities: emission estimates
# ----------------------------
def estimate_commute_emissions(mode: str, distance_km: float) -> float:
    factor = EMISSION["commute"].get(mode.lower(), 0.0)
    return factor * max(distance_km, 0.0)

def estimate_food_emissions(meal_type: str, meals_per_day: float = 1.0) -> float:
    factor = EMISSION["food"].get(meal_type.lower(), EMISSION["food"]["meat_moderate"])
    return factor * max(meals_per_day, 0.0)

def estimate_electricity_emissions(kwh_per_day: float) -> float:
    return EMISSION["electricity_per_kwh"] * max(kwh_per_day, 0.0)

# ----------------------------
# Feature vector for personalizer
# ----------------------------
def build_feature_vector(total_emissions, suggestion_index, num_suggestions):
    # normalize total by 20 kg/day (rough upper bound) for stable inputs
    norm = float(total_emissions) / 20.0
    vec = np.zeros(1 + num_suggestions, dtype=np.float32)
    vec[0] = norm
    vec[1 + suggestion_index] = 1.0
    return vec

# ----------------------------
# Synthetic warm-start data (for personalizer)
# ----------------------------
def generate_synthetic_data(num_samps=500):
    X = []
    y = []
    for _ in range(num_samps):
        total = np.random.uniform(0.5, 25.0)
        suggestion_idx = np.random.randint(0, len(SUGGESTIONS))
        stype = SUGGESTIONS[suggestion_idx]["type"]
        base = 0.25 if stype == "commute" else 0.35 if stype == "food" else 0.3
        prob = np.clip(base + 0.02 * (total - 2.0), 0.05, 0.95)
        label = float(np.random.rand() < prob)
        X.append(build_feature_vector(total, suggestion_idx, len(SUGGESTIONS)))
        y.append(label)
    return np.stack(X), np.array(y, dtype=np.float32)

# ----------------------------
# Save / load history
# ----------------------------
def save_history(history, path=HISTORY_PATH):
    with open(path, "w") as f:
        json.dump(history, f, indent=2)

def load_history(path=HISTORY_PATH):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}

# ----------------------------
# Estimate saving hint per suggestion
# ----------------------------
def estimate_saving_hint(suggestion_id, commute_mode, distance_km, meal_type, meals, kwh):
    commute_current = estimate_commute_emissions(commute_mode, distance_km * 2)
    food_current = estimate_food_emissions(meal_type, meals)
    elec_current = estimate_electricity_emissions(kwh)

    if suggestion_id == "commute_switch_public":
        if commute_mode == "car" and distance_km > 0:
            new = estimate_commute_emissions("bus", distance_km * 2)
            return max(commute_current - new, 0.0)
        return 0.0
    elif suggestion_id == "commute_bike":
        if commute_mode in ("car", "bus", "train") and 0 < distance_km <= 10:
            new = estimate_commute_emissions("bike", distance_km * 2)
            return max(commute_current - new, 0.0)
        return 0.0
    elif suggestion_id == "diet_reduce_meat":
        old = EMISSION["food"].get(meal_type, 2.5)
        new = EMISSION["food"]["vegetarian"]
        return max(old - new, 0.0)
    elif suggestion_id == "electricity_reduce":
        return 0.10 * elec_current
    return 0.0

# ----------------------------
# Build Keras personalizer model
# ----------------------------
def build_personalizer(input_dim, hidden=16, lr=0.01):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(hidden, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy"])
    return model

# ----------------------------
# Train personalizer on synthetic data (warm start)
# ----------------------------
def warm_start_personalizer(model, epochs=6, batch_size=32):
    X, y = generate_synthetic_data()
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

# ----------------------------
# Tiny sentiment model: bag-of-words + single dense unit
# We retrain this tiny model from SENTIMENT_DATA every run (it's fast).
# ----------------------------
def build_and_train_sentiment():
    # build vocab from SENTIMENT_DATA
    vocab = {}
    for text, _ in SENTIMENT_DATA:
        for w in text.lower().split():
            if w not in vocab:
                vocab[w] = len(vocab)
    if len(vocab) == 0:
        vocab = {"": 0}

    X = []
    y = []
    for text, label in SENTIMENT_DATA:
        vec = np.zeros(len(vocab), dtype=np.float32)
        for w in text.lower().split():
            if w in vocab:
                vec[vocab[w]] = 1.0
        X.append(vec)
        y.append(float(label))
    X = np.stack(X)
    y = np.array(y, dtype=np.float32)

    # model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(vocab),)),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.05),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy"])
    # train (tiny)
    model.fit(X, y, epochs=50, verbose=0)
    return model, vocab

def predict_sentiment(text, sentiment_model, vocab):
    vec = np.zeros(len(vocab), dtype=np.float32)
    for w in text.lower().split():
        if w in vocab:
            vec[vocab[w]] = 1.0
    prob = float(sentiment_model.predict(vec.reshape(1, -1), verbose=0)[0, 0])
    return prob  # 0..1 (0 negative, 1 positive)

# ----------------------------
# Online update for personalizer (one-sample fine-tune)
# ----------------------------
def online_update_personalizer(model, feat, label, epochs=8):
    X = feat.reshape(1, -1).astype(np.float32)
    y = np.array([label], dtype=np.float32)
    model.fit(X, y, epochs=epochs, verbose=0)

# ----------------------------
# Main interactive loop
# ----------------------------
def run_agent():
    print("🌱 Eco-Friendly Lifestyle Agent (TensorFlow version)")
    history = load_history()
    user_name = history.get("name")
    if not user_name:
        user_name = input("What's your name? ").strip() or "Friend"
        history["name"] = user_name
        history["records"] = []

    # Personalizer model: load if exists, otherwise build & warm-start and save
    input_dim = 1 + len(SUGGESTIONS)
    if MODEL_DIR.exists():
        try:
            personalizer = tf.keras.models.load_model(str(MODEL_DIR))
            print("Loaded existing personalizer model.")
        except Exception:
            print("Could not load saved model; rebuilding and warm-starting.")
            personalizer = build_personalizer(input_dim)
            warm_start_personalizer(personalizer)
            personalizer.save(str(MODEL_DIR))
    else:
        print("Building personalizer model and warm-starting with synthetic data...")
        personalizer = build_personalizer(input_dim)
        warm_start_personalizer(personalizer)
        personalizer.save(str(MODEL_DIR))

    # Sentiment model (tiny) - rebuild & train from the embedded data (fast)
    sentiment_model, sentiment_vocab = build_and_train_sentiment()
    # (We don't persist the tiny sentiment model/vocab here because it's trained from static embedded data.)

    # Infinite interactive loop; user can type "quit" at any prompt
    while True:
        print("\n--- New report (type 'quit' anywhere to exit) ---")
        mood_text = input("How are you feeling today? ").strip()
        if mood_text.lower() == "quit":
            break
        mood_score = predict_sentiment(mood_text, sentiment_model, sentiment_vocab)
        if mood_score > 0.5:
            print("Glad to hear you're feeling positive! 🙂")
        else:
            print("I hear you — let's see what small actions could help 🌱")

        commute_mode = input("Commute mode (car/bus/train/bike/walk) [car]: ").strip().lower() or "car"
        if commute_mode == "quit": break
        try:
            distance_km = float(input("One-way commute distance in km [10]: ") or 10.0)
        except Exception:
            distance_km = 10.0

        meal_type = input("Diet type (meat_heavy/meat_moderate/vegetarian/vegan) [meat_moderate]: ").strip() or "meat_moderate"
        if meal_type == "quit": break
        try:
            meals = float(input("Meals per day [1]: ") or 1.0)
        except Exception:
            meals = 1.0

        try:
            kwh = float(input("Estimated electricity usage today in kWh [10]: ") or 10.0)
        except Exception:
            kwh = 10.0

        # compute emissions
        commute_em = estimate_commute_emissions(commute_mode, distance_km * 2)  # round-trip
        food_em = estimate_food_emissions(meal_type, meals)
        elec_em = estimate_electricity_emissions(kwh)
        total = commute_em + food_em + elec_em

        print(f"\nEstimated daily emissions (kg CO₂):\n - Commute: {commute_em:.2f}\n - Food: {food_em:.2f}\n - Electricity: {elec_em:.2f}\n => TOTAL ≈ {total:.2f} kg CO₂/day")

        # candidate suggestions with personalized scores
        candidate_info = []
        for i, s in enumerate(SUGGESTIONS):
            saving = estimate_saving_hint(s["id"], commute_mode, distance_km, meal_type, meals, kwh)
            feat = build_feature_vector(total, i, len(SUGGESTIONS))
            score = float(personalizer.predict(feat.reshape(1, -1), verbose=0)[0, 0])
            candidate_info.append((s, saving, score, feat))
        # rank by impact * acceptance score
        candidate_info.sort(key=lambda t: (t[2] * t[1]), reverse=True)

        print("\nSuggestions (ranked):")
        for idx, (s, saving, score, _) in enumerate(candidate_info):
            print(f"{idx+1}. {s['label']} — Est. saving: {saving:.2f} kg CO₂ — Personalized score: {score:.2f}")

        # present top suggestion
        top = candidate_info[0]
        s, saving, score, feat = top
        print(f"\nTop suggestion: {s['label']}\nEstimated saving: {saving:.2f} kg CO₂")
        resp = input("Will you try this (yes/no)? ").strip().lower()
        if resp == "quit":
            break
        accepted = 1.0 if resp in ("y", "yes", "sure", "ok") else 0.0

        # record and persist
        record = {
            "mood": mood_text,
            "mood_score": mood_score,
            "commute_mode": commute_mode,
            "distance_km": distance_km,
            "meal_type": meal_type,
            "meals": meals,
            "kwh": kwh,
            "total_emissions": total,
            "suggestion_id": s["id"],
            "suggestion_saving": saving,
            "model_score_before": score,
            "accepted": accepted
        }
        history.setdefault("records", []).append(record)
        save_history(history)

        # online update personalizer and save
        online_update_personalizer(personalizer, feat, accepted, epochs=8)
        try:
            personalizer.save(str(MODEL_DIR))
        except Exception:
            # best-effort save; ignore failures
            pass

        if accepted:
            print("Awesome — thanks for trying it! I'll remember this preference.")
        else:
            print("No worries — small steps are still progress.")

        print("\nType 'quit' at any prompt to exit, or continue adding reports.")

    print("\nGoodbye! Your progress is saved locally (user_history.json and model).")

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    try:
        run_agent()
    except KeyboardInterrupt:
        print("\nExited by user.")