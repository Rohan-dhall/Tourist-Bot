import csv
import json
import re
import os
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

# ================== CONFIG ==================
NVIDIA_API_KEY = "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"
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "deepseek-ai/deepseek-v4-pro"

client = OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)
app = Flask(__name__)

# ================== LOAD ALL DATA FROM CSV ==================
DATA_DIR = "data"
CITIES = ["jaipur", "agra", "katra", "jammu", "udaipur", "kerala"]

def load_csv(filepath):
    data = []
    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:   # utf-8-sig automatically handles BOM
            reader = csv.DictReader(f)
            for row in reader:
                cleaned = {}
                for k, v in row.items():
                    # Skip if key is None or empty after stripping
                    if k is None:
                        continue
                    key = k.strip()
                    if not key:
                        continue
                    value = v.strip() if v else ""
                    cleaned[key] = value
                data.append(cleaned)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found.")
    return data

# Load all datasets into memory
activities_db = {}
hotels_db = {}
flights_db = {}
trains_db = {}

for city in CITIES:
    activities_db[city] = load_csv(os.path.join(DATA_DIR, f"activities_{city}.csv"))
    hotels_db[city] = load_csv(os.path.join(DATA_DIR, f"hotels_{city}.csv"))
    flights_db[city] = load_csv(os.path.join(DATA_DIR, f"flights_from_{city}.csv"))
    trains_db[city] = load_csv(os.path.join(DATA_DIR, f"trains_from_{city}.csv"))

# ================== LLM HELPERS ==================
def call_llm(messages, temperature=0.2, max_tokens=2000):
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"thinking": False}},
            stream=False
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM error: {e}")
        return None

def extract_json(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None

# ================== INTENT CLASSIFICATION ==================
def classify_intent(query):
    system = """You are a travel assistant intent classifier. Output ONLY valid JSON:
{
  "intent": "activities|hotels|flights|trains|unknown",
  "entities": {
    "city": "city name in lowercase (jaipur, agra, katra, jammu, udaipur, kerala)",
    "destination": "destination city for flights/trains (optional)",
    "sort_by": "price|rating|none"
  },
  "confidence": 0.0-1.0
}
For queries about activities/sightseeing: intent=activities.
For accommodation: intent=hotels.
For flights: intent=flights.
For trains: intent=trains.
If the user asks to sort by cheapest/lowest price, set sort_by=price.
If they ask for best/highest rated/top reviews, set sort_by=rating.
Otherwise sort_by=none."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": query}]
    raw = call_llm(messages, 0.1, 300)
    parsed = extract_json(raw) or {}
    return {
        "intent": parsed.get("intent", "unknown"),
        "entities": parsed.get("entities", {}),
        "confidence": parsed.get("confidence", 0.8)
    }

# ================== SORTING & TRUST LOGIC ==================
MIN_REVIEWS = 30   # items with fewer reviews are considered untrustworthy

def sort_items(items, sort_by):
    """Sort list of dicts. Excludes items with low review counts when sorting by rating."""
    if sort_by == "price":
        return sorted(items, key=lambda x: float(x.get("price", x.get("price_per_night", 0))))
    elif sort_by == "rating":
        # Filter out items with too few reviews
        filtered = [item for item in items if int(item.get("review_count", 0)) >= MIN_REVIEWS]
        # Sort by rating descending, then review_count descending
        return sorted(filtered, key=lambda x: (float(x.get("rating", 0)), int(x.get("review_count", 0))), reverse=True)
    else:
        return items   # no sorting

def format_item_list(items, type_label, city=None):
    """Return a readable message for a list of items."""
    if not items:
        return "No matching results found."
    top = items[:5]   # show max 5
    heading = f"Here are the top {type_label} in {city.title()}:" if city else f"Top {type_label}:"
    lines = [heading]
    for i, item in enumerate(top, 1):
        name = item.get("name") or item.get("train_name") or item.get("airline") or ""
        price = item.get("price") or item.get("price_per_night") or "?"
        rating = item.get("rating", "?")
        reviews = item.get("review_count", "?")
        link = item.get("booking_link", "#")
        lines.append(f"{i}. {name} — ₹{price} | Rating: {rating} ({reviews} reviews) [Book]({link})")
    if len(items) > 5:
        lines.append(f"... and {len(items)-5} more. Refine your search to see all.")
    return "\n".join(lines)

# ================== CHAT ENDPOINT ==================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "user_query" not in data:
        return jsonify({"error": "Missing user_query"}), 400
    query = data["user_query"].strip()

    # Step 1: Classify intent
    intent_res = classify_intent(query)
    intent = intent_res["intent"]
    entities = intent_res["entities"] or {}

    # Safe extraction – convert None to empty string before lowering
    city = (entities.get("city") or "").lower()
    sort_by = (entities.get("sort_by") or "none").lower()
    destination = (entities.get("destination") or "").lower()

    # Validate sort_by
    if sort_by not in ("price", "rating"):
        sort_by = "none"

    # Fallback: if LLM didn't catch city, try to find it in the query
    if not city and intent in ("activities", "hotels", "flights", "trains"):
        for c in CITIES:
            if c in query.lower():
                city = c
                break

    if not city:
        answer = "Please specify a city (Jaipur, Agra, Katra, Jammu, Udaipur, Kerala) for your query."
    else:
        # Step 2: Retrieve and sort data based on intent
        if intent == "activities":
            items = activities_db.get(city, [])
            if not items:
                answer = f"No activities found for {city.title()}."
            else:
                sorted_items = sort_items(items, sort_by)
                answer = format_item_list(sorted_items, "activities", city)

        elif intent == "hotels":
            items = hotels_db.get(city, [])
            if not items:
                answer = f"No hotels found for {city.title()}."
            else:
                sorted_items = sort_items(items, sort_by)
                answer = format_item_list(sorted_items, "hotels", city)

        elif intent == "flights":
            dep_items = flights_db.get(city, [])
            if not dep_items:
                answer = f"No flights found from {city.title()}."
            else:
                if destination:
                    dep_items = [f for f in dep_items if f["arrival_city"] == destination]
                    if not dep_items:
                        answer = f"No flights found from {city.title()} to {destination.title()}."
                        return jsonify({"response": answer, "intent": intent, "confidence": intent_res["confidence"]})
                sorted_items = sort_items(dep_items, sort_by)
                answer = format_item_list(sorted_items, "flights", city)

        elif intent == "trains":
            dep_items = trains_db.get(city, [])
            if not dep_items:
                answer = f"No trains found from {city.title()}."
            else:
                if destination:
                    dep_items = [t for t in dep_items if t["arrival_city"] == destination]
                    if not dep_items:
                        answer = f"No trains found from {city.title()} to {destination.title()}."
                        return jsonify({"response": answer, "intent": intent, "confidence": intent_res["confidence"]})
                sorted_items = sort_items(dep_items, sort_by)
                answer = format_item_list(sorted_items, "trains", city)

        else:
            answer = "I can help with activities, hotels, flights, and trains. Please mention a city and what you need."

    response = {
        "response": answer,
        "intent": intent,
        "confidence": intent_res["confidence"],
        "entities": entities
    }
    return jsonify(response)

# ================== FRONTEND ==================
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)