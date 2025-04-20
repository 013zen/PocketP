from flask import Flask, render_template, request, jsonify, redirect, url_for
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
from threading import Lock
import shutil
from uh import detect_all_structures
from uh import estimate_emotional_state_semantic
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher
from datetime import datetime
from typing import List, Dict
from uh import detect_all_structures, get_existential_drift
import ast
import difflib
import copy
import re 
import uuid

os.makedirs("logs", exist_ok=True)


# Load embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def fuzzy_match(a, b, threshold=0.8):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

# Lock to protect file writes
file_lock = Lock()

load_dotenv()

# Set the key into the environment directly (works on Windows)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


app = Flask(__name__, static_folder='static')



# Global filename for user history
USER_HISTORY_FILENAME = 'user_history.json'




DEFAULT_THEME_STRUCTURE = {
    "beliefs": [],
    "tensions": [],
    "unresolved": False
}

DEFAULT_MAP_STRUCTURE = {
    "agency": copy.deepcopy(DEFAULT_THEME_STRUCTURE),
    "fulfillment": copy.deepcopy(DEFAULT_THEME_STRUCTURE),
    "relationships": copy.deepcopy(DEFAULT_THEME_STRUCTURE),
    "identity": copy.deepcopy(DEFAULT_THEME_STRUCTURE),
    "values": copy.deepcopy(DEFAULT_THEME_STRUCTURE),
    "desires": copy.deepcopy(DEFAULT_THEME_STRUCTURE)
}


def get_thought_map_path(username):
    os.makedirs("thought_maps", exist_ok=True)
    return os.path.join("thought_maps", f"{username}.json")

def load_thought_map(username):
    path = get_thought_map_path(username)

    if not os.path.exists(path):
        print(f"🧠 No existing map for {username}, initializing default structure.")
        save_thought_map(username, copy.deepcopy(DEFAULT_MAP_STRUCTURE))
        return copy.deepcopy(DEFAULT_MAP_STRUCTURE)

    try:
        with open(path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("⚠️ Corrupted thought map. Creating emergency backup.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        corrupted_path = f"{path.replace('.json', '')}_CORRUPTED_{timestamp}.json"
        shutil.copyfile(path, corrupted_path)
        print(f"📁 Backup saved to: {corrupted_path}")
        save_thought_map(username, copy.deepcopy(DEFAULT_MAP_STRUCTURE))
        return copy.deepcopy(DEFAULT_MAP_STRUCTURE)

def save_thought_map(username, new_map):
    path = get_thought_map_path(username)

    # Update last_updated timestamp for any modified domains
    now = datetime.now().isoformat()
    for domain in new_map:
        domain_data = new_map[domain]
        if domain_data["beliefs"] or domain_data["tensions"]:
            domain_data["last_updated"] = now

    # Create daily backup before overwriting
    backup_dir = "backups"
    os.makedirs(backup_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    backup_path = os.path.join(backup_dir, f"{username}_thought_map_{today}.json")

    if not os.path.exists(backup_path):
        try:
            if os.path.exists(path):
                shutil.copyfile(path, backup_path)
                print(f"📦 Daily backup created: {backup_path}")
        except Exception as e:
            print(f"⚠️ Backup error for {username}: {e}")

    with open(path, 'w') as f:
        json.dump(new_map, f, indent=2)





DEFAULT_THEME_STRUCTURE = {
    "beliefs": [],  # each belief is a dict with: summary, source, confidence, timestamp
    "tensions": [],  # each tension is a dict with: summary, source, timestamp
    "unresolved": False,  # flag to prioritize domain in future prompts
    "last_updated": None  # optional: tracks when this theme was last edited
}







def keyword_overlap(text, keywords):
    text_words = set(re.findall(r'\w+', text.lower()))
    return any(k.lower() in text_words for k in keywords)

def semantic_similarity(a, b, threshold=0.75):
    emb_a = model.encode(a, convert_to_tensor=True)
    emb_b = model.encode(b, convert_to_tensor=True)
    score = util.cos_sim(emb_a, emb_b).item()
    return score > threshold

def is_structure_referenced(user_input, structure):
    content = structure.get('content', '')
    keywords = structure.get('linked_map_entries', [])
    
    if keyword_overlap(user_input, keywords):
        return True
    if fuzzy_match(user_input, content):
        return True
    if semantic_similarity(user_input, content):
        return True
    return False

def update_structure_relevance(user_input, user_history):
    now = datetime.utcnow().isoformat()
    active = user_history.get('active', [])
    archived = user_history.get('archived', [])
    updated_active = []

    for struct in active:
        if is_structure_referenced(user_input, struct):
            struct['unreferenced_count'] = 0
        else:
            struct['unreferenced_count'] = struct.get('unreferenced_count', 0) + 1

        if struct['unreferenced_count'] >= 8:
            struct['archived_on'] = now
            archived.append(copy.deepcopy(struct))
        else:
            updated_active.append(struct)

    # Check archive for resurfacing
    resurfaced = []
    for archived_struct in archived[:]:
        if is_structure_referenced(user_input, archived_struct):
            archived_struct['resurfaced_count'] = archived_struct.get('resurfaced_count', 0) + 1
            archived_struct['recalled_on'] = now
            archived_struct['unreferenced_count'] = 0
            resurfaced.append(archived_struct)
            archived.remove(archived_struct)

    updated_active.extend(resurfaced)

    # Return updated history
    return {
        'active': updated_active,
        'archived': archived
    }












def save_user_history_with_links(
    username: str,
    user_input: str,
    parsed_analysis: dict,
    user_data_override: dict = None,
    resolved_structures: list = None,
    active_structures: list = None,  
    map_links: list = None,
    ai_response: str = None
) -> dict:

    import os, json
    directory = "user_data"
    os.makedirs(directory, exist_ok=True)
    history_path = os.path.join(directory, f"{username}_history.json")

    # 🧠 Load old history first (even if user_data_override is passed)
    try:
        with open(history_path, "r") as f:
            contents = f.read()
            if contents.strip():
                old_data = json.loads(contents)
            else:
                raise json.JSONDecodeError("Empty file", contents, 0)
    except (FileNotFoundError, json.JSONDecodeError):
        old_data = {
            "user_input": [],
            "ai_response": [],
            "dialogue": [],
            "active": [],
            "resolved": [],
            "archived": [],
            "threads": [],
            "map_links": []
        }

    # 🧠 Merge override into old data
    user_data = user_data_override or old_data
    for key in ["user_input", "ai_response", "dialogue", "active", "resolved", "archived", "threads", "map_links"]:
        user_data.setdefault(key, old_data.get(key, []))

    # ✅ Append user input + response
    user_data["user_input"].append(user_input)

    if ai_response is not None:
        user_data["ai_response"].append(ai_response)

        dialogue_entry = {
            "user": user_input,
            "ai": ai_response
        }

        # ✅ Add metadata about what structure the reflection addressed
        if parsed_analysis:
            if "structure_id" in parsed_analysis:
                dialogue_entry["structure_id"] = parsed_analysis["structure_id"]
            if "structure_summary" in parsed_analysis:
                dialogue_entry["structure_summary"] = parsed_analysis["structure_summary"]

        user_data["dialogue"].append(dialogue_entry)

    # ✅ Add resolved structures
    if resolved_structures:
        user_data["resolved"].extend(resolved_structures)

    
    # ✅ Add map links (even if empty to preserve consistency)
    if map_links is not None:
        user_data.setdefault("map_links", [])
        existing = {(l["source"], l["target"]) for l in user_data["map_links"]}
        for link in map_links:
            source = link.get("source")
            target = link.get("target")
            if source and target and (source, target) not in existing:

                user_data["map_links"].append(link)

        for link in map_links:
            sid = link.get("structure_id")
            for struct in user_data.get("active", []):
                if struct.get("id") == sid:
                    struct.setdefault("thought_map_links", []).append(link)


    # ✅ Add active structures from parsed analysis
    if active_structures:
        user_data["active"] = active_structures






    # ✅ Timestamp active
    turn_index = len(user_data["user_input"])
    for struct in user_data.get("active", []):
        struct.setdefault("created_turn", turn_index)
        struct["last_seen_turn"] = turn_index

    print("📦 Map links saved:", json.dumps(user_data.get("map_links", []), indent=2))


    # ✅ Save
    with open(history_path, "w") as f:
        json.dump(user_data, f, indent=2)

    return user_data



















def log_thought_link(username, belief_or_tension, source_structure, theme, kind="belief", input_index=None, resolved_turn=None):
    import os
    import json
    from datetime import datetime

    path = f"thought_links/{username}_maplog.json"
    os.makedirs("thought_links", exist_ok=True)

    # ✅ Load existing file safely
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                content = f.read().strip()
                data = json.loads(content) if content else []
        except json.JSONDecodeError:
            print(f"⚠️ Corrupted maplog for {username}, starting fresh.")
            data = []
    else:
        data = []

    # ✍️ New entry
    entry = {
        "theme": theme,
        "timestamp": str(datetime.now()),
        "source_structure_id": source_structure.get("id"),
        "source_type": source_structure.get("type"),
        "source_snippet": source_structure.get("source_snippet") or source_structure.get("summary"),
        "input_index": input_index,
        "resolved_on_turn": resolved_turn,
    }

    if kind == "belief":
        entry["belief_summary"] = belief_or_tension
    else:
        entry["tension_summary"] = belief_or_tension
        entry["status"] = "active"

    data.append(entry)

    # ✅ Save updated log
    with open(path, "w") as f:
        json.dump(data, f, indent=2)







def fuzzy_match(a, b, threshold=0.65):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold



def apply_map_delta_with_links(current_map, delta, unresolved_structures=None, username=None):

    map_links = []
    now = datetime.now().isoformat()

    for raw_theme, updates in delta.items():
        theme = raw_theme.lower().strip()
        if theme not in current_map:
            print(f"➕ New theme '{theme}' not in thought map. Initializing.")
            current_map[theme] = copy.deepcopy(DEFAULT_THEME_STRUCTURE)

        theme_updated = False

        # ✳️ Add beliefs
        for belief_text in updates.get("beliefs", []):
            if not belief_text or not isinstance(belief_text, str):
                continue

            existing = [b["summary"] for b in current_map[theme]["beliefs"]]
            if belief_text not in existing:
                belief_entry = {
                    "summary": belief_text,
                    "timestamp": now,
                    "origin": "map_delta"
                }

                structure_id = None
                best_match = None
                best_score = 0

                if unresolved_structures:
                    for s in unresolved_structures:
                        structure_text = s.get("source_snippet") or s.get("summary")
                        if isinstance(structure_text, dict):
                            structure_text = " vs. ".join(structure_text.values())
                        if not isinstance(structure_text, str):
                            continue
                        match_score = difflib.SequenceMatcher(None, belief_text.lower(), structure_text.lower()).ratio()
                        if match_score > best_score:
                            best_match = s
                            best_score = match_score

                if best_match:
                    s = best_match
                    belief_entry.update({
                        "source_structure": s.get("type"),
                        "source_snippet": s.get("source_snippet") or s.get("summary"),
                        "source_structure_id": structure_id or str(uuid.uuid4()),
                        "match_score": round(best_score, 2)
                    })

                    s.setdefault("thought_map_links", []).append({
                        "theme": theme,
                        "type": "belief",
                        "summary": belief_text,
                        "timestamp": now,
                        "match_score": round(best_score, 2)
                    })

                    map_links.append({
                        "theme": theme,
                        "timestamp": now,
                        "belief": belief_entry,
                        "structure_summary": s.get("summary"),
                        "structure_id": s.get("id"),
                        "type": s.get("type"),
                        "match_score": round(best_score, 2),
                        "source": s.get("summary"),
                        "target": belief_entry.get("summary")
                    })

                    if username:
                        log_thought_link(username, belief_text, s, theme, kind="belief")

                current_map[theme]["beliefs"].append(belief_entry)
                print(f"🌱 Added belief to [{theme}]: {belief_text}")
                theme_updated = True

        # ✳️ Add tensions
        for raw_tension in updates.get("tensions", []):
            if isinstance(raw_tension, dict):
                tension_text = (
                    raw_tension.get("summary") or
                    raw_tension.get("tension_summary") or
                    f"{raw_tension.get('source', '')} vs. {raw_tension.get('resolution', '')}".strip()
                )
                timestamp = raw_tension.get("timestamp", now)
            else:
                tension_text = raw_tension
                timestamp = now

            if not tension_text or not isinstance(tension_text, str) or not tension_text.strip():
                print(f"⚠️ Skipping tension: could not extract text from {raw_tension}")
                continue

            existing = [t["summary"] for t in current_map[theme]["tensions"]]
            if tension_text not in existing:
                tension_entry = {
                    "summary": tension_text,
                    "timestamp": timestamp,
                    "origin": "map_delta"
                }

                best_match = None
                best_score = 0

                if unresolved_structures:
                    for s in unresolved_structures:
                        structure_text = s.get("source_snippet") or s.get("summary")
                        if isinstance(structure_text, dict):
                            structure_text = " vs. ".join(structure_text.values())
                        if not isinstance(structure_text, str):
                            continue
                        match_score = difflib.SequenceMatcher(None, tension_text.lower(), structure_text.lower()).ratio()
                        if match_score > best_score:
                            best_match = s
                            best_score = match_score

                if best_match:
                    s = best_match
                    tension_entry.update({
                        "source_structure": s.get("type"),
                        "source_snippet": s.get("source_snippet") or s.get("summary"),
                        "structure_id": s.get("id", ""),
                        "match_score": round(best_score, 2)
                    })

                    s.setdefault("thought_map_links", []).append({
                        "theme": theme,
                        "type": "tension",
                        "summary": tension_text,
                        "timestamp": timestamp,
                        "match_score": round(best_score, 2)
                    })

                    map_links.append({
                        "theme": theme,
                        "tension": tension_text,
                        "structure_summary": s.get("summary"),
                        "structure_id": s.get("id"),
                        "type": s.get("type"),
                        "match_score": round(best_score, 2)
                    })

                    if username:
                        log_thought_link(username, tension_text, s, theme, kind="tension")

                current_map[theme]["tensions"].append(tension_entry)
                current_map[theme]["unresolved"] = True
                print(f"⚡ Added tension to [{theme}]: {tension_text}")
                theme_updated = True

        if theme_updated:
            current_map[theme]["last_updated"] = now
        else:
            print(f"ℹ️ No new beliefs or tensions added to [{theme}].")

    return current_map, map_links



















def convert_sets_to_lists(obj):
    if isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, list):
        return [convert_sets_to_lists(v) for v in obj]
    else:
        return obj



def analyze_input(username, user_input):
    from collections import Counter

    try:
        history = load_user_history(username)
        unresolved = history.get("active_structures", [])

        # Detect structures using Python backend
        parsed = detect_all_structures(user_input)
        structures = parsed["structures"]

        # Estimate emotional state using semantic cues
        emotional_state = estimate_emotional_state_semantic(user_input)

        # Determine dominant structure type
        dominant_type = None
        if structures:
            type_counts = Counter([s['type'] for s in structures])
            dominant_type = type_counts.most_common(1)[0][0]

        # Detect existential drift
        drift = next((s for s in structures if s["type"] == "existential_drift"), None)
        existential_drift = {
            "present": True,
            "anchor_text": drift.get("summary"),
            "notes": "Detected via Python backend"
        } if drift else {"present": False}

        return {
            "dominant_type": dominant_type,
            "structures": structures,
            "emotional_state": parsed.get("emotional_state", emotional_state),
            "causal_links": parsed.get("causal_links", []),
            "existential_drift": existential_drift
        }

    except Exception as e:
        print("❌ analyze_input failed:", e)
        return {
            "dominant_type": "unknown",
            "structures": [],
            "emotional_state": [],
            "causal_links": [],
            "existential_drift": {"present": False}
        }










@app.route("/register", methods=["POST"])
def register_user():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password are required!"}), 400

    # Load or create userpp.json
    os.makedirs("user_data", exist_ok=True)
    userpp_path = "userpp.json"
    if os.path.exists(userpp_path):
        with open(userpp_path, "r") as f:
            user_db = json.load(f)
    else:
        user_db = {}

    if username in user_db:
        return jsonify({"error": "User already exists!"}), 400

    # ✅ Save new user to userpp.json
    user_db[username] = {"password": password}
    with open(userpp_path, "w") as f:
        json.dump(user_db, f, indent=2)

    # ✅ Create user history
    user_history_path = f"user_data/{username}_history.json"
    empty_history = {
        "user_input": [],
        "ai_response": [],
        "dialogue": [],
        "active_structures": [],
        "resolved": [],
        "archived": [],
        "threads": [],
        "map_links": []
    }
    with open(user_history_path, "w") as f:
        json.dump(empty_history, f, indent=2)

    # ✅ Create thought map
    os.makedirs("thought_maps", exist_ok=True)
    default_map = {
        "agency": {"beliefs": [], "tensions": [], "unresolved": False},
        "identity": {"beliefs": [], "tensions": [], "unresolved": False},
        "values": {"beliefs": [], "tensions": [], "unresolved": False},
        "fulfillment": {"beliefs": [], "tensions": [], "unresolved": False},
        "desires": {"beliefs": [], "tensions": [], "unresolved": False},
        "relationships": {"beliefs": [], "tensions": [], "unresolved": False}
    }
    with open(f"thought_maps/{username}.json", "w") as f:
        json.dump(default_map, f, indent=2)

    return jsonify({"message": f"User {username} registered successfully!"})

















@app.route("/login", methods=["POST"])
def login_user():
    """Handle user login."""
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password are required!"}), 400

    # Check if user exists and password matches
    if check_user_exists(username):
        with open('userpp.json', 'r') as f:
            data = json.load(f)
            if data.get(username, {}).get("password") == password:
                # Instead of redirect, send a JSON response with a success message
                return jsonify({"message": "Login successful!"}), 200
            else:
                return jsonify({"error": "Incorrect password!"}), 401
    else:
        return jsonify({"error": "User not found!"}), 404



def check_user_exists(username):
    return os.path.exists(f"user_data/{username}_history.json")




@app.route("/pocketphilosopher")
def pocket_philosopher():
    """Serve the Pocket Philosopher page."""
    return render_template("pocketphilosopher.html")


@app.route("/")
def home():
    return render_template("login_register.html")













def is_resolvable(parsed_json: dict, user_history: dict = None) -> bool:
    
    
    if not parsed_json or not isinstance(parsed_json, dict):
        print("⚠️ is_resolvable() received None or invalid parsed_json.")
        return False

    if "structures" not in parsed_json:
        print("⚠️ No structures found in parsed_json.")
        return False

    contradiction_structures = [
        s for s in parsed_json["structures"]
        if s.get("type") == "contradiction" and s.get("confidence", 0) >= 0.75
    ]

    if len(contradiction_structures) != 1:
        return False

    # Avoid resolution if drift or disillusionment are present
    existential_drift = parsed_json.get("existential_drift")
    if isinstance(existential_drift, dict) and existential_drift.get("present"):
        return False

    emotional_state = parsed_json.get("emotional_state", [])
    if isinstance(emotional_state, list) and "disillusionment" in emotional_state:
        return False

    # Require some prior conversation for depth
    if user_history:
        user_inputs = user_history.get("user_input", [])
        if len(user_inputs) < 2:
            return False

    return True







def load_user_history(username):
    path = os.path.join("user_data", f"{username}_history.json")

    # Default structure (new format)
    default_history = {
        "dialogue": [],
        "active_structures": [],
        "inactive_structures": [],
        "map_links": [],
        "causal_links": []
    }

    if not os.path.exists(path):
        return default_history

    try:
        with open(path, 'r') as f:
            data = json.load(f)

            # If it's an old-format file, migrate keys
            migrated = {
                "dialogue": data.get("dialogue", []),
                "active_structures": data.get("active", data.get("active_structures", [])),
                "inactive_structures": data.get("resolved", data.get("inactive_structures", [])) + data.get("archived", []),
                "map_links": data.get("map_links", []),
                "causal_links": data.get("causal_links", [])
            }

            return migrated

    except (json.JSONDecodeError, IOError):
        print(f"⚠️ Failed to load user history for {username}, returning blank.")
        return default_history




def mark_structure_as_resolved(structure, username, user_input, parsed_analysis):


    history = load_user_history(username)
    unresolved = history.get("unresolved_structures", [])
    resolved = history.get("resolved_structures", [])

    matched_structure = None
    for s in unresolved:
        if s['source_snippet'] == structure.get('source_snippet'):
            matched_structure = s
            break

    if matched_structure:
        unresolved.remove(matched_structure)
        matched_structure['resolved'] = True
        resolved.append(matched_structure)

    history['resolved_structures'] = resolved
    history['unresolved_structures'] = unresolved
    save_user_history_with_links(username, user_input, parsed_analysis, user_data_override=None, resolved_structures=None, map_links=None)




def compose_final_response(parsed_json, strategy_notes, username, return_map_delta=False, attempting_resolution=False):
    import json

    if not isinstance(parsed_json, dict):
        print("❌ Invalid parsed_json")
        fallback = {
            "response": "I'm having trouble making sense of that. Could you share your thought in a different way?",
            "map_delta": None
        }
        return (fallback["response"], fallback["map_delta"], None, None) if return_map_delta else fallback["response"]

    user_history = load_user_history(username)
    dialogue_history = user_history.get("dialogue", [])[-6:]
    conversation_history = "\n".join([
        f"User: {entry['user']}\nAI: {entry['ai']}" for entry in dialogue_history
    ]) or "(No prior conversation)"

    user_input = strategy_notes.get("raw_input", "")
    response_mode = strategy_notes.get("response_mode", "standard_reflection")
    top_structure = strategy_notes.get("top_structure")
    thread_focus = strategy_notes.get("thread_focus")
    thought_map = strategy_notes.get("thought_map", {})

    structure_summary = ""
    if top_structure:
        structure_summary += f"- Current focus: {top_structure.get('type')} → {top_structure.get('summary')}\n"
    if thread_focus:
        structure_summary += f"- Long-term thread: {thread_focus.get('type')} → {thread_focus.get('summary')}\n"
    if not structure_summary:
        structure_summary = "(No structure focus detected)"

    structure_instructions = {
        "hold_tension": "1. Reflect the opposing forces.\n2. Validate both sides.\n3. Let them sit with it.\n4. Ask something that might help them resolve it.",
        "name_uncertainty": "1. Acknowledge drift.\n2. Name the groundlessness.\n3. Ask what might bring them one reflective, open-ended, question.",
        "clarify_language": "1. Point out the conflation.\n2. Invite distinction.\n3. Offer an analogy that might reframe the conflation for them without resolving it for them.",
        "resolve_contradiction": "1. Name the contradiction.\n2. Suggest a synthesis.\n3. End with insight, not a question.",
        "reflective_probe": "1. Surface subtle uncertainty.\n2. Frame gently.\n3. Ask an exploratory, open-ended, question meant to push the conversation forward.",
        "standard_reflection": "1. Name the emotional tone.\n2. Reflect the tension.\n3. Offer gentle framing.\n4. Use one of several strategies to deepen reflection: Offer a fresh analogy (avoid gardens, oceans, books, or paintings), Construct a thought experiment tailored to their input, Distill the core conceptual tension or belief conflict, Ask a Socratic-style question that tests an implicit assumption, Flip or invert the premise to reveal a new angle, End with a single open-ended question that reflects on the strategy used"
    }

    structure_text = structure_instructions.get(response_mode, structure_instructions["standard_reflection"])
    notes_json = json.dumps(strategy_notes, indent=2)
    map_json = json.dumps(thought_map, indent=2)

    reflection_prompt = f"""
You are Pocket Philosopher — a personal thinking assistant who helps users reflect on their tensions, beliefs, and sense of meaning.

Write one **concise paragraph (2–4 sentences)** that:
- Gently re-engages the tension or philosophical theme
- Validates the complexity of the user’s experience
- Use **one** of several strategies to deepen reflection — you’re especially encouraged to:
  - Construct a thought experiment tailored to their input
  - Flip or invert the premise to reveal a new angle
  - Ask a Socratic-style question that challenges an assumption
  - Distill the core conceptual tension or belief conflict
  - (Use analogies sparingly; avoid gardens, oceans, books, or paintings unless novel and precise)


--- USER INPUT ---
"{user_input[:300]}..."

--- RESPONSE MODE ---
{response_mode}

--- STRUCTURE FOCUS ---
{structure_summary}

--- RECENT CONVERSATION ---
{conversation_history}

--- STRATEGY NOTES (raw) ---
{notes_json}

--- STRUCTURE INSTRUCTIONS ---
{structure_text}
""".strip()

    recent_turns = user_history.get("dialogue", [])[-4:]
    dialogue_block = "\n\n".join([
        f"User: {t['user']}\nAI: {t['ai']}" for t in recent_turns
    ])

    mapping_prompt = f"""
You are Pocket Philosopher.

Based on the following exchange and user input, extract **any new beliefs or tensions** that relate to:

- Identity
- Fulfillment
- Desires
- Values
- Agency
- Relationships

Return only this format:
{{
  "response": "...",
  "map_delta": {{
    "agency": {{ "beliefs": [...], "tensions": [...] }},
    "identity": {{ "beliefs": [...], "tensions": [...] }},
    "values": {{ "beliefs": [...], "tensions": [...] }},
    "fulfillment": {{ "beliefs": [...], "tensions": [...] }},
    "desires": {{ "beliefs": [...], "tensions": [...] }},
    "relationships": {{ "beliefs": [...], "tensions": [...] }}
  }}
}}

--- RECENT CONVERSATION ---
{dialogue_block}

--- CURRENT USER INPUT ---
"{user_input}"

--- CURRENT THOUGHT MAP ---
{map_json}

Important: Do NOT explain anything. Return valid JSON only.
""".strip()

    try:
        reflection_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": reflection_prompt}],
            max_tokens=500,
            temperature=0.75
        )
        final_response = reflection_response.choices[0].message.content.strip()
        print("📘 Reflection:", final_response)

        map_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": mapping_prompt}],
            max_tokens=500,
            temperature=0.5
        )
        raw_json = map_response.choices[0].message.content.strip()
        print("📦 Raw Map GPT:", raw_json)

        try:
            map_delta = json.loads(raw_json).get("map_delta", {})
        except json.JSONDecodeError:
            print("⚠️ Map delta not valid JSON.")
            map_delta = {}

    except Exception as e:
        print("❌ GPT call error:", e)
        final_response = "Something didn’t quite parse. Could you rephrase that?"
        map_delta = {}

    if return_map_delta:
        return final_response, map_delta, top_structure, thread_focus
    return final_response





def refine_analysis(existing_analysis, new_analysis, archived=None, dormant=None):
    """Refine the existing analysis by checking if any structures should be updated or skipped."""
    if archived is None:
        archived = []
    if dormant is None:
        dormant = []

    archived_summaries = {s.get("summary") for s in archived if isinstance(s, dict)}
    dormant_summaries = {s.get("summary") for s in dormant if isinstance(s, dict)}
    existing_summaries = {s.get("summary") for s in existing_analysis if isinstance(s, dict)}

    updated_analysis = []

    for new in new_analysis.get("structures", []):
        if not isinstance(new, dict):
            continue

        summary = new.get("summary")
        if not summary:
            continue

        # Skip if already resolved
        if new.get("resolved"):
            continue

        # Skip if already archived or dormant
        if summary in archived_summaries or summary in dormant_summaries:
            print(f"⏭️ Skipping reintroduction of {summary} (archived/dormant)")
            continue

        # If duplicate of active structure, update it (confidence-based)
        matched = False
        for existing in updated_analysis:
            if existing.get("summary") == summary:
                matched = True
                if new.get("confidence", 0) > existing.get("confidence", 0):
                    existing.update(new)
                break

        if not matched and summary not in existing_summaries:
            updated_analysis.append(new)

    return updated_analysis




def check_contradiction_resolution(previous_input, current_input):
    # Look for a shift from one perspective to another
    if "realize" in current_input.lower() or "now understand" in current_input.lower():
        # Look for a direct shift in reasoning or conclusion
        if "should" in previous_input and "value" in current_input:
            return True
    return False



resolution_model = SentenceTransformer("all-MiniLM-L6-v2")

def resolve_structures_with_input(username, user_input, parsed, prior_unresolved=None, map_links=None, skip_if_same_input=True):
    if prior_unresolved is None:
        try:
            all_data = load_user_history(username)
            unresolved = all_data.get(username, {}).get("unresolved_structures", [])
        except Exception as e:
            print(f"⚠️ Failed to load history for resolution: {e}")
            return []
    else:
        unresolved = prior_unresolved

    if map_links is None:
        try:
            map_links = load_recent_map_links(username)
        except:
            map_links = []

    resolved_now = []
    new_summaries = {s.get("summary", "") for s in parsed.get("structures", [])}

    for struct in unresolved:
        if skip_if_same_input and struct.get("summary", "") in new_summaries:
            print(f"🛑 Skipping resolution for just-introduced: {struct['summary']}")
            continue

        if struct["type"] not in {s["type"] for s in parsed.get("structures", [])}:
            continue

        old_snip = ""
        if isinstance(struct.get("source_snippet"), dict):
            left = struct["source_snippet"].get("left", "")
            right = struct["source_snippet"].get("right", "")
            old_snip = f"{left} vs. {right}".lower()
        else:
            old_snip = struct.get("source_snippet", "").lower()

        new_snip = user_input.lower()
        sim_score = SequenceMatcher(None, old_snip, new_snip).ratio()

        belief_links = [
            l for l in map_links
            if l.get("source_structure_id") == struct.get("id") and l.get("type") == "belief"
        ]

        if sim_score > 0.65 or len(belief_links) >= 2:
            struct["resolved"] = True
            struct["resolution_snippet"] = user_input
            print(f"✅ Structure resolved: {struct['summary']}")
            resolved_now.append(struct)

    return resolved_now




def load_recent_map_links(username, max_links=10):
    try:
        history = load_user_history(username)
        return history.get("map_links", [])[-max_links:]
    except:
        return []



















def generate_resolution_synthesis_gpt(reflection, structure, thought_map, username):
    prompt = f"""
You are a philosophical conversation guide. A user has just resolved a previously discussed tension or belief structure. Your job is to:

1. Acknowledge and affirm their resolution with emotional and intellectual empathy.
2. Summarize the resolved tension in natural, gentle language.
3. Tie it to a belief or theme from their current thought map (if relevant).
4. Offer either:
   - A new open-ended question to continue the reflection in a meaningful way,
   - Or a soft suggestion for another possible tension they might want to explore.

Avoid repeating the user's resolution phrase verbatim unless it's particularly expressive. Keep your tone warm, thoughtful, and lightly curious.

### Reflection that led to resolution:
{reflection}

### Structure resolved:
{json.dumps(structure, indent=2)}

### Thought map:
{json.dumps(thought_map, indent=2)}

Respond with a single paragraph of reflection.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=250,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("❌ GPT API call failed:", e)
        return "I'm glad that made sense to you. Is there anything else you'd like to explore or clarify next?"









# Simplified GPT wrapper for probing purposes
def chat_with_gpt(prompt: str, model="gpt-4-turbo", temperature=0.7) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ GPT probing failed:", e)
        return "Could you say that a different way?"

# Optional clean-up hook if you need it
def extract_response_text(raw: str) -> str:
    return raw.strip() if isinstance(raw, str) else ""




def generate_initial_probe(user_input: str, username: str) -> str:
    user_history = load_user_history(username)
    dialogue = user_history.get("dialogue", [])[-5:]
    conversation_history = "\n\n".join(
        [f"User: {d.get('user', '')}\nAI: {d.get('ai', '')}" for d in dialogue]
    ) or "(No prior conversation)"

    # 🧠 History-wide structure/emotion context
    full_pass = detect_all_structures_full_history(user_history.get("user_input", []))
    thread_focus, _ = weigh_structures(full_pass)
    emotion_summary = ", ".join(f"{k}: {round(v, 2)}" for k, v in full_pass.get("emotional_state", {}).items()) or "none"
    thread_summary = thread_focus.get("summary") if thread_focus else "none"

    prompt = f"""
You're Pocket Philosopher.

The user just said:
\"{user_input}\"

No formal structure was detected in this message (e.g. contradiction, value conflict, drift).
Your goal is to ask **one simple, open-ended question** to help surface whatever might be underneath.

--- RECENT CONVERSATION ---
{conversation_history}

--- CUMULATIVE THREAD CONTEXT ---
- Thread-level theme: {thread_summary}
- Cumulative emotional tones: {emotion_summary}

Write one **short, sincere, open-ended question** that:
- Gently invites the user to expand or clarify their statement
- Might reveal an underlying belief, conflict, or uncertainty
- Helps you better understand what they meant or why it matters

Do not summarize, explain, or reflect. Ask only one question.
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=200,
            temperature=0.6
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ GPT error generating initial probe:", e)
        return "Could you say a bit more about what that means to you?"





def infer_map_domain(struct):
    summary = struct.get("summary", "").lower()
    if "purpose" in summary or "meaning" in summary:
        return "fulfillment"
    elif "action" in summary or "agency" in summary:
        return "agency"
    elif "relationships" in summary:
        return "relationships"
    return "identity"




def update_last_seen_turns(history):
    if "dialogue" not in history:
        return history

    id_last_seen = {}

    for turn, entry in enumerate(history["dialogue"]):
        sid = entry.get("structure_id")
        if sid:
            id_last_seen[sid] = turn  # overwrite with most recent

    def fix_list(structures):
        for s in structures:
            sid = s.get("id")
            if sid in id_last_seen:
                s["last_seen_turn"] = id_last_seen[sid]
            elif "created_turn" in s:
                s.setdefault("last_seen_turn", s["created_turn"])
        return structures

    history["active"] = fix_list(history.get("active", []))
    history["resolved"] = fix_list(history.get("resolved", []))
    return history






def archive_stale_structures(history, current_turn, max_age=10):
    still_active = []
    moved_to_archive = []

    for s in history["active"]:
        age = current_turn - s.get("last_seen_turn", s.get("created_turn", 0))
        if not s.get("resolved", False) and age > max_age:
            moved_to_archive.append(s)
        else:
            still_active.append(s)

    history["active"] = still_active
    history.setdefault("archived", []).extend(moved_to_archive)
    return history






def determine_response_mode(parsed_analysis, user_data, resolved_now):
    if parsed_analysis is None:
        print("❌ parsed_analysis is None — defaulting to standard_reflection.")
        return "standard_reflection"

    dominant_type = parsed_analysis.get("dominant_type", "").lower()

    if resolved_now:
        return "resolve_contradiction"
    
    if is_resolvable(parsed_analysis, user_data):
        return "resolve_contradiction"

    # Add logic based on detected dominant structure
    if dominant_type == "contradiction":
        return "hold_tension"
    elif dominant_type == "value_conflict":
        return "hold_tension"
    elif dominant_type == "existential_drift":
        return "name_uncertainty"
    elif dominant_type == "conflation":
        return "clarify_language"

    # No clear structure — fallback to reflection
    return "standard_reflection"





def get_dominant_type(structures: list[dict]) -> str:
    """Return the most common structure type in a list of structures."""
    from collections import Counter
    if not structures:
        return "none"
    type_counts = Counter(s.get("type", "unknown") for s in structures)
    dominant = type_counts.most_common(1)[0][0]
    return dominant





def detect_all_structures_full_history(user_inputs: List[str]) -> Dict:
    """Analyze all prior user inputs for cumulative structure and emotion patterns."""
    from collections import defaultdict
    import uuid

    all_structures = []
    all_causal_links = []
    emotion_accumulator = defaultdict(float)
    all_emotion_sources = []
    seen_summaries = set()

    for i, thought in enumerate(user_inputs):
        result = detect_all_structures(thought)

        for s in result.get("structures", []):
            summary = s.get("summary", "")
            if summary not in seen_summaries:
                seen_summaries.add(summary)
                s = s.copy()
                s["id"] = str(uuid.uuid4())
                s["created_turn"] = i
                s["last_seen_turn"] = i
                s["resolved"] = False
                s["attempts"] = 0
                all_structures.append(s)

        for link in result.get("causal_links", []):
            all_causal_links.append(link)

        for k, v in result.get("emotional_state", {}).items():
            emotion_accumulator[k] += v

        all_emotion_sources.extend(result.get("emotion_sources", []))

    # Normalize emotional scores by number of inputs
    num_inputs = len(user_inputs) or 1
    averaged_emotion_state = {
        k: round(v / num_inputs, 4) for k, v in emotion_accumulator.items()
    }

    return {
        "structures": all_structures,
        "causal_links": all_causal_links,
        "emotional_state": averaged_emotion_state,
        "emotion_sources": all_emotion_sources,
        "dominant_type": get_dominant_type(all_structures)
    }






def generate_structure_summary(structure):
    s_type = structure.get("type", "")
    snippet = (structure.get("source_snippet") or "").strip()


    if not snippet:
        return "(unspecified)"

    if s_type == "contradiction":
        parts = re.split(r"\bbut\b|\bhowever\b|\byet\b", snippet, flags=re.IGNORECASE)
        if len(parts) == 2:
            return f"Contradiction between: “{parts[0].strip()}” vs. “{parts[1].strip()}”"
        return f"Contradiction detected in: “{snippet}”"

    elif s_type == "value_assertion":
        return f"User asserts a core value in: “{snippet}”"

    elif s_type == "reinforcement":
        return f"Belief reinforced in: “{snippet}”"

    elif s_type == "existential_drift":
        return f"Drift or uncertainty expressed in: “{snippet}”"

    elif s_type == "value_conflict":
        return f"Conflict between personal values in: “{snippet}”"

    elif s_type == "conflation":
        return f"Possible conflation in: “{snippet}”"

    return f"{s_type.replace('_', ' ').title()} detected in: “{snippet}”"


def generate_summaries_if_missing(structures):
    for s in structures:
        if not s.get("summary"):
            s["summary"] = generate_structure_summary(s)
    return structures




def matches_resolution_phrases(text: str) -> bool:
    patterns = [
        r"\b(that makes sense|i feel better|you're right|i get it now|i needed that|not worried anymore|i'm okay with it|i've decided)\b"
    ]
    text = text.lower()
    return any(re.search(pat, text) for pat in patterns)











@app.route("/reflect", methods=["POST"])
def reflect():
    data = request.get_json()
    user_input = data.get("thought", "").strip()
    username = data.get("username", "")
    resolved_now = []
    tensions_resolved = []

    thought_map = load_thought_map(username)
    user_data = load_user_history(username)


    
    full_history_analysis = detect_all_structures_full_history(user_data.get("user_input", []))
    full_structures = full_history_analysis.get("structures", [])
    full_emotions = full_history_analysis.get("emotional_state", {})
    thread_focus, _ = weigh_structures(full_history_analysis)

    last_dialogue = user_data.get("dialogue", [])[-1] if user_data.get("dialogue") else None
    last_structure_id = last_dialogue.get("structure_id") if last_dialogue else None
    last_active_structures = user_data.get("active_structures", [])

# ✅ Try resolving by ID match first
    if last_structure_id:
        for struct in last_active_structures:
            if struct.get("id") == last_structure_id and matches_resolution_phrases(user_input):
                print("🎯 Resolved previously reflected structure (via ID):", struct.get("summary"))
                struct["resolved"] = True
                struct["resolved_on_turn"] = len(user_data.get("user_input", [])) + 1

                struct["resolution_snippet"] = user_input
                resolved_now.append(struct)

# ✅ Fallback: resolve by summary if ID doesn’t match
    if not resolved_now and last_dialogue:
        last_summary = last_dialogue.get("structure_summary")
        for struct in last_active_structures:
            if (
                struct.get("summary") == last_summary
                and matches_resolution_phrases(user_input)
                and not struct.get("resolved")
            ):
                print("🎯 Resolved previously reflected structure (via summary match):", struct.get("summary"))
                struct["resolved"] = True
                struct["resolved_on_turn"] = len(user_data.get("user_input", [])) + 1

                struct["resolution_snippet"] = user_input
                resolved_now.append(struct)


    # ✅ Fallback: resolve by type if both ID and summary failed
    if not resolved_now and last_dialogue and matches_resolution_phrases(user_input):
        summary = last_dialogue.get("structure_summary", "").lower()
        structure_type = None

        if "contradiction" in summary:
            structure_type = "contradiction"
        elif "value" in summary:
            structure_type = "value_assertion"
        elif "drift" in summary:
            structure_type = "existential_drift"
        elif "uncertainty" in summary:
            structure_type = "uncertainty"
        elif "conflation" in summary:
            structure_type = "conflation"

        for struct in last_active_structures:
            if struct.get("type") == structure_type and not struct.get("resolved"):
                print("🎯 Resolved previously reflected structure (via type fallback):", struct.get("summary"))
                struct["resolved"] = True
                struct["resolved_on_turn"] = len(user_data.get("user_input", [])) + 1

                struct["resolution_snippet"] = user_input
                resolved_now.append(struct)
                break






        if resolved_now:
    # ✅ Move resolved structures to inactive
            if "inactive_structures" not in user_data:
                user_data["inactive_structures"] = []
            for struct in resolved_now:
                if struct in user_data["active_structures"]:
                    user_data["active_structures"].remove(struct)
                user_data["inactive_structures"].append(struct)

    # Instead of going to GPT, generate synthesis
            last_reflection = last_dialogue.get("ai") if last_dialogue else ""
            response = generate_resolution_synthesis_gpt(last_reflection, resolved_now[0], thought_map, username)

            user_data = save_user_history_with_links(
                username=username,
                user_input=user_input,
                parsed_analysis={},
                user_data_override=user_data,
                resolved_structures=resolved_now,
                ai_response=response
            )

        return jsonify({
            "response": response,
            "followup": "",
            "thought_map": thought_map,
            "active_structure": None,
            "thread_focus": None,
            "tensions_resolved": resolved_now
        })




    structure_analysis = detect_all_structures(user_input)
    structures = structure_analysis.get("structures", [])







    previous_unresolved = user_data.get("active_structures", [])
    positive_signals = [
        "less anxious", "more confident", "feel better", "you're right", "exactly right",
        "thank you", "not as anxious", "feeling good", "stronger than I thought",
        "resolved", "comfortable", "better now"
    ]

    if any(p in user_input.lower() for p in positive_signals):
        contradiction = next((s for s in reversed(previous_unresolved)
                              if s.get("type") == "contradiction" and not s.get("resolved")), None)
        if contradiction:
            contradiction["resolved"] = True
            contradiction["resolution_snippet"] = user_input
            resolved_now = [contradiction]
    else:
        resolved_now = resolve_structures_with_input(
            username, user_input, structure_analysis,
            prior_unresolved=previous_unresolved,
            skip_if_same_input=True
        )

    current_unresolved = [s for s in previous_unresolved if not s.get("resolved")]
    for s in structures:
        if not any(s["summary"] == u["summary"] for u in current_unresolved):
            s["attempts"] = 0
            s["resolved"] = False
            current_unresolved.append(s)

    parsed_analysis = analyze_input(username, user_input)
    parsed = parsed_analysis
    structures = parsed.get("structures", [])

    if parsed_analysis and "structures" in parsed_analysis:
        parsed_analysis["structures"] = generate_summaries_if_missing(parsed_analysis["structures"])

    structures = generate_summaries_if_missing(structures)
    structure_analysis["structures"] = structures









    if not resolved_now and not structures and full_emotions:


        response = generate_reflective_probe(user_input, full_emotions, username)

        user_data = save_user_history_with_links(
            username=username,
            user_input=user_input,
            parsed_analysis={},
            user_data_override=user_data,
            ai_response=response
        )
        print("❓ Using reflective probe — emotion detected but no structure.")
        return jsonify({
            "response": response,
            "followup": "",
            "thought_map": thought_map,
            "active_structure": None,
            "thread_focus": None,
            "tensions_resolved": []
        })

    if parsed_analysis is None or (not parsed_analysis.get("structures") and not is_resolvable(parsed_analysis, user_data)):
        response = generate_initial_probe(user_input, username)
        user_data = save_user_history_with_links(
            username=username,
            user_input=user_input,
            parsed_analysis={},
            user_data_override=user_data,
            ai_response=response
        )
        print("❓ Using initial probe — no structure or strong emotional signal detected.")
        return jsonify({
            "response": response,
            "followup": "",
            "thought_map": thought_map,
            "active_structure": None,
            "thread_focus": None,
            "tensions_resolved": []
        })

    if resolved_now:
        for s in resolved_now:
            domain = infer_map_domain(s) or "identity"
            belief = {
                "summary": f"Resolved tension between: {s['summary']}",
                "source": s.get("resolution_snippet", user_input),
                "confidence": s.get("confidence", 0.8),
                "timestamp": str(datetime.now())
            }
            thought_map[domain]["beliefs"].append(belief)
        save_thought_map(username, thought_map)

    user_data = update_structure_relevance(user_input, user_data)
    user_data.setdefault("active_structures", [])
    turn_index = len(user_data.get("user_input", []))
    for s in current_unresolved:
        s.setdefault("created_turn", turn_index)
        s["last_seen_turn"] = turn_index
        if not any(s["summary"] == a["summary"] for a in user_data["active_structures"]):
            user_data["active_structures"].append(s)

    user_data = archive_stale_structures(user_data, turn_index)
    user_data = update_last_seen_turns(user_data)

    response_mode = determine_response_mode(parsed_analysis, user_data, resolved_now)
    top_structure, deferred_structures = weigh_structures(parsed_analysis)
    emotion_sources = trace_emotion_sources(structures)

    strategy_notes = {
        "top_structure": top_structure,
        "thread_focus": thread_focus,
        "deferred": deferred_structures,
        "causal_links": parsed_analysis.get("causal_links", []),
        "emotional_state": structure_analysis.get("emotional_state", []),
        "emotion_sources": emotion_sources,
        "resolved_structures": resolved_now,
        "raw_input": user_input,
        "thought_map": thought_map,
        "recent_inputs": user_data.get("user_input", [])[-3:],
        "unresolved": user_data.get("active_structures", [])[-5:],
        "threads": user_data.get("threads", [])[-3:],
        "response_mode": response_mode
    }

    try:
        final_reply, map_delta, active_structure, _ = compose_final_response(
            parsed_analysis,
            strategy_notes,
            username,
            return_map_delta=True,
            attempting_resolution=is_resolvable(parsed_analysis, user_data)
        )
    except Exception as e:
        print("❌ GPT generation error:", e)
        final_reply = "Something didn’t quite parse. Could you rephrase that?"
        map_delta = {}
        active_structure = None

    def ensure_summary(struct):
        if struct and not struct.get("summary"):
            if not struct.get("source_snippet") and user_input:
                struct["source_snippet"] = user_input
            struct["summary"] = generate_structure_summary(struct)

    ensure_summary(top_structure)
    ensure_summary(active_structure)

    map_links = []
    if map_delta:
        thought_map, map_links = apply_map_delta_with_links(
            thought_map, map_delta,
            unresolved_structures=current_unresolved,
            username=username
        )
        save_thought_map(username, thought_map)

        for domain in ["fulfillment", "identity", "values", "desires"]:
            tensions = thought_map[domain].get("tensions", [])
            for belief in map_delta.get(domain, {}).get("beliefs", []):
                belief_summary = belief if isinstance(belief, str) else belief.get("summary", "")
                for tension in tensions[:]:
                    tension_summary = tension.get("summary", "")
                    if all(word in belief_summary.lower() for word in tension_summary.lower().split(" vs. ")):
                        tensions_resolved.append({
                            "domain": domain,
                            "tension": tension,
                            "resolved_by": belief
                        })
                        thought_map[domain]["tensions"].remove(tension)

    followup = ""
    if backend_wants_to_continue_dialogue(parsed_analysis, strategy_notes, user_input, final_reply):
        followup = generate_followup(parsed_analysis, strategy_notes, user_input, username, thought_map)

    actual_reply = followup or final_reply

    # ✅ CLEAN ID MAPPING (REPLACES OLD BLOCK)
    def match_original_structure(structure, user_data):
        if not structure:
            return None
        for s in user_data.get("active_structures", []):
            if s.get("summary") == structure.get("summary") and s.get("type") == structure.get("type"):
                return s
    # Fallback: assign ID if it was missing
        if "id" not in structure:
            structure["id"] = str(uuid.uuid4())
        return structure

    active_structure = match_original_structure(active_structure, user_data)
    top_structure = match_original_structure(top_structure, user_data)

    target_struct = match_original_structure(active_structure or top_structure, user_data)

    if target_struct:
        parsed_analysis["structure_id"] = target_struct.get("id")
        parsed_analysis["structure_summary"] = target_struct.get("summary")


    user_data = save_user_history_with_links(
        username=username,
        user_input=user_input,
        parsed_analysis=parsed_analysis,
        active_structures=structures,
        resolved_structures=resolved_now,
        map_links=map_links,
        ai_response=actual_reply
    )


    if followup:
        return jsonify({
            "response": followup,
            "followup": "",
            "thought_map": thought_map,
            "active_structure": active_structure,
            "thread_focus": thread_focus,
            "tensions_resolved": tensions_resolved
    })
    else:
        return jsonify({
            "response": final_reply,
            "followup": "",
            "thought_map": thought_map,
            "active_structure": active_structure,
            "thread_focus": thread_focus,
            "tensions_resolved": tensions_resolved
    })













def backend_wants_to_continue_dialogue(parsed, strategy_notes, user_input, final_response):
    unresolved = strategy_notes.get("unresolved_structures", [])
    resolved = strategy_notes.get("resolved_structures", [])

    def is_ambiguous(text):
        phrases = [
            "i don't know", "what do you mean", "maybe",
            "i guess", "sure", "okay", "whatever"
        ]
        return any(p in text.lower() for p in phrases)

    if unresolved or resolved:
        return True
    if len(user_input.strip()) < 25 or is_ambiguous(user_input):
        return True
    if not final_response.strip().endswith("?"):
        return True

    return False




def generate_reflective_probe(user_input: str, emotion_scores: dict, username: str, thread_focus: dict = None) -> str:
    user_history = load_user_history(username)
    dialogue = user_history.get("dialogue", [])[-5:]

    history = [f"User: {turn.get('user', '(no user input)')}\nAI: {turn.get('ai', '(no AI reply)')}" for turn in dialogue]
    conversation_history = "\n\n".join(history) if history else "(No prior conversation)"

    emotion_text = ", ".join(f"{k}: {round(v, 2)}" for k, v in emotion_scores.items()) or "none"
    thread_summary = thread_focus.get("summary") if thread_focus else None
    thread_line = f"Longer-term philosophical theme: \"{thread_summary}\"" if thread_summary else "No dominant theme detected."

    # 🧠 Prompt: generate multiple candidates, then choose one
    prompt = f"""
You are Pocket Philosopher, a gentle and emotionally intelligent reflection guide.

The user just said:
\"{user_input}\"

Their emotional tone and phrasing suggest something deeper is present — even though no formal contradiction or structure was found.

Cumulative Emotion Signals: {emotion_text}
{thread_line}

--- RECENT CONVERSATION HISTORY (use this to detect themes, tensions, and emotional patterns) ---
{conversation_history}

Your task is to:

1. Analyze what the user may be wrestling with emotionally or philosophically — not just from their most recent statement, but in light of the last few exchanges. Are they hesitating? Avoiding something? Showing doubt or conflict?

2. Based on that, generate **three emotionally and intellectually distinct follow-up questions** that:
   - Offer varied interpretive framings
   - Encourage meaningful self-reflection
   - Avoid yes/no phrasing or mirroring their exact words

3. Select the most insightful and relevant question, and return ONLY that question.

Use varied tones. One can be gentle, one more curious, one more philosophical.
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=300,
            temperature=0.85
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ GPT error generating reflective probe:", e)
        return "What feels unresolved or unclear about that for you?"








def generate_followup(parsed, strategy_notes, user_input, username, thought_map=None):
    import json

    user_history = load_user_history(username)
    unresolved = user_history.get("active_structures", [])
    resolved = strategy_notes.get("resolved_structures", [])
    thread_focus = strategy_notes.get("thread_focus", None)
    emotion_scores = strategy_notes.get("emotional_state", {})

    dialogue = user_history.get("dialogue", [])[-4:]
    conversation_history = "\n\n".join(
        [f"User: {d.get('user', '')}\nAI: {d.get('ai', '')}" for d in dialogue]
    ) or "(No prior conversation)"

    # 🧠 Format cumulative emotion text
    emotion_text = ", ".join(f"{k}: {round(v, 2)}" for k, v in emotion_scores.items()) or "none"

    if unresolved:
        s = unresolved[0]
        prompt = f"""
The user has an unresolved philosophical or emotional tension:

- Type: {s.get('type')}
- Summary: {s.get('summary')}
- Originally triggered by: \"{s.get('source_snippet', 'unknown')}\"
- Latest Input: \"{user_input}\"
- Emotion signals: {emotion_text}
- Thread-level focus: {thread_focus.get('summary') if thread_focus else "None"}

--- CONVERSATION HISTORY ---
{conversation_history}

Write one **concise paragraph (2–4 sentences)** that:
- Gently re-engages the tension or philosophical theme
- Validates the complexity of the user’s experience
- Use one of several strategies to deepen reflection:
  - Offer a fresh analogy (avoid gardens, oceans, books, or paintings)
  - Construct a thought experiment tailored to their input
  - Distill the core conceptual tension or belief conflict
  - Ask a Socratic-style question that tests an implicit assumption
  - Flip or invert the premise to reveal a new angle
- End with a single open-ended question that reflects on the strategy used.
""".strip()


    else:
        prompt = f"""
The user just said: \"{user_input}\"

There are no clearly unresolved structures at this time.

--- CONVERSATION HISTORY ---
{conversation_history}

Their current thought map is:
{json.dumps(thought_map or {}, indent=2)}

Cumulative emotional cues: {emotion_text}
Longer-term thread focus: {thread_focus.get('summary') if thread_focus else "None"}

Write one **concise paragraph (2–4 sentences)** that:
- Gently re-engages the tension or philosophical theme
- Validates the complexity of the user’s experience
- Use one of several strategies to deepen reflection:
  - Offer a fresh analogy (avoid gardens, oceans, books, or paintings)
  - Construct a thought experiment tailored to their input
  - Distill the core conceptual tension or belief conflict
  - Ask a Socratic-style question that tests an implicit assumption
  - Flip or invert the premise to reveal a new angle
- End with a single open-ended question that reflects on the strategy used.
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=450,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ GPT error generating follow-up:", e)
        return "Is there something in this that feels worth exploring more?"







def explain_emotion(parsed_json):
    emotional_state = parsed_json.get("emotional_state", [])
    causal_links = parsed_json.get("causal_links", [])
    structures = parsed_json.get("structures", [])

    if not emotional_state:
        return ""

    emotion = emotional_state[0]  # prioritize first emotion for now

    cause = None
    for link in causal_links:
        cause = link.get("from")
        if cause:
            break

    cause_detail = next((s for s in structures if s.get("type") == cause), None)

    if not cause or not cause_detail:
        return f"It sounds like you're feeling {emotion}, but it's not yet clear what’s fueling that."

    anchor = cause_detail.get("anchor_text", "")
    note = cause_detail.get("notes", "")

    response = f"This {emotion} may be emerging from a deeper tension — something like {cause}. "
    if anchor:
        response += f"You mentioned: '{anchor}'. "
    if note:
        response += f"That seems to suggest: {note}."

    return response





def trace_emotion_sources(structures: List[Dict], user_input: str = "") -> Dict[str, Dict]:
    emotion_map = {}

    # Structure-based sources
    for s in structures:
        s_type = s.get("type")
        snippet = s.get("source_snippet", "")

        if s_type == "existential_drift":
            emotion_map["disillusionment"] = {
                "structure_type": s_type,
                "source_snippet": snippet
            }

        elif s_type in ["contradiction", "value_conflict"]:
            emotion_map["confusion"] = {
                "structure_type": s_type,
                "source_snippet": snippet
            }
            emotion_map.setdefault("frustration", {
                "structure_type": s_type,
                "source_snippet": snippet
            })

        elif s_type == "conflation":
            emotion_map["frustration"] = {
                "structure_type": s_type,
                "source_snippet": snippet
            }

        if "modal" in s.get("signals", []):
            emotion_map["uncertainty"] = {
                "structure_type": s_type,
                "source_snippet": snippet
            }

    # Fill in missing sources using semantic tags
    if user_input:
        semantic = estimate_emotional_state_semantic(user_input)
        for emo in semantic:
            if emo not in emotion_map:
                emotion_map[emo] = {
                    "structure_type": "semantic_inference",
                    "notes": f"Detected via semantic analysis of user input."
                }

    return emotion_map




def weigh_structures(parsed_json):
    structures = parsed_json.get("structures", [])
    causal_links = parsed_json.get("causal_links", [])

    weights = []

    for s in structures:
        structure_type = s.get("type")
        confidence = s.get("confidence", "implicit")
        resistance = s.get("resistance_level", "moderate")
        anchor = s.get("anchor_text", "")
        importance = 0

        # Base weight by structure type
        if structure_type == "existential drift":
            importance += 5
        elif structure_type == "contradiction":
            importance += 4
        elif structure_type == "false_frame":
            importance += 3
        elif structure_type == "conflation":
            importance += 2

        # Boost by confidence
        if confidence == "explicit":
            importance += 2

        # Adjust by resistance
        if resistance == "low":
            importance += 1
        elif resistance == "high":
            importance -= 1

        # Boost if it's part of a causal chain
        for link in causal_links:
            if structure_type == link.get("from"):
                importance += 2
            if structure_type == link.get("to"):
                importance += 1

        weights.append({
            "structure": s,
            "type": structure_type,
            "importance": importance
        })

    # Sort by descending importance
    weights.sort(key=lambda x: x["importance"], reverse=True)

    top_structure = weights[0] if weights else None
    others = weights[1:] if len(weights) > 1 else []

    return top_structure, others






@app.route("/history/<username>")
def get_user_history(username):
    user_data = load_user_history(username)
    return jsonify(user_data.get("dialogue", []))






@app.route("/thoughtmap/<username>")
def get_thought_map(username):
    filepath = f"thought_maps/{username}.json"
    if not os.path.exists(filepath):
        return jsonify({})
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        print(f"⚠️ Failed to load thought map for {username}: {e}")
        return jsonify({})






if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5009)
