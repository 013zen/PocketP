import json
import re
import spacy
import unicodedata
import uuid
import os
from typing import List, Dict
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from datetime import datetime

# Load models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

# === THEMATIC CONSTANTS ===
VALUE_THEMES = {
    "freedom": ["independence", "autonomy", "self-reliance"],
    "connection": ["belonging", "relationship", "support"],
    "control": ["stability", "order", "predictability"],
    "growth": ["risk", "exploration", "change"],
    "certainty": ["clarity", "truth", "security"],
    "openness": ["curiosity", "wonder", "fluidity"]
}

EMOTION_PROTOTYPES = {
    "frustration": ["Nothing I do seems to work.", "I'm sick of trying."],
    "confusion": ["I don't understand what I'm doing.", "Everything feels unclear."],
    "sadness": ["I feel empty inside.", "I'm deeply sad."],
    "anxiety": ["I'm scared of what's coming.", "Everything feels out of control."],
    "uncertainty": ["I don't know what to do with my life."],
    "disillusionment": ["Everything feels pointless.", "Hard work doesn’t pay off."]
}

# === STRUCTURE PRIORITY ORDER ===
def get_dominant_type(structures):
    priority = {
        "contradiction": 1,
        "existential_drift": 2,
        "value_conflict": 3,
        "conflation": 4,
        "reinforcement": 5,
        "value_assertion": 6
    }
    if not structures:
        return "none"
    sorted_structures = sorted(structures, key=lambda s: priority.get(s["type"], 99))
    return sorted_structures[0]["type"]

# === UTILITY FUNCTIONS ===
def normalize(text):
    return re.sub(r"[’‘`]", "'", unicodedata.normalize("NFKD", text.lower()))

def extract_snippet(text: str, start: int, end: int, window: int = 60) -> str:
    snippet_start = max(0, start - window)
    snippet_end = min(len(text), end + window)
    return text[snippet_start:snippet_end].strip()

def create_structure(type_: str, summary: str, confidence: float, signals: List[str], **kwargs) -> Dict:
    data = {
        "type": type_,
        "summary": summary,
        "confidence": round(confidence, 2),
        "signals": signals,
        "id": str(uuid.uuid4())
    }
    data.update(kwargs)
    return data

# === SEMANTIC EMOTION ESTIMATION ===
def estimate_emotional_state_semantic(user_input: str, top_n: int = 6, min_score: float = 0.3) -> Dict[str, float]:
    input_emb = model.encode(user_input, convert_to_tensor=True)
    scores = {}

    for emotion, examples in EMOTION_PROTOTYPES.items():
        example_embs = model.encode(examples, convert_to_tensor=True)
        similarity = util.cos_sim(input_emb, example_embs).max().item()
        if similarity > min_score:
            scores[emotion] = similarity

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n])


# === EMOTION CATEGORY ESTIMATION ===
def estimate_emotional_state(structures: List[Dict], user_input: str = "") -> List[str]:
    state = set()

    for s in structures:
        if s["type"] == "existential_drift":
            state.add("disillusionment")
        elif s["type"] in ["contradiction", "value_conflict"]:
            state.add("confusion")
        elif s["type"] == "conflation":
            state.add("frustration")
        elif s["type"] == "reinforcement":
            state.add("pride")
        elif s["type"] == "value_assertion":
            state.add("conviction")
        if "modal" in s.get("signals", []):
            state.add("uncertainty")

    if user_input:
        state.update(estimate_emotional_state_semantic(user_input))

    return list(state)

# === STRUCTURE DETECTION ENTRY ===
def detect_all_structures(text: str, prior_unresolved: List[Dict] = None) -> Dict:
    # Step 1: Get value assertions up front
    value_assertions = get_value_assertions(text)

    # Step 2: Estimate emotion first so contradiction threshold can adjust
    emotion_state = estimate_emotional_state([], text)


    # Step 3: Detect all structures
    contradictions = get_contradictions(text, prior_unresolved=prior_unresolved, emotion_scores=emotion_state)
    structures = (
        get_value_conflicts(text) +
        contradictions +
        get_conflations(text) +
        get_existential_drift(text) +
        value_assertions +
        get_reinforcements(text, value_assertions=value_assertions)
    )

    # Step 4: Link reinforcements
    structures = link_reinforcements_to_values(structures)

    # Step 5: Analyze structure relationships
    causal_links = infer_causal_links(structures)
    emotion_sources = trace_emotion_sources(structures, text)
    existential_drift = next((s for s in structures if s["type"] == "existential_drift"), None)

    return {
        "dominant_type": get_dominant_type(structures),
        "structures": structures,
        "causal_links": causal_links,
        "emotional_state": emotion_state,
        "emotion_sources": emotion_sources,
        "existential_drift": existential_drift
    }





def get_value_conflicts(text: str) -> List[Dict]:
    detected_conflicts = []
    doc = nlp(text)

    # Step 1: Break input into clauses (via dependency parsing + custom rules)
    clauses = []
    for sent in doc.sents:
        chunks = re.split(r"\b(but|however|though|yet|although)\b", sent.text, flags=re.IGNORECASE)
        # Merge chunks into pairs (before, after) if they imply contrast
        for i in range(0, len(chunks) - 1, 2):
            left = chunks[i].strip()
            right = chunks[i+2].strip() if i+2 < len(chunks) else ""
            if left and right:
                clauses.append((left, right))

    # If no contrastive split found, fallback to treating whole input as one clause
    # If no contrastive split found, fallback to treating whole input as one clause
    if not clauses:
        return []

    # NEW FILTER: Remove clause pairs that are trivially short or empty
    valid_clauses = [
        (left, right) for (left, right) in clauses
        if len(left.split()) > 2 and len(right.split()) > 2
    ]
    if not valid_clauses:
        return []

    # Step 2: Compare each clause to each value group
    for left_text, right_text in valid_clauses:
    

        left_emb = model.encode(left_text, convert_to_tensor=True)
        right_emb = model.encode(right_text, convert_to_tensor=True)

        for k1, v1 in VALUE_THEMES.items():
            emb1 = model.encode(v1, convert_to_tensor=True)
            sim1_left = util.cos_sim(left_emb, emb1).max().item()
            sim1_right = util.cos_sim(right_emb, emb1).max().item()

            for k2, v2 in VALUE_THEMES.items():
                if k1 >= k2:
                    continue
                emb2 = model.encode(v2, convert_to_tensor=True)
                sim2_left = util.cos_sim(left_emb, emb2).max().item()
                sim2_right = util.cos_sim(right_emb, emb2).max().item()

                # We're looking for oppositional values in opposing clauses
                left_hits = [(k1, sim1_left), (k2, sim2_left)]
                right_hits = [(k1, sim1_right), (k2, sim2_right)]

                left_top = max(left_hits, key=lambda x: x[1])
                right_top = max(right_hits, key=lambda x: x[1])

                # Conflict if each clause aligns with a different value domain strongly
                if left_top[0] != right_top[0] and left_top[1] > 0.35 and right_top[1] > 0.35:
                    conflict = create_structure(
                        "value_conflict",
                        f"{left_top[0]} vs {right_top[0]}",
                        confidence=round((left_top[1] + right_top[1]) / 2, 2),
                        signals=["semantic_overlap", "contrastive_clause"],
                        source_snippet=f"{left_text} / {right_text}"
                    )
                    detected_conflicts.append(conflict)

    return detected_conflicts





def get_contradictions(text: str, prior_unresolved: List[Dict] = None, emotion_scores: Dict[str, float] = None) -> List[Dict]:
    doc = nlp(text)
    lowered = text.lower()

    # 🧼 Ignore trivial uncertainty if very short
    short_uncertainties = [
        "i don't know", "i dont know", "i'm not sure", "im not sure",
        "i can't tell", "i cant tell", "not sure how", "don’t see why"
    ]
    if any(phrase in lowered for phrase in short_uncertainties) and len(text.split()) < 12:
        return []

    # 🧠 Try to find clauses separated by contrastive or conditional terms
    clause_delimiters = r"\b(but|yet|however|although|even though|though|if|because|when)\b"
    parts = re.split(clause_delimiters, text, flags=re.IGNORECASE)

    clause_pairs = []
    if len(parts) >= 3:
        for i in range(0, len(parts) - 2, 2):
            left = parts[i].strip()
            right = parts[i + 2].strip()
            if left and right:
                clause_pairs.append((left, right))

    if not clause_pairs:
        return []

    contradictions = []
    for left, right in clause_pairs:
        emb1 = model.encode(left, convert_to_tensor=True)
        emb2 = model.encode(right, convert_to_tensor=True)
        semantic_opposition = 1 - util.cos_sim(emb1, emb2).item()

        # Lower threshold if emotional uncertainty is high
        threshold = 0.35
        if emotion_scores and emotion_scores.get("uncertainty", 0) > 0.25:
            threshold = 0.3

        if semantic_opposition > threshold:
            clause_signals = ["semantic_opposition"]
            if any(token.dep_ == "neg" for token in doc):
                clause_signals.append("negation")
            if any(token.tag_ == "MD" for token in doc):
                clause_signals.append("modal")
            if re.search(r"\b(but|yet|however|although|even though|though)\b", lowered):
                clause_signals.append("contrastive_conjunction")
            if len(clause_signals) == 1:
                clause_signals.append("implied_opposition")

            # Subtype inference
            subtype = None
            if "want" in lowered and ("can't" in lowered or "not" in lowered):
                subtype = "desire_vs_limit"
            elif "believe" in lowered and re.search(r"\b(but|yet)\b", lowered):
                subtype = "belief_vs_experience"
            elif "should" in lowered and "not" in lowered:
                subtype = "certainty_vs_doubt"

            confidence = min(0.65 + 0.05 * len(clause_signals), 0.95)
            contradiction = create_structure(
                "contradiction",
                "internal tension or opposing claims",
                confidence=confidence,
                signals=clause_signals,
                source_snippet={"left": left, "right": right},
                subtype=subtype or "unspecified"
            )

            # Link to prior unresolved structure if similar
            if prior_unresolved:
                for past in prior_unresolved:
                    past_snippet = str(past.get("source_snippet", ""))
                    sim = util.cos_sim(
                        model.encode(text, convert_to_tensor=True),
                        model.encode(past_snippet, convert_to_tensor=True)
                    ).item()
                    if sim > 0.75:
                        contradiction["linked_to"] = past.get("id")
                        break

            contradictions.append(contradiction)

    return contradictions







def get_conflations(text: str, username: str = "") -> List[Dict]:
    patterns = [
        r"\b([\w\s\'\"-]+?)\s+(is|equals|=)\s+([\w\s\'\"-]+?)\b",
        r"\b([\w\s\'\"-]+?)\s+(and|or)\s+([\w\s\'\"-]+?)\s+(are|is)\s+(the same|identical|equivalent|interchangeable|same thing)\b",
        r"\bto\s+([\w\s\'\"-]+?)\s+is\s+to\s+([\w\s\'\"-]+?)\b",
        r"\b([\w\s\'\"-]+?)\s+means\s+([\w\s\'\"-]+?)\b",
        r"\b([\w\s\'\"-]+?)\s+is\s+just\s+([\w\s\'\"-]+?)\b",
        r"\b([\w\s\'\"-]+?)\s+is\s+nothing\s+but\s+([\w\s\'\"-]+?)\b"
    ]

    candidates = []

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            raw_snippet = match.group(0).strip()
            all_groups = [g.strip().strip('\'"') for g in match.groups()]
            content_terms = [t.lower() for t in all_groups if t.lower() not in {
                "is", "are", "to", "and", "or", "means", "equals", "=", 
                "the same", "identical", "equivalent", "interchangeable",
                "just", "nothing", "but"
            }]

            if len(content_terms) < 2:
                continue

            term1, term2 = content_terms[:2]
            if term1 == term2 or len(term1) < 3 or len(term2) < 3:
                continue

            # Semantic check
            embs = model.encode([term1, term2], convert_to_tensor=True)
            similarity = util.cos_sim(embs[0], embs[1]).item()
            distance = 1 - similarity

            # Subtype classification
            subtype = "label_equivalence"
            if any(x in term1 for x in ["freedom", "happiness", "love", "God", "reality"]) or \
               any(x in term2 for x in ["freedom", "happiness", "love", "God", "reality"]):
                subtype = "category_error"
            elif any(x in term1 for x in ["money", "job", "career", "school"]) and \
                 any(x in term2 for x in ["freedom", "joy", "identity", "meaning"]):
                subtype = "means_vs_end"

            if distance > 0.3:
                structure = create_structure(
                    "conflation",
                    f"{term1} conflated with {term2}",
                    confidence=round(distance, 2),
                    signals=["pattern_match", "semantic_distance"],
                    source_snippet=raw_snippet,
                    subtype=subtype,
                    terms=[term1, term2]
                )

                # Optional: belief conflict enhancement
                if username:
                    conflicting_beliefs = check_belief_conflict_with_conflation(term1, term2, username)
                    if conflicting_beliefs:
                        structure["belief_conflict"] = conflicting_beliefs

                candidates.append(structure)

    return candidates





def get_existential_drift(text: str) -> List[Dict]:
    import re
    from datetime import datetime
    import difflib
    from sentence_transformers import util

    lowered = normalize(text)
    now = datetime.now().isoformat()

    # Step 1: Filter trivial uncertainty
    trivial_uncertainties = [
        "i don't know how", "i'm not sure how",
        "i don't understand", "i don't know what to do",
        "i can't tell", "nothing specific", "not sure what to say"
    ]
    if any(phrase in lowered for phrase in trivial_uncertainties):
        return []

    # Step 2: Drift prototype clusters
    prototype_clusters = {
        "identity_confusion": [
            "I don't know who I am anymore.",
            "I’ve lost touch with myself.",
            "I used to know who I was."
        ],
        "meaninglessness": [
            "Nothing really matters.",
            "Everything I do feels pointless.",
            "What’s the point of anything?"
        ],
        "emotional_flatness": [
            "I feel empty inside.",
            "I feel numb.",
            "I go through the motions but don’t feel anything."
        ],
        "agency_loss": [
            "I’m stuck and don’t know what to do.",
            "I feel like I have no control.",
            "I'm trapped."
        ],
        "mortality_anxiety": [
            "I'm afraid of dying.",
            "I'm terrified of death.",
            "I can't stop thinking about dying.",
            "Death scares me.",
            "I'm consumed by thoughts of mortality."
        ]
    }

    WEIGHT_BY_SUBTYPE = {
        "identity_confusion": 0.95,
        "agency_loss": 0.85,
        "meaninglessness": 0.8,
        "emotional_flatness": 0.7,
        "mortality_anxiety": 0.82
    }

    MAP_TAGS_BY_SUBTYPE = {
        "identity_confusion": ["identity"],
        "agency_loss": ["agency"],
        "meaninglessness": ["values", "identity"],
        "emotional_flatness": ["agency", "fulfillment"],
        "mortality_anxiety": ["identity", "values"]
    }

    input_emb = model.encode(text, convert_to_tensor=True)

    best_match = None
    best_score = 0
    best_subtype = None

    for subtype, examples in prototype_clusters.items():
        proto_embs = model.encode(examples, convert_to_tensor=True)
        similarity = util.cos_sim(input_emb, proto_embs).max().item()
        if similarity > best_score:
            best_score = similarity
            best_match = examples[proto_embs.argmax().item()]
            best_subtype = subtype

    # Optional keyword fallback boost for mortality phrases
    if best_score < 0.68 and re.search(r"\\b(death|dying|mortality|afraid of dying|terrified of death|fear of dying)\\b", lowered):
        best_score = 0.7
        best_subtype = "mortality_anxiety"
        best_match = "I'm afraid of dying."

    if best_score > 0.68:
        return [create_structure(
            "existential_drift",
            "loss of meaning, direction, or identity",
            confidence=round(best_score, 2),
            signals=["semantic_similarity"],
            source_snippet=text,
            matched_prototype=best_match,
            subtype=best_subtype,
            weight=WEIGHT_BY_SUBTYPE.get(best_subtype, 0.7),
            map_tags=MAP_TAGS_BY_SUBTYPE.get(best_subtype, [])
        )]

    return []




def infer_causal_links(structures: List[Dict]) -> List[Dict]:
    links = []
    unresolved = [s for s in structures if not s.get("resolved", False)]
    by_type = {}

    for s in unresolved:
        by_type.setdefault(s["type"], []).append(s)

    def add_link(from_s, to_s, explanation):
        links.append({
            "link_id": str(uuid.uuid4()),
            "from": from_s.get("id"),
            "to": to_s.get("id") if to_s else None,
            "from_type": from_s["type"],
            "to_type": to_s["type"] if to_s else None,
            "explanation": explanation,
            "snippet": from_s.get("source_snippet", ""),
            "timestamp": datetime.now().isoformat()
        })

    # Contradiction → Drift
    for c in by_type.get("contradiction", []):
        for d in by_type.get("existential_drift", []):
            add_link(c, d, "Unresolved contradictions may lead to feelings of drift or confusion.")

    # False frame → Drift
    for f in by_type.get("false_frame", []):
        for d in by_type.get("existential_drift", []):
            add_link(f, d, "False framing of effort or meaning can increase existential confusion.")

    # Value conflict → Contradiction
    for v in by_type.get("value_conflict", []):
        for c in by_type.get("contradiction", []):
            add_link(v, c, "Conflicting values can generate internal contradictions.")

    # Conflation → Confusion (implicit)
    for conflation in by_type.get("conflation", []):
        add_link(conflation, None, "Equating distinct ideas can lead to misunderstanding and mental friction.")

    # Value conflict → Existential drift
    for v in by_type.get("value_conflict", []):
        for d in by_type.get("existential_drift", []):
            add_link(v, d, "A deep conflict between values may unsettle one's sense of meaning.")

    return links






def estimate_emotional_state(structures: List[Dict], user_input: str = "", top_n: int = 3) -> Dict[str, float]:
    """Returns emotion-to-confidence mapping (merged structural and semantic scores)."""
    emotion_scores = {}

    # Structure-based inferences: add fixed structural boosts
    structure_weights = {
        "existential_drift": {"disillusionment": 0.3},
        "contradiction": {"confusion": 0.2, "frustration": 0.1},
        "value_conflict": {"confusion": 0.2},
        "conflation": {"frustration": 0.2},
        "reinforcement": {"pride": 0.25},
        "value_assertion": {"conviction": 0.25}
    }

    for s in structures:
        s_type = s.get("type")
        if s_type in structure_weights:
            for emotion, weight in structure_weights[s_type].items():
                emotion_scores[emotion] = emotion_scores.get(emotion, 0) + weight

        if "modal" in s.get("signals", []):
            emotion_scores["uncertainty"] = emotion_scores.get("uncertainty", 0) + 0.15

    # Semantic prototype similarity (normalized)
    if user_input:
        semantic_scores = estimate_emotional_state_semantic(user_input)
        for emotion, score in semantic_scores.items():
            emotion_scores[emotion] = emotion_scores.get(emotion, 0) + round(score * 0.7, 2)  # Scale if needed

    # Normalize scores between 0 and 1
    max_score = max(emotion_scores.values(), default=1)
    normalized = {k: round(v / max_score, 3) for k, v in emotion_scores.items()}

    # Optionally: top N emotions only
    sorted_emotions = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_emotions[:top_n])




def estimate_emotional_state_semantic(user_input: str, top_n: int = 6) -> Dict[str, float]:
    user_input = user_input.strip().lower()

    # 🚫 Trivial greetings or very short inputs should return no emotion
    if len(user_input.split()) < 3 or user_input in {"hi", "hello", "hey", "good morning", "good evening"}:
        return {}

    input_emb = model.encode(user_input, convert_to_tensor=True)
    scores = {}

    for emotion, examples in EMOTION_PROTOTYPES.items():
        example_embs = model.encode(examples, convert_to_tensor=True)
        similarity = util.cos_sim(input_emb, example_embs).max().item()
        scores[emotion] = similarity

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n])






def get_value_assertions(text: str) -> List[Dict]:
    value_patterns = [
        # Imperatives: rules or duties
        (r"\b(should|must|need to|have to|ought to|it's important to)\b\s+(.*)", "imperative", 0.75),
        # Judgments: moral or evaluative statements
        (r"\b(is|are)\b.*?\b(good|bad|right|wrong|important|essential|sacred|meaningful|valuable)\b", "judgment", 0.7),
        # Personal beliefs
        (r"\b(I believe|I think|I feel like)\b\s+(.*)", "belief_statement", 0.65),
    ]

    matches = []
    seen_snippets = set()

    for pattern, subtype, base_conf in value_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            snippet = extract_snippet(text, match.start(), match.end())

            if snippet.lower() in seen_snippets:
                continue
            seen_snippets.add(snippet.lower())

            phrase = match.group(0)
            value_term = extract_value_term(phrase)
            theme = map_to_value_theme(value_term) if value_term else None

            structure = create_structure(
                "value_assertion",
                summary="expression of a core belief or value judgment",
                confidence=base_conf,
                signals=["value_keyword"],
                source_snippet=snippet,
                subtype=subtype,
                matched_phrase=phrase,
                value=value_term,
                theme=theme
            )
            matches.append(structure)

    return matches




def extract_value_term(phrase: str) -> str:
    doc = nlp(phrase)
    candidates = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text) > 3]
    return candidates[0] if candidates else None

def map_to_value_theme(term: str) -> str:
    for theme, keywords in VALUE_THEMES.items():
        if any(k in term for k in keywords):
            return theme
    return None





def get_reinforcements(text: str, value_assertions: List[Dict] = []) -> List[Dict]:
    reinforcement_phrases = [
        r"\b(that's (why|exactly why)|I still believe|I always felt|I stand by|I keep saying)\b",
        r"\b(no doubt|without question|definitely|clearly|obviously|certainly)\b"
    ]

    matches = []
    seen_snippets = set()

    for pattern in reinforcement_phrases:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            snippet = extract_snippet(text, match.start(), match.end())

            if snippet.lower() in seen_snippets:
                continue
            seen_snippets.add(snippet.lower())

            phrase = match.group(0).lower()

            # Determine subtype of reinforcement
            if any(x in phrase for x in ["still believe", "always felt", "stand by", "keep saying"]):
                subtype = "persistent"
            elif any(x in phrase for x in ["no doubt", "without question", "definitely", "clearly", "obviously", "certainly"]):
                subtype = "certainty"
            else:
                subtype = "emphatic"

            structure = create_structure(
                "reinforcement",
                "reassertion or emphasis of a belief or position",
                confidence=0.7,
                signals=["affirmation_keyword"],
                source_snippet=snippet,
                subtype=subtype,
                matched_phrase=phrase
            )

            # Try linking to a value_assertion using fuzzy match
            best_match = None
            highest_ratio = 0.0

            for va in value_assertions:
                ratio = SequenceMatcher(None, snippet.lower(), va["source_snippet"].lower()).ratio()
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_match = va

            if best_match and highest_ratio > 0.5:
                structure["reinforces"] = best_match["source_snippet"]
                structure["link_strength"] = round(highest_ratio, 2)

            matches.append(structure)

    return matches




def get_existential_drift(text: str) -> List[Dict]:
    lowered = normalize(text)

    # Step 1: Filter trivial uncertainty
    trivial_uncertainties = [
        "i don't know how", "i'm not sure how",
        "i don't understand", "i don't know what to do",
        "i can't tell", "nothing specific", "not sure what to say"
    ]
    if any(phrase in lowered for phrase in trivial_uncertainties):
        return []

    # Step 2: Drift prototype clusters
    prototype_clusters = {
        "identity_confusion": [
            "I don't know who I am anymore.",
            "I’ve lost touch with myself.",
            "I used to know who I was."
        ],
        "meaninglessness": [
            "Nothing really matters.",
            "Everything I do feels pointless.",
            "What’s the point of anything?"
        ],
        "emotional_flatness": [
            "I feel empty inside.",
            "I feel numb.",
            "I go through the motions but don’t feel anything."
        ],
        "agency_loss": [
            "I’m stuck and don’t know what to do.",
            "I feel like I have no control.",
            "I'm trapped."
        ],

        "mortality_anxiety": [
            "I'm afraid of dying.",
            "The thought of death terrifies me.",
            "I can't stop thinking about the fact that I'm going to die.",
            "I lie awake thinking about death.",
            "I'm overwhelmed by the idea of ceasing to exist."
        ]





    }

    WEIGHT_BY_SUBTYPE = {
        "identity_confusion": 0.95,
        "agency_loss": 0.85,
        "meaninglessness": 0.8,
        "emotional_flatness": 0.7,
        "mortality_anxiety": 0.88

    }

    MAP_TAGS_BY_SUBTYPE = {
        "identity_confusion": ["identity"],
        "agency_loss": ["agency"],
        "meaninglessness": ["values", "identity"],
        "emotional_flatness": ["agency", "fulfillment"],
        "mortality_anxiety": ["identity", "fulfillment"]

    }

    input_emb = model.encode(text, convert_to_tensor=True)

    best_match = None
    best_score = 0
    best_subtype = None

    for subtype, examples in prototype_clusters.items():
        if not examples:
            continue

        proto_embs = model.encode(examples, convert_to_tensor=True)
        similarities = util.cos_sim(input_emb, proto_embs)[0]

        if similarities.shape[0] == 0:
            continue

        similarity = similarities.max().item()
        if similarity > best_score:
            best_score = similarity
            best_match = examples[similarities.argmax().item()]
            best_subtype = subtype


    # Step 3: Structure creation
    if best_score > 0.68:
        return [create_structure(
            "existential_drift",
            "loss of meaning, direction, or identity",
            confidence=round(best_score, 2),
            signals=["semantic_similarity"],
            source_snippet=text,
            matched_prototype=best_match,
            subtype=best_subtype,
            weight=WEIGHT_BY_SUBTYPE.get(best_subtype, 0.7),
            map_tags=MAP_TAGS_BY_SUBTYPE.get(best_subtype, [])
        )]

    return []







def link_reinforcements_to_values(structures: List[Dict]) -> List[Dict]:
    reinforcements = [s for s in structures if s["type"] == "reinforcement"]
    value_assertions = [s for s in structures if s["type"] == "value_assertion"]

    for r in reinforcements:
        best_match = None
        highest_score = 0.0

        for v in value_assertions:
            # Fuzzy match on source snippets
            ratio = SequenceMatcher(None, r["source_snippet"].lower(), v["source_snippet"].lower()).ratio()
            if ratio > highest_score:
                highest_score = ratio
                best_match = v

        if best_match and highest_score > 0.5:
            r["reinforces"] = best_match["source_snippet"]
            r["link_strength"] = round(highest_score, 2)

    return structures



def trace_emotion_sources(structures: List[Dict], user_input: str = "") -> Dict[str, List[Dict]]:
    emotion_map = {}

    for s in structures:
        s_type = s.get("type")
        snippet = s.get("source_snippet", "")
        struct_id = s.get("id", "")
        subtype = s.get("subtype", "unspecified")

        def add_emotion(emotion_label):
            emotion_map.setdefault(emotion_label, []).append({
                "structure_type": s_type,
                "subtype": subtype,
                "source_snippet": snippet,
                "structure_id": struct_id,
                "matched_prototype": s.get("matched_prototype", None)
            })

        if s_type == "existential_drift":
            add_emotion("disillusionment")

        elif s_type in ["contradiction", "value_conflict"]:
            add_emotion("confusion")
            add_emotion("frustration")

        elif s_type == "conflation":
            add_emotion("frustration")

        elif s_type == "reinforcement":
            add_emotion("pride")

        elif s_type == "value_assertion":
            add_emotion("conviction")

        if "modal" in s.get("signals", []):
            add_emotion("uncertainty")

    return emotion_map
 
