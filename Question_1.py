"""
q1.py  –  Prompt Engineering: Extracting Per-Session Progress Scores

Pipeline overview
-----------------
    labeled_notes.json   ──► score ──► compute_metrics() ──► print results  (Q1a: validate prompt)
    unlabeled_notes.json ──► score ──► save                                  (Q1b: score at scale)

The LLM's job
-------------
For each client, the model receives the full sequence of session notes and
must return one progress score per consecutive note pair:

    notes 1→2 : score
    notes 2→3 : score
    ...
    notes 11→12 : score

Scores are integers 1–4, returned as a JSON list, e.g. [3, 2, 1, 2, ...].

What is already done for you
------------------------------
- Parsing and validating the LLM's JSON response
- Retrying once automatically if the response is malformed
- Looping over every client in a dataset
- Aligning true vs. predicted scores into a flat list of (true, predicted) pairs
- Building and printing the confusion matrix
- Saving all outputs to JSON

Your tasks  (search for # TODO to find each one)
--------------------------------------------------
1. build_prompt()      Write the prompt that instructs the LLM.
2. call_llm()          Wire up your chosen LLM API (OpenAI, Gemini, Anthropic, etc.).
3. compute_metrics()   Define and compute the performance metric(s) you will use
                       to evaluate and compare prompt versions.

Expected inputs:
    data/labeled_notes.json     – hand-scored by Patel; use this to test your prompt
    data/unlabeled_notes.json   – apply your validated prompt here

Expected outputs:
    output/evaluated_labeled_results.json   – scored test set with true labels (Q1a)
    output/scored_notes.json                – scored unlabeled clients (Q1b, feeds Q2)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import time

from tqdm import tqdm


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class BaseQ1Config:
    client_id_key: str = "client_id"
    notes_key: str = "notes"
    note_number_key: str = "note_number"
    note_text_key: str = "note_text"
    true_vector_key: str = "scored_progress"
    pred_vector_key: str = "estimated_trajectory_vector"

    valid_scores: tuple[int, ...] = (0, 1, 2, 3)


@dataclass
class Q1ALabeledConfig(BaseQ1Config):
    test_path: str = "labeled_notes.json"
    evaluated_output_path: str = "output/q1/evaluated_labeled_results.json"


@dataclass
class Q1BUnlabeledConfig(BaseQ1Config):
    unlabeled_path: str = "data/unlabeled_notes.json"
    output_path: str = "outputs/scored_notes.json"


# ============================================================================
# DATA LOADING / SAVING
# ============================================================================

def ensure_parent_dir(path: str | Path) -> Path:
    """Create parent folders for an output path and return it as a Path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load a top-level JSON list from disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON list in {path}.")
    return data


def save_json(data: Any, path: str) -> None:
    """Save JSON to disk and create parent folders if needed."""
    output_path = ensure_parent_dir(path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def get_vector_pair(
    record: Dict[str, Any],
    config: BaseQ1Config,
) -> tuple[str, List[int], List[int]]:
    """Pull the client id, true vector, and estimated vector from one scored record."""
    client_id = str(record[config.client_id_key])
    true_vector = record.get(config.true_vector_key, [])
    estimated_vector = record.get(config.pred_vector_key, [])
    return client_id, true_vector, estimated_vector


def build_step_comparisons(
    client_id: str,
    true_vector: List[int],
    estimated_vector: List[int],
) -> List[Dict[str, Any]]:
    """Build one row per compared step between the true and estimated vectors."""
    rows = []
    for step_idx, (true_score, estimated_score) in enumerate(
        zip(true_vector, estimated_vector),
        start=1,
    ):
        rows.append(
            {
                "client_id": client_id,
                "step_number": step_idx,
                "true_score": true_score,
                "estimated_score": estimated_score,
            }
        )
    return rows


def build_client_comparison(
    record: Dict[str, Any],
    config: BaseQ1Config,
) -> Dict[str, Any]:
    """Create the per-client comparison payload used by evaluation code."""
    client_id, true_vector, estimated_vector = get_vector_pair(record, config)
    step_rows = build_step_comparisons(client_id, true_vector, estimated_vector)
    return {
        "client_id": client_id,
        "true_vector": true_vector,
        "estimated_vector": estimated_vector,
        "n_true_scores": len(true_vector),
        "n_estimated_scores": len(estimated_vector),
        "n_compared_scores": len(step_rows),
        "step_comparisons": step_rows,
    }


def build_evaluation_comparisons(
    scored_test_data: List[Dict[str, Any]],
    config: BaseQ1Config,
) -> Dict[str, Any]:
    """Build client-level and step-level comparison tables for evaluation."""
    client_level_comparisons = []
    step_level_comparisons = []

    for record in scored_test_data:
        client_summary = build_client_comparison(record, config)
        client_level_comparisons.append(client_summary)
        step_level_comparisons.extend(client_summary["step_comparisons"])

    return {
        "n_clients": len(scored_test_data),
        "client_level_comparisons": client_level_comparisons,
        "step_level_comparisons": step_level_comparisons,
    }


def build_confusion_matrix(
    step_rows: List[Dict[str, Any]],
    valid_scores: List[int] | tuple[int, ...],
) -> Dict[str, Any]:
    """Build a confusion matrix with row totals, column totals, and a printable table."""
    matrix = {
        true_score: {estimated_score: 0 for estimated_score in valid_scores}
        for true_score in valid_scores
    }

    for row in step_rows:
        true_score = row["true_score"]
        estimated_score = row["estimated_score"]
        if true_score in matrix and estimated_score in matrix[true_score]:
            matrix[true_score][estimated_score] += 1

    row_totals = {
        true_score: sum(
            matrix[true_score][estimated_score] for estimated_score in valid_scores
        )
        for true_score in valid_scores
    }
    column_totals = {
        estimated_score: sum(
            matrix[true_score][estimated_score] for true_score in valid_scores
        )
        for estimated_score in valid_scores
    }
    grand_total = sum(row_totals.values())

    headers = ["true\\pred", *[str(score) for score in valid_scores], "Total"]
    row_label_width = max(
        len(headers[0]),
        len("Total"),
        max(len(str(score)) for score in valid_scores),
    )
    cell_width = max(
        5,
        max(
            len(str(value))
            for value in [
                *[
                    matrix[true_score][estimated_score]
                    for true_score in valid_scores
                    for estimated_score in valid_scores
                ],
                *row_totals.values(),
                *column_totals.values(),
                grand_total,
            ]
        ),
    )

    header_line = " | ".join(
        [headers[0].rjust(row_label_width)]
        + [header.rjust(cell_width) for header in headers[1:]]
    )
    separator_line = "-+-".join(
        ["-" * row_label_width] + ["-" * cell_width for _ in headers[1:]]
    )

    table_lines = [header_line, separator_line]
    for true_score in valid_scores:
        row_values = [
            str(matrix[true_score][estimated_score])
            for estimated_score in valid_scores
        ]
        row_line = " | ".join(
            [str(true_score).rjust(row_label_width)]
            + [value.rjust(cell_width) for value in row_values]
            + [str(row_totals[true_score]).rjust(cell_width)]
        )
        table_lines.append(row_line)

    total_line = " | ".join(
        ["Total".rjust(row_label_width)]
        + [
            str(column_totals[estimated_score]).rjust(cell_width)
            for estimated_score in valid_scores
        ]
        + [str(grand_total).rjust(cell_width)]
    )
    table_lines.append(separator_line)
    table_lines.append(total_line)

    return {
        "labels": list(valid_scores),
        "counts": matrix,
        "row_totals": row_totals,
        "column_totals": column_totals,
        "grand_total": grand_total,
        "table": "\n".join(table_lines),
    }


# ============================================================================
# TODO 1 of 3 — PROMPT
# ============================================================================

def build_prompt(notes_json_str: str) -> str:
    """
    Write the prompt that instructs the LLM to score a client's note sequence.

    The pipeline calls this once per client and passes the result directly to
    call_llm().  The notes arrive pre-serialised as a JSON string.

    What the LLM must do
    --------------------
    Read the notes in order and, for every consecutive pair (note N → note N+1),
    assign a progress score from 0 to 3.
    
    For a client with 12 notes, the model must return exactly 11 scores.

    Required output format
    ----------------------
    A JSON list and nothing else — no explanation, no markdown, no extra keys:
        [3, 2, 4, 1, 2, 3, 3, 2, 4, 4, 3]

    Parameters
    ----------
    notes_json_str : str
        The client's full note sequence serialised as a JSON string.
        Each note is a dict with keys "note_number" and "note_text".

    Returns
    -------
    str
        The complete prompt to send to the LLM.
    """
    # TODO ── your implementation here ──────────────────────────────────────
    return f'''You are David Patel, a senior Speech-Language Pathologist (SLP) with 15 years of experience in pediatric speech-language therapy. You specialize in evaluating children’s progress in articulation, language, and phonological development.

Clinical Background

In pediatric speech-language therapy, progress is evaluated along two dimensions.

Goal level (from simpler to more complex skills):
- Sound production (isolation)
- Syllable production
- Single word production
- Carrier phrase production
- Sentence production
- Spontaneous speech or conversation

Level of independence within a goal:
- By imitation (the clinician models first)
- With cueing (verbal, visual, or tactile prompts)
- Independently (no prompts needed)

Progress may be demonstrated by movement to a higher goal level, reduced cueing within the same level, or more consistent and generalized performance across activities or contexts.

Scoring Task

You will be given an ordered sequence of session notes for ONE client.

Your task is to evaluate RELATIVE progress BETWEEN SESSIONS.

IMPORTANT TEMPORAL RULES:
- Compare each session ONLY to the immediately preceding session.
- Do NOT use information from earlier sessions when assigning a score.
- Do NOT judge absolute skill level; judge change from Session N to Session N+1 only.
- Do NOT infer progress unless it is explicitly or clearly implied in the notes.

For each consecutive pair of session notes (Session N → Session N+1), assign exactly one score from 0 to 3.

Scoring Scale

Score 0 — Maintenance or Minimal Change  
The child is functioning at essentially the same level as before. Accuracy, cueing, independence, and goal level are similar, with no clear improvement.

Typical language includes:
“similar to last week,” “continued practice,” “ongoing work,” “same level of support.”

Important clarification — the following are STILL Score 0:
- Parent reports increased effort at home without in-session change
- Good engagement or participation without better accuracy or independence
- Continued work at the same level with similar support
- Vague or anecdotal “some improvement” that remains inconsistent

Score 1 — Small but Clear Improvement  
There is modest but observable progress within the same general level. At least one concrete improvement must be documented.

Examples include:
- Accuracy improves (e.g., ~40% to ~60%)
- Cueing decreases (e.g., maximal to moderate)
- Early generalization to a new task or context

Score 2 — Meaningful Clinical Progress  
There is a clear, clinically meaningful shift that would matter for treatment or discharge decisions.

Indicators include:
- Consistent performance replacing variability
- Clearly reduced support or cueing
- Skills generalizing across multiple activities, settings, or targets

Score 3 — Major Gain or Step Up in Level  
This score should be used sparingly. Assign only when a clear breakthrough is evident.

Examples include:
- Movement to a higher goal level (e.g., word → phrase, phrase → sentence)
- Independent production without prompting
- First clear evidence of spontaneous use

Boundary Guidance

Score 0 vs 1:
If no specific, observable improvement can be identified, assign Score 0.
If even one clear and concrete improvement is present, assign Score 1.

Score 1 vs 2:
Score 2 requires a noticeable clinical shift (consistency, independence, or generalization).
Incremental or partial gains remain Score 1.

Score 2 vs 3:
Score 3 is reserved for true level jumps or clear independence.
Strong improvement within the same level is Score 2.

Do NOT increase scores based solely on improved attention, motivation, or engagement unless this directly results in improved speech or language performance.

Worked Examples

Score 3 example:
Session 1: /k/ produced only in isolation and syllables with maximal cueing.
Session 2: /k/ produced at the word level with minimal cueing and early spontaneous use.
→ Clear jump in both level and independence → Score 3.

Score 2 example:
Session 2: Variable word-level performance with moderate support.
Session 3: Consistent accuracy, reduced cueing, and early phrase-level use.
→ Meaningful clinical progress → Score 2.

Score 1 example:
Session 3: Solid phrase-level performance.
Session 4: Early sentence attempts introduced, but accuracy drops.
→ Small, concrete improvement → Score 1.

Score 0 example:
Session 4 and Session 5: Similar sentence-level performance, accuracy, and cueing.
→ No observable change → Score 0.

Positive tone alone does NOT imply progress:
If notes emphasize effort, participation, or engagement without performance change, the correct score remains 0.

Critical Rule

Maintenance is common in speech-language therapy.
If evidence of progress is ambiguous or weak, default to Score 0.
Assign Scores 1–3 only when improvement is clearly observable in the notes.

Calibration Check

Across a realistic session sequence:
- Score 0 should be common
- Score 1 should appear when incremental gains are documented
- Score 2 should reflect meaningful clinical shifts
- Score 3 should be rare but must be assigned when clearly warranted

Do not inflate scores to satisfy calibration if evidence does not support it.

Your Task

You will be given session notes in JSON format.
Each note includes a note_number and note_text.

For every consecutive pair (note 1 → 2, note 2 → 3, etc.), assign one progress score.

SESSION NOTES:
{notes_json_str}

Return only a JSON list of integers with exactly one score per consecutive pair.
Do not include explanations or additional text.

Example:
[3, 1, 0, 0, 1, 2, 0, 0, 0, 1, 0]'''

# ============================================================================
# TODO 2 of 3 — LLM CALL
# ============================================================================

def call_llm(prompt: str) -> str:
    """
    Send a prompt to your chosen LLM and return the raw response text.

    Parameters
    ----------
    prompt : str
        The string returned by build_prompt().

    Returns
    -------
    str
        The model's raw text response (the pipeline will parse it).

    Instructions
    ------------
    Pick ONE of the three provider examples below, uncomment it, and add
    your API key.  Delete the other two and the raise at the bottom.

    Tips
    ----
    - Set temperature=0.0 so results are deterministic and reproducible.
    - Do not post-process the response here — return it raw.  Parsing and
      validation happen in parse_vector_from_response().
    """
    # ── Option A: OpenAI ────────────────────────────────────────────────────
    import os
    from openai import OpenAI
    
    client = OpenAI(api_key="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")
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()

    # ── Option B: Anthropic (Claude) ────────────────────────────────────────
    # import os
    # import anthropic
    #
    # client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # resp = client.messages.create(
    #     model="claude-sonnet-4-6",
    #     max_tokens=1024,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # return resp.content[0].text.strip()

    # ── Option C: Google Gemini ──────────────────────────────────────────────
    # import os
    # from google import genai
    
    # client = genai.Client(api_key="AIzaSyAyX35T577mFxGQ9KQe9oJ_mD_ZsSU5nmQ")
    # resp = client.models.generate_content(
    #     model="gemini-2.0-flash",
    #     contents=prompt,
    # )
    # return resp.text.strip()

    # # TODO ── uncomment one option above, then delete this line ─────────────
    # raise NotImplementedError("Implement call_llm() by uncommenting one option above")


# ============================================================================
# CLIENT-LEVEL SCORING
# ============================================================================

def parse_vector_from_response(
    response_text: str,
    expected_length: int,
    valid_scores: List[int] | tuple[int, ...] = (1, 2, 3, 4),
) -> List[int]:
    """
    Parse the model's response into one full trajectory vector.

    This function checks that:
    - the response is a JSON list
    - every item is an allowed score
    - the list length matches the number of note-to-note transitions

    Example valid response:
    [3, 2, 1]
    """
    try:
        data = json.loads(response_text)
        if not isinstance(data, list):
            raise ValueError("Model did not return a list")

        cleaned = []
        for value in data:
            score = int(value)
            if score not in valid_scores:
                raise ValueError(f"Invalid score {score}")
            cleaned.append(score)

        if len(cleaned) != expected_length:
            raise ValueError(
                f"Expected vector length {expected_length}, got {len(cleaned)}"
            )
        return cleaned
    except Exception:
        return []


def get_validated_vector_from_llm(prompt, expected_length, config, client_id):
    for attempt in (1, 2):
        raw_response = call_llm(prompt)
        estimated_vector = parse_vector_from_response(
            raw_response, expected_length, config.valid_scores
        )
        if estimated_vector:
            return estimated_vector
        print(f"Invalid response for {client_id}, attempt {attempt}: {repr(raw_response)}")

    # Instead of raising, return a fallback vector of all 1s
    print(f"WARNING: Using fallback vector for {client_id}")
    return [1] * expected_length  # or skip by returning []


def score_client_record(
    client_record: Dict[str, Any],
    config: BaseQ1Config,
) -> Dict[str, Any]:
    """
    Score one client's full note sequence.

    What this function does:
    - pulls all notes for one client
    - turns those notes into a JSON string for the prompt
    - calls the LLM once for the whole sequence
    - parses the returned vector of progress scores
    - returns one output record with the estimated vector

    If the input record already has a true scored vector, it is copied into the
    output too so the evaluation step can compare true vs estimated values.
    """
    all_notes = client_record[config.notes_key]
    client_id = str(client_record[config.client_id_key])
    notes_json_str = json.dumps(all_notes, ensure_ascii=False, indent=2)
    expected_length = max(len(all_notes) - 1, 0)

    prompt = build_prompt(notes_json_str)
    estimated_vector = get_validated_vector_from_llm(
        prompt=prompt,
        expected_length=expected_length,
        config=config,
        client_id=client_id,
    )

    scored_record = {
        config.client_id_key: client_record[config.client_id_key],
        config.notes_key: client_record[config.notes_key],
        config.pred_vector_key: estimated_vector,
    }
    if config.true_vector_key in client_record:
        scored_record[config.true_vector_key] = client_record[config.true_vector_key]
    return scored_record


def score_dataset(
    data: List[Dict[str, Any]],
    config: BaseQ1Config,
    progress_desc: str,
) -> List[Dict[str, Any]]:
    """Score every client record in a dataset and return the scored records."""
    scored = []

    for client_record in tqdm(data, desc=progress_desc):
        scored_record = score_client_record(client_record, config)
        scored.append(scored_record)

        # --- ADDED THIS LINE ---
        time.sleep(12)

    return scored


# ============================================================================
# EVALUATION SECTION
# ============================================================================

# ============================================================================
# TODO 3 of 3 — PERFORMANCE METRICS
# ============================================================================

def compute_metrics(step_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute one or more performance metrics from the step-level comparisons.

    The assignment asks you to choose and justify an evaluation approach that
    is appropriate for this task.  Implement your chosen metric(s) here.

    Parameters
    ----------
    step_rows : List[Dict[str, Any]]
        One dict per scored note transition across all clients.
        Each dict has at minimum:
            "true_score"      – Patel's hand-assigned score (int, 1–4)
            "estimated_score" – your LLM's predicted score  (int, 1–4)
        Example:
            [
              {"client_id": "C_0011", "step_number": 1,
               "true_score": 3, "estimated_score": 2},
              {"client_id": "C_0011", "step_number": 2,
               "true_score": 2, "estimated_score": 2},
              ...
            ]

    Returns
    -------
    Dict[str, Any]
        A dict mapping metric name → value.  Whatever you return here will
        be printed by print_evaluation().  Example shape:
            {"metricA": 0.61, "metricB": 0.88}

    """
    # TODO ── your implementation here ──────────────────────────────────────
    # Step 1: extract true_scores and pred_scores as lists from step_rows.
    # Filter out invalid scores (only keep 1-4)
    valid_scores_set = set((1, 2, 3, 4))
    filtered_rows = [
        row for row in step_rows
        if int(row["true_score"]) in valid_scores_set and int(row["estimated_score"]) in valid_scores_set
    ]
    
    true_scores = [int(row["true_score"]) for row in filtered_rows]
    pred_scores = [int(row["estimated_score"]) for row in filtered_rows]
    
    if not true_scores:
        return {
            "n_compared_scores": 0,
            "accuracy": 0.0,
            "adjacent_accuracy": 0.0,
            "mean_absolute_error": 0.0,
            "quadratic_weighted_kappa": 0.0,
        }

    #
    # Step 2: compute your chosen metric(s).
    n = len(true_scores)
    exact_matches = sum(
        1 for true_score, pred_score in zip(true_scores, pred_scores)
        if true_score == pred_score
    )
    adjacent_matches = sum(
        1 for true_score, pred_score in zip(true_scores, pred_scores)
        if abs(true_score - pred_score) <= 1
    )
    mean_absolute_error = sum(
        abs(true_score - pred_score)
        for true_score, pred_score in zip(true_scores, pred_scores)
    ) / n

    labels = [1, 2, 3, 4]
    observed_counts = {
        true_label: {pred_label: 0 for pred_label in labels}
        for true_label in labels
    }
    for true_score, pred_score in zip(true_scores, pred_scores):
        observed_counts[true_score][pred_score] += 1

    true_marginals = {
        label: sum(1 for score in true_scores if score == label)
        for label in labels
    }
    pred_marginals = {
        label: sum(1 for score in pred_scores if score == label)
        for label in labels
    }

    min_label = min(labels)
    max_label = max(labels)
    scale_width = max_label - min_label

    weighted_observed = 0.0
    weighted_expected = 0.0
    for true_label in labels:
        for pred_label in labels:
            weight = ((true_label - pred_label) / scale_width) ** 2
            observed = observed_counts[true_label][pred_label] / n
            expected = (
                true_marginals[true_label] * pred_marginals[pred_label]
            ) / (n * n)
            weighted_observed += weight * observed
            weighted_expected += weight * expected

    quadratic_weighted_kappa = (
        1.0 - (weighted_observed / weighted_expected)
        if weighted_expected > 0
        else 1.0
    )

    #
    # Step 3: return a dict, e.g.:
    #   return {
    #       "metricA": ...,
    #       "metricB": ...,
    #   }
    return {
        "n_compared_scores": n,
        "accuracy": round(exact_matches / n, 4),
        "adjacent_accuracy": round(adjacent_matches / n, 4),
        "mean_absolute_error": round(mean_absolute_error, 4),
        "quadratic_weighted_kappa": round(quadratic_weighted_kappa, 4),
    }


def evaluate_predictions(
    config: Q1ALabeledConfig,
) -> Dict[str, Any]:
    """
    Compare each client's true scored_vector with the predicted
    estimated_trajectory_vector, then compute metrics and the confusion matrix.
    """
    scored_test_data = load_json(config.evaluated_output_path)
    comparisons = build_evaluation_comparisons(scored_test_data, config)
    step_rows = comparisons["step_level_comparisons"]

    metrics = compute_metrics(step_rows)
    confusion_matrix = build_confusion_matrix(step_rows, config.valid_scores)

    return {
        **metrics,
        "confusion_matrix": confusion_matrix,
    }


def print_evaluation(results: Dict[str, Any]) -> None:
    print("\n=== Evaluation Results ===")
    for key, value in results.items():
        if key == "confusion_matrix" and isinstance(value, dict):
            print("confusion_matrix:")
            print(value.get("table", ""))
        else:
            print(f"{key}: {value}")


# ============================================================================
# PIPELINES
# ============================================================================

def run_test_pipeline(config: Q1ALabeledConfig) -> List[Dict[str, Any]]:
    """Run the Q1 pipeline on labeled test data."""
    import os
    if os.path.exists(config.evaluated_output_path):
        print(f"Loading existing scored data from {config.evaluated_output_path}")
        scored_test_data = load_json(config.evaluated_output_path)
    else:
        test_data = load_json(config.test_path)

        scored_test_data = score_dataset(
            data=test_data,
            config=config,
            progress_desc="Scoring labeled clients",
        )
        save_json(scored_test_data, config.evaluated_output_path)

    results = evaluate_predictions(config)
    print_evaluation(results)

    return scored_test_data


def run_unlabeled_pipeline(config: Q1BUnlabeledConfig) -> List[Dict[str, Any]]:
    """Run the Q1 pipeline on unlabeled note data and save scored outputs."""
    unlabeled_data = load_json(config.unlabeled_path)

    scored_unlabeled_data = score_dataset(
        data=unlabeled_data,
        config=config,
        progress_desc="Scoring unlabeled clients",
    )
    save_json(scored_unlabeled_data, config.output_path)

    return scored_unlabeled_data


# ============================================================================
# ENTRY POINT
# ============================================================================
#
# HOW TO WORK THROUGH THIS FILE
# ──────────────────────────────
# There are three functions marked # TODO that you must implement:
#
#   1. build_prompt()      Write the prompt that tells the LLM what to do.
#   2. call_llm()          Wire up your LLM API (uncomment one of the three
#                          provider options and add your API key).
#   3. compute_metrics()   Define the metric(s) you will use to evaluate and
#                          compare prompt versions.
#
# Recommended order:
#   Step 1 — implement build_prompt(), call_llm(), and compute_metrics()
#   Step 2 — run run_test_pipeline(LABELED_CONFIG) to score the labeled set
#             and see your metrics + confusion matrix printed to the terminal
#   Step 3 — iterate on your prompt; re-run Step 2 to compare versions
#   Step 4 — once satisfied, run run_unlabeled_pipeline(UNLABELED_CONFIG)
#             to score all 300 clients → produces scored_notes.json for Q2
#
# TIP: before running at scale, test your prompt on a single client record:
#
#   import json
#   sample = load_json("data/labeled_notes.json")[0]
#   notes_str = json.dumps(sample["notes"], indent=2)
#   print(build_prompt(notes_str))           # inspect the prompt visually
#   print(call_llm(build_prompt(notes_str))) # check the raw model response
# ============================================================================
import csv

def save_scored_notes_csv(
    scored_unlabeled_data: List[Dict[str, Any]],
    output_csv_path: str,
    client_id_key: str = "client_id",
    pred_vector_key: str = "estimated_trajectory_vector",
) -> None:
    """
    Save model-scored progress vectors into a flat CSV with columns:
        client_id, session, score
    """
    output_path = ensure_parent_dir(output_csv_path)

    rows = []
    for record in scored_unlabeled_data:
        client_id = record[client_id_key]
        scores = record.get(pred_vector_key, [])

        for idx, score in enumerate(scores, start=1):
            rows.append(
                {
                    "client_id": client_id,
                    "session": idx,
                    "score": score,
                }
            )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["client_id", "session", "score"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_path}")



if __name__ == "__main__":
    LABELED_CONFIG = Q1ALabeledConfig(
        test_path="labeled_notes.json",
        evaluated_output_path="output/q1/evaluated_labeled_results.json",
    )
    UNLABELED_CONFIG = Q1BUnlabeledConfig(
        unlabeled_path="unlabeled_notes.json",
        output_path="output/q1/scored_notes.json",
    )

    # Step 2: validate your prompt on the labeled test set
    run_test_pipeline(LABELED_CONFIG)

    # Step 4: score all unlabeled clients (only after prompt is validated)
    run_unlabeled_pipeline(UNLABELED_CONFIG)
    
    scored_unlabeled_data = load_json("output/q1/scored_notes.json")
    

    save_scored_notes_csv(
        scored_unlabeled_data=scored_unlabeled_data,
        output_csv_path="output/q1/scored_notes.csv",
        client_id_key=UNLABELED_CONFIG.client_id_key,
        pred_vector_key=UNLABELED_CONFIG.pred_vector_key,
    )
