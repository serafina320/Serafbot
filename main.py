"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SERAF v2  —  Statistical & Extremized Reasoning AI Forecaster              ║
║  Spring 2026 Metaculus AI Tournament                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Architecture                                                               ║
║  ──────────────────────────────────────────────────────────────────────     ║
║  default     openrouter/openai/gpt-5.4       primary deep reasoning         ║
║  challenger  openrouter/openai/gpt-5.2       independent second opinion     ║
║  arbiter     openrouter/claude-sonnet-4.6    devil's-advocate / tiebreak    ║
║  parser      openrouter/openai/gpt-5.2       structured-output extraction   ║
║  research    Exa neural search + Tinyfish    live dual-engine web pipeline  ║
║                                                                             ║
║  Statistical Principles Applied                                             ║
║  ──────────────────────────────────────────────────────────────────────     ║
║  1.  Base-rate / reference-class anchoring                                  ║
║  2.  Inside-view + outside-view synthesis                                   ║
║  3.  Three-model ensemble + variance-aware extremizing (Satopaa 2014)       ║
║  4.  Community-prediction Bayesian prior (update by private research)       ║
║  5.  Devil's-advocate sub-prompt for binary questions near 50%              ║
║  6.  Numeric sanity-check pass (physically reasonable spread?)              ║
║  7.  Category-weighted model voting (geopolitics/science/economics/tech)    ║
║  8.  Brier-score tracker with auto-tune of extremizing alpha                ║
║  9.  Status-quo conservatism + humble wide intervals                        ║
║  10. No silent fallbacks — missing env vars raise immediately               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Required .env keys
──────────────────
OPENROUTER_API_KEY   your OpenRouter key
EXA_API_KEY          your Exa neural-search key
TINYFISH_API_KEY     your Tinyfish key
METACULUS_TOKEN      your Metaculus API token
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import dotenv
import httpx

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Environment validation — raise immediately, no silent fallback
# ─────────────────────────────────────────────────────────────────────────────
_REQUIRED_ENV = ["OPENROUTER_API_KEY", "EXA_API_KEY", "TINYFISH_API_KEY"]
_missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
if _missing:
    raise EnvironmentError(
        f"[SERAF] Missing required environment variables: {_missing}\n"
        "Add them to your .env file before running."
    )

EXA_API_KEY      = os.environ["EXA_API_KEY"]
TINYFISH_API_KEY = os.environ["TINYFISH_API_KEY"]


# ─────────────────────────────────────────────────────────────────────────────
# Brier-score tracker — persisted to brier_history.json
# Records resolved predictions and auto-tunes extremizing alpha over time.
# ─────────────────────────────────────────────────────────────────────────────
BRIER_FILE = Path(__file__).parent / "brier_history.json"


def _load_brier_history() -> dict[str, Any]:
    if BRIER_FILE.exists():
        with open(BRIER_FILE) as f:
            return json.load(f)
    return {"alpha": 1.6, "scores": [], "n": 0, "mean_brier": None}


def _save_brier_history(history: dict[str, Any]) -> None:
    with open(BRIER_FILE, "w") as f:
        json.dump(history, f, indent=2)


def record_brier_score(prediction: float, outcome: float) -> float:
    """
    Call this after a question resolves to record the result and tune alpha.
    outcome: 1.0 = YES, 0.0 = NO.
    Returns the updated extremizing alpha.
    Usage: record_brier_score(0.72, 1.0)
    """
    history = _load_brier_history()
    score = (prediction - outcome) ** 2
    history["scores"].append(score)
    history["n"] += 1
    history["mean_brier"] = statistics.mean(history["scores"])

    mean_b = history["mean_brier"]
    # Well-calibrated (<0.15) → push alpha higher (more extremizing is warranted)
    # Over-confident (>0.25) → pull alpha back toward 1.0
    if mean_b < 0.10:
        history["alpha"] = min(2.5, history["alpha"] + 0.05)
    elif mean_b < 0.15:
        history["alpha"] = min(2.2, history["alpha"] + 0.02)
    elif mean_b > 0.25:
        history["alpha"] = max(1.1, history["alpha"] - 0.05)
    elif mean_b > 0.20:
        history["alpha"] = max(1.2, history["alpha"] - 0.02)

    _save_brier_history(history)
    logger.info(
        f"[BRIER] score={score:.4f} mean={mean_b:.4f} "
        f"n={history['n']} alpha={history['alpha']:.2f}"
    )
    return history["alpha"]


def _get_current_alpha() -> float:
    return _load_brier_history().get("alpha", 1.6)


# ─────────────────────────────────────────────────────────────────────────────
# Extremizing helpers (Satopaa et al. 2014)
# ─────────────────────────────────────────────────────────────────────────────

def _extremize_binary(p: float, alpha: float) -> float:
    """p' = p^alpha / (p^alpha + (1-p)^alpha). Clipped to [0.02, 0.98]."""
    if p <= 0.0:
        return 0.02
    if p >= 1.0:
        return 0.98
    num = p ** alpha
    den = num + (1.0 - p) ** alpha
    return max(0.02, min(0.98, num / den))


def _aggregate_binary_predictions(
    preds: list[float],
    weights: list[float] | None = None,
) -> float:
    """
    Weighted-mean ensemble with variance-aware extremizing.
    Low variance (agreement) → extremize with Brier-tuned alpha.
    High variance (disagreement) → stay conservative at weighted mean.
    """
    if not preds:
        raise ValueError("[SERAF] No predictions to aggregate.")
    if len(preds) == 1:
        return max(0.02, min(0.98, preds[0]))

    weights = weights or [1.0] * len(preds)
    total_w = sum(weights)
    w_mean  = sum(p * w for p, w in zip(preds, weights)) / total_w
    var     = statistics.variance(preds)

    HIGH_VAR = 0.025
    alpha    = _get_current_alpha()

    if var < HIGH_VAR:
        scaled = min(alpha * (1.0 + (HIGH_VAR - var) * 10), 2.8)
        result = _extremize_binary(w_mean, scaled)
        logger.info(
            f"[EXTREMIZE] var={var:.4f} w_mean={w_mean:.3f} "
            f"alpha={scaled:.2f} final={result:.3f}"
        )
    else:
        result = max(0.02, min(0.98, w_mean))
        logger.info(
            f"[CONSERVATIVE] var={var:.4f} w_mean={w_mean:.3f} final={result:.3f}"
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Category-weight table
# Weights[0]=gpt-5.4  Weights[1]=gpt-5.2  Weights[2]=claude-sonnet-4.6
# Update as you accumulate Brier scores per category from tournament results.
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_WEIGHTS: dict[str, list[float]] = {
    "geopolitics":  [1.0, 0.9, 1.1],
    "science":      [1.1, 1.0, 0.9],
    "economics":    [1.0, 1.1, 0.9],
    "technology":   [1.1, 1.0, 1.0],
    "environment":  [1.0, 0.9, 1.1],
    "sports":       [0.9, 1.0, 1.0],
    "default":      [1.0, 1.0, 1.0],
}


def _infer_category(question_text: str) -> str:
    t = question_text.lower()
    if any(k in t for k in ["war", "election", "president", "government", "nato",
                              "sanction", "treaty", "conflict", "military", "diplomat"]):
        return "geopolitics"
    if any(k in t for k in ["gdp", "inflation", "unemployment", "market", "interest rate",
                              "stock", "economic", "trade", "tariff", "recession", "fiscal"]):
        return "economics"
    if any(k in t for k in ["ai", "model", "gpu", "compute", "software", "tech",
                              "chip", "llm", "robot", "algorithm", "data center"]):
        return "technology"
    if any(k in t for k in ["climate", "carbon", "emission", "temperature",
                              "species", "environment", "wildfire", "drought"]):
        return "environment"
    if any(k in t for k in ["study", "vaccine", "drug", "gene", "protein",
                              "research", "experiment", "physics", "chemistry", "trial"]):
        return "science"
    if any(k in t for k in ["championship", "league", "tournament", "score",
                              "medal", "olympic", "fifa", "nba", "nfl"]):
        return "sports"
    return "default"


# ─────────────────────────────────────────────────────────────────────────────
# Exa neural search
# ─────────────────────────────────────────────────────────────────────────────

async def _exa_search(query: str, num_results: int = 8) -> str:
    url = "https://api.exa.ai/search"
    headers = {
        "x-api-key": EXA_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "numResults": num_results,
        "useAutoprompt": True,
        "contents": {"text": {"maxCharacters": 800}},
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        results = data.get("results", [])
        if not results:
            return "[Exa] No results found."
        parts = []
        for r in results:
            title   = r.get("title", "No title")
            snippet = (r.get("text") or "")[:600]
            source  = r.get("url", "")
            parts.append(f"**{title}**\n{snippet}\n({source})")
        return "\n\n".join(parts)
    except Exception as exc:
        logger.warning(f"[Exa] Failed for '{query}': {exc}")
        return f"[Exa] Unavailable: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tinyfish search
# ─────────────────────────────────────────────────────────────────────────────

async def _tinyfish_search(query: str, num_results: int = 6) -> str:
    url = "https://api.tinyfish.ai/search"
    headers = {
        "Authorization": f"Bearer {TINYFISH_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"query": query, "limit": num_results}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        results = data.get("results", data.get("items", []))
        if not results:
            return "[Tinyfish] No results found."
        parts = []
        for r in results:
            title   = r.get("title", r.get("name", "No title"))
            snippet = (r.get("snippet", r.get("content", r.get("text", ""))) or "")[:500]
            source  = r.get("url", r.get("link", ""))
            parts.append(f"**{title}**\n{snippet}\n({source})")
        return "\n\n".join(parts)
    except Exception as exc:
        logger.warning(f"[Tinyfish] Failed for '{query}': {exc}")
        return f"[Tinyfish] Unavailable: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Dual-engine research pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def _dual_search_research(
    question: MetaculusQuestion,
    synthesis_llm: GeneralLlm,
) -> str:
    """
    Runs Exa + Tinyfish in parallel across 3 query angles (base-rate,
    current-status, expert-opinion), then synthesises into a structured
    intelligence brief via the synthesis LLM (Claude Sonnet 4.6).
    """
    q_text = question.question_text
    queries = [
        f"{q_text} historical base rate statistics data",
        f"{q_text} latest news 2025 2026 current status update",
        f"{q_text} expert analysis forecast prediction consensus",
    ]

    # Fire all 6 searches (3 Exa + 3 Tinyfish) concurrently
    tasks = []
    for q in queries:
        tasks.append(_exa_search(q, num_results=6))
        tasks.append(_tinyfish_search(q, num_results=5))

    raw = await asyncio.gather(*tasks, return_exceptions=True)

    sections = []
    for i, q in enumerate(queries):
        exa_out  = raw[i * 2]   if isinstance(raw[i * 2],   str) else f"[Exa error: {raw[i*2]}]"
        fish_out = raw[i * 2+1] if isinstance(raw[i * 2+1], str) else f"[Tinyfish error: {raw[i*2+1]}]"
        sections.append(
            f"### Query: {q}\n"
            f"**Exa:**\n{exa_out}\n\n"
            f"**Tinyfish:**\n{fish_out}"
        )

    raw_combined = "\n\n---\n\n".join(sections)

    synthesis_prompt = clean_indents(
        f"""
        You are SERAF's research analyst. Synthesise the raw search results below
        into a structured intelligence brief for a professional superforecaster.

        FORECASTING QUESTION:
        {q_text}

        RESOLUTION CRITERIA:
        {question.resolution_criteria}

        FINE PRINT:
        {question.fine_print}

        RAW SEARCH RESULTS:
        {raw_combined}

        Output EXACTLY this structure:

        ## Base Rate / Historical Context
        (Historical frequency of this type of event; include numbers where available)

        ## Current Status ({datetime.now().strftime("%Y-%m-%d")})
        (What is known right now? Most recent developments?)

        ## Key Evidence FOR (Yes / Higher outcome)
        - bullet points, most important first

        ## Key Evidence AGAINST (Yes / Higher outcome)
        - bullet points, most important first

        ## Expert & Market Signals
        (Polls, prediction markets, institutional forecasts, expert quotes)

        ## Likely Resolution Direction
        (One paragraph summary. Do NOT give a probability — that is SERAF's job.)

        Be factual and flag conflicting evidence clearly. Cite sources inline.
        """
    )

    try:
        brief = await synthesis_llm.invoke(synthesis_prompt)
    except Exception as exc:
        logger.warning(f"[Research] Synthesis failed: {exc}. Returning raw results.")
        brief = raw_combined

    logger.info(f"[SERAF] Research complete for {question.page_url}")
    return brief


# ─────────────────────────────────────────────────────────────────────────────
# Community prior helper
# ─────────────────────────────────────────────────────────────────────────────

def _community_prior_text(question: MetaculusQuestion) -> str:
    try:
        cp = getattr(question, "community_prediction", None)
        if cp is None:
            return ""
        if isinstance(cp, float):
            pct = cp * 100
            return clean_indents(
                f"""
                COMMUNITY PRIOR: The Metaculus community currently forecasts {pct:.1f}%.
                Treat this as a Bayesian prior. The crowd is generally well-calibrated.
                Require strong, concrete evidence from your research to deviate more than
                10–15 percentage points from this prior in either direction.
                """
            )
        return (
            f"\nCOMMUNITY PRIOR: {cp}\n"
            "Treat this as a Bayesian prior. Update based on your research.\n"
        )
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Numeric sanity checker
# ─────────────────────────────────────────────────────────────────────────────

async def _sanity_check_numeric(
    question: NumericQuestion,
    percentile_list: list[Percentile],
    checker_llm: GeneralLlm,
) -> list[Percentile]:
    """
    Ask the checker LLM to verify the distribution is physically reasonable.
    Returns a corrected list if problems are found, otherwise the original.
    """
    try:
        p_str = "\n".join(f"P{p.percentile}: {p.value}" for p in percentile_list)
        check_prompt = clean_indents(
            f"""
            You are a calibration auditor reviewing a numeric forecast distribution.

            QUESTION: {question.question_text}
            UNITS: {question.unit_of_measure}
            LOWER BOUND: {question.lower_bound}
            UPPER BOUND: {question.upper_bound}

            PROPOSED DISTRIBUTION:
            {p_str}

            Verify ALL of the following:
            1. Values are strictly increasing: P10 < P20 < P40 < P60 < P80 < P90?
            2. Values lie within [{question.lower_bound}, {question.upper_bound}]
               (or are reasonable for open bounds)?
            3. Spread is physically plausible (not absurdly narrow or wide)?
            4. Units are consistent with the question?

            If all checks pass, reply EXACTLY: PASS

            If corrections are needed, reply:
            CORRECTED
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            """
        )
        result = (await checker_llm.invoke(check_prompt)).strip()

        if result.startswith("PASS"):
            logger.info("[SANITY] Numeric forecast PASSED.")
            return percentile_list

        if result.startswith("CORRECTED"):
            logger.info("[SANITY] Numeric forecast CORRECTED.")
            corrected: list[Percentile] = await structure_output(
                result, list[Percentile],
                model=checker_llm,
                num_validation_samples=1,
            )
            if corrected:
                return corrected

        return percentile_list

    except Exception as exc:
        logger.warning(f"[SANITY] Check failed ({exc}). Using original.")
        return percentile_list


# ─────────────────────────────────────────────────────────────────────────────
# SERAF v2 Bot
# ─────────────────────────────────────────────────────────────────────────────

class Seraf(ForecastBot):
    """
    SERAF v2 — Spring 2026 Metaculus AI Tournament.

    LLM slots
    ─────────
    "default"    → openrouter/openai/gpt-5.4          primary reasoning
    "challenger" → openrouter/openai/gpt-5.2          independent second pass
    "arbiter"    → openrouter/claude-sonnet-4.6       devil's advocate & synthesis
    "parser"     → openrouter/openai/gpt-5.2          structured output
    """

    _max_concurrent_questions = 1
    _concurrency_limiter       = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    # Binary questions within this probability range get a devil's-advocate pass
    DEVIL_ADVOCATE_ZONE = (0.35, 0.65)

    # ── System prompt (shared across all question types) ──────────────── #
    def _system_prompt(self) -> str:
        return clean_indents(
            """
            You are SERAF, a world-class superforecaster specialising in calibrated
            probabilistic reasoning.

            Core principles you always apply:
            1. BASE RATE FIRST — anchor on historical frequency before case specifics.
            2. INSIDE + OUTSIDE VIEW — specific details AND reference class both matter.
            3. STATUS QUO WEIGHT — absent a clear forcing function, nothing changes.
            4. CALIBRATION — overconfidence kills Brier scores. Never exceed 90% or
               go below 10% without overwhelming unambiguous evidence.
            5. COMMUNITY SIGNAL — Metaculus crowds are well-calibrated. Deviating from
               the community prior requires explicit, evidence-backed justification.
            6. HUMBLE INTERVALS — for numeric questions, err toward wider spreads.
               Unknown unknowns are real and frequent.
            7. SCOPE VERIFICATION — always check units, scale, and resolution criteria
               before committing to a final answer.
            """
        )

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            arbiter_llm = self.get_llm("arbiter", "llm")
            return await _dual_search_research(question, arbiter_llm)

    ##################################### BINARY #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """
        Three-model ensemble with devil's-advocate pass near 50%.
        Category-weighted aggregation with Brier-tuned extremizing.
        """
        community_prior = _community_prior_text(question)
        category        = _infer_category(question.question_text)
        cat_w           = list(CATEGORY_WEIGHTS.get(category, CATEGORY_WEIGHTS["default"]))

        primary_prompt = clean_indents(
            f"""
            {self._system_prompt()}

            QUESTION: {question.question_text}

            BACKGROUND:
            {question.background_info}

            RESOLUTION CRITERIA (not yet satisfied):
            {question.resolution_criteria}

            {question.fine_print}

            RESEARCH BRIEF:
            {research}

            {community_prior}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Work through each step:
            (a) Time remaining until resolution.
            (b) Historical base rate for this class of event (cite a number if known).
            (c) Community prior assessment — why might the crowd be right or wrong?
            (d) Status quo — what happens if the world continues unchanged?
            (e) Strongest argument for YES outcome.
            (f) Strongest argument for NO outcome.
            (g) Evidence quality — how reliable and complete is available data?
            (h) Bias check — availability heuristic? Recency bias? Narrative fallacy?

            {self._get_conditional_disclaimer_if_necessary(question)}

            Be humble. Only go above 85% or below 15% with overwhelming evidence.
            End with exactly: "Probability: ZZ%"
            """
        )

        challenger_prompt = clean_indents(
            f"""
            {self._system_prompt()}

            You are providing an INDEPENDENT second opinion. Deliberately seek out
            the case that the consensus view is wrong.

            QUESTION: {question.question_text}

            RESOLUTION CRITERIA:
            {question.resolution_criteria}

            RESEARCH BRIEF:
            {research}

            {community_prior}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            (a) Strongest case for the NON-OBVIOUS outcome.
            (b) Systematic errors the primary forecaster might be making.
            (c) Tail risks or black-swan scenarios deserving weight.
            (d) Your independent calibrated probability.

            {self._get_conditional_disclaimer_if_necessary(question)}

            End with exactly: "Probability: ZZ%"
            """
        )

        arbiter_prompt = clean_indents(
            f"""
            {self._system_prompt()}

            You are the ARBITER. Reason independently — do not anchor on any
            prior model's estimate.

            QUESTION: {question.question_text}

            RESOLUTION CRITERIA:
            {question.resolution_criteria}

            RESEARCH BRIEF:
            {research}

            {community_prior}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Focus on:
            - Asymmetric information the other models might overlook.
            - Resolution-criteria edge cases that change the answer.
            - Whether the time horizon shifts probabilities significantly.

            {self._get_conditional_disclaimer_if_necessary(question)}

            End with exactly: "Probability: ZZ%"
            """
        )

        # Run all three models concurrently
        primary_r, challenger_r, arbiter_r = await asyncio.gather(
            self.get_llm("default",    "llm").invoke(primary_prompt),
            self.get_llm("challenger", "llm").invoke(challenger_prompt),
            self.get_llm("arbiter",    "llm").invoke(arbiter_prompt),
        )

        # Parse all three concurrently
        bp_primary, bp_challenger, bp_arbiter = await asyncio.gather(
            structure_output(primary_r,    BinaryPrediction,
                             model=self.get_llm("parser", "llm"),
                             num_validation_samples=self._structure_output_validation_samples),
            structure_output(challenger_r, BinaryPrediction,
                             model=self.get_llm("parser", "llm"),
                             num_validation_samples=self._structure_output_validation_samples),
            structure_output(arbiter_r,    BinaryPrediction,
                             model=self.get_llm("parser", "llm"),
                             num_validation_samples=self._structure_output_validation_samples),
        )

        v1 = max(0.02, min(0.98, bp_primary.prediction_in_decimal))
        v2 = max(0.02, min(0.98, bp_challenger.prediction_in_decimal))
        v3 = max(0.02, min(0.98, bp_arbiter.prediction_in_decimal))
        preds = [v1, v2, v3]

        # ── Devil's-advocate pass if all cluster near 50% ────────────── #
        mean_initial = statistics.mean(preds)
        lo, hi = self.DEVIL_ADVOCATE_ZONE
        if lo <= mean_initial <= hi:
            logger.info(
                f"[DEVIL] Predictions near 50% (mean={mean_initial:.2f}). "
                "Running devil's-advocate sub-prompt."
            )
            devil_prompt = clean_indents(
                f"""
                {self._system_prompt()}

                Three forecasters independently estimated this question near 50%,
                indicating maximum uncertainty. Your mission: search for a single
                decisive factor that BREAKS the uncertainty and moves the probability
                clearly above 65% or below 35%.

                QUESTION: {question.question_text}
                RESOLUTION CRITERIA: {question.resolution_criteria}
                RESEARCH BRIEF: {research}

                (a) Is there any decisive factor being systematically overlooked?
                (b) Does the resolution criteria have an asymmetry favouring one side?
                (c) What does history say about questions of this type — do they tend
                    to resolve YES or NO?

                If you find a decisive factor: give your updated probability.
                If genuinely uncertain: stay near 50%.

                End with exactly: "Probability: ZZ%"
                """
            )
            devil_r = await self.get_llm("arbiter", "llm").invoke(devil_prompt)
            bp_devil: BinaryPrediction = await structure_output(
                devil_r, BinaryPrediction,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )
            v4 = max(0.02, min(0.98, bp_devil.prediction_in_decimal))
            preds.append(v4)
            cat_w.append(1.0)
            logger.info(f"[DEVIL] Devil's-advocate vote: {v4:.3f}")

        final_pred = _aggregate_binary_predictions(preds, weights=cat_w[:len(preds)])

        combined_reasoning = clean_indents(
            f"""
            ## Primary Reasoning (GPT-5.4, weight={cat_w[0]})
            {primary_r}

            ## Challenger Reasoning (GPT-5.2, weight={cat_w[1]})
            {challenger_r}

            ## Arbiter Reasoning (Claude Sonnet 4.6, weight={cat_w[2]})
            {arbiter_r}

            ---
            ## Ensemble Summary
            Category: {category}
            Raw votes: GPT-5.4={v1:.3f}, GPT-5.2={v2:.3f}, Claude={v3:.3f}
            {"Devil's advocate: "+str(round(preds[3],3) if len(preds)>3 else 'N/A')}
            Final aggregated prediction: {final_pred:.3f}
            """
        )

        logger.info(
            f"[SERAF] Binary {question.page_url} "
            f"votes={[round(p,3) for p in preds[:3]]} "
            f"cat={category} final={final_pred:.3f}"
        )
        return ReasonedPrediction(prediction_value=final_pred, reasoning=combined_reasoning)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        """Compatibility shim used by conditional question helper."""
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        bp: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        return ReasonedPrediction(
            prediction_value=max(0.02, min(0.98, bp.prediction_in_decimal)),
            reasoning=reasoning,
        )

    ##################################### MULTIPLE CHOICE #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        community_prior = _community_prior_text(question)
        prompt = clean_indents(
            f"""
            {self._system_prompt()}

            QUESTION: {question.question_text}

            OPTIONS: {question.options}

            BACKGROUND:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            RESEARCH BRIEF:
            {research}

            {community_prior}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Steps:
            (a) Time until resolution.
            (b) Historical base-rate distribution across these option categories.
            (c) Status quo — which option if nothing changes?
            (d) Scenario where the unexpected option wins.
            (e) Confirm probabilities sum to 100%.
            (f) Assign at least 2% to non-impossible tail options.

            {self._get_conditional_disclaimer_if_necessary(question)}

            Final answer — ALL options in order {question.options}:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            All option names must exactly match one of: {question.options}
            Remove any "Option" prefix if not part of the original name.
            Include 0%-probability options — do not skip them.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"[SERAF] MC forecast for {question.page_url}")
        pol: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        return ReasonedPrediction(prediction_value=pol, reasoning=reasoning)

    ##################################### NUMERIC #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        community_prior      = _community_prior_text(question)

        prompt = clean_indents(
            f"""
            {self._system_prompt()}

            QUESTION: {question.question_text}

            BACKGROUND:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            UNITS: {question.unit_of_measure if question.unit_of_measure else "Not stated — infer from context"}

            RESEARCH BRIEF:
            {research}

            {community_prior}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_msg}
            {upper_msg}

            Formatting rules (CRITICAL):
            - Use the stated units exactly as given.
            - No scientific notation.
            - Values MUST be STRICTLY INCREASING: P10 < P20 < P40 < P60 < P80 < P90.
              Double-check this before writing your final answer.

            Steps:
            (a) Time until resolution.
            (b) Central estimate if nothing changes (P50 anchor).
            (c) Trend-continuation estimate.
            (d) Expert / market consensus (cite if research supports it).
            (e) Low-scenario description (P10).
            (f) High-scenario description (P90).
            (g) SCOPE CHECK: Is the spread physically reasonable given bounds?
                Are fat tails accounted for?

            {self._get_conditional_disclaimer_if_necessary(question)}

            Err toward WIDER intervals. Unknown unknowns are real.

            Final answer:
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"[SERAF] Numeric forecast for {question.page_url}")

        parsing_instructions = clean_indents(
            f"""
            Parse a numeric forecast distribution.
            - Question: "{question.question_text}"
            - Units: {question.unit_of_measure}
            - Bounds: {question.lower_bound} to {question.upper_bound} {question.unit_of_measure}
            - Convert scientific notation to plain numbers.
            - Convert unit mismatches (e.g. "$500M" with units "B $" → 0.5).
            - Do NOT return output if no explicit percentiles are present.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        # Sanity check using challenger model
        checker_llm    = self.get_llm("challenger", "llm")
        percentile_list = await _sanity_check_numeric(
            question, percentile_list, checker_llm
        )

        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"[SERAF] Numeric {question.page_url}: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        community_prior      = _community_prior_text(question)

        prompt = clean_indents(
            f"""
            {self._system_prompt()}

            QUESTION: {question.question_text}

            BACKGROUND:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            RESEARCH BRIEF:
            {research}

            {community_prior}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_msg}
            {upper_msg}

            Formatting rules (CRITICAL):
            - All dates in YYYY-MM-DD format.
            - If time-of-day matters: YYYY-MM-DDTHH:MM:SSZ (UTC).
            - Dates MUST be in STRICTLY CHRONOLOGICAL order.
              P10 = EARLIEST, P90 = LATEST. Verify before writing.

            Steps:
            (a) Time until resolution.
            (b) Status-quo date (current trajectory unchanged).
            (c) Trend-continuation date.
            (d) Expert consensus date.
            (e) Earliest plausible scenario (P10).
            (f) Latest plausible scenario (P90).

            {self._get_conditional_disclaimer_if_necessary(question)}

            Final answer:
            Percentile 10: YYYY-MM-DD
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD
            """
        )
        return await self._date_prompt_to_forecast(question, prompt)

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"[SERAF] Date forecast for {question.page_url}")

        parsing_instructions = clean_indents(
            f"""
            Parse a date forecast distribution.
            - Question: "{question.question_text}"
            - Bounds: {question.lower_bound} to {question.upper_bound}
            - Format as valid ISO datetime strings (midnight UTC if no hour given).
            - Do NOT return output if no explicit percentiles are present.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        percentile_list = [
            Percentile(percentile=dp.percentile, value=dp.value.timestamp())
            for dp in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"[SERAF] Date {question.page_url}: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### HELPERS #####################################

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            ub   = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
            lb   = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
            unit = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            ub   = question.upper_bound.date().isoformat()
            lb   = question.lower_bound.date().isoformat()
            unit = ""
        else:
            raise ValueError(f"[SERAF] Unsupported question type: {type(question)}")

        upper_msg = (
            f"The question creator thinks the answer is unlikely to exceed {ub} {unit}."
            if question.open_upper_bound
            else f"The answer cannot exceed {ub} {unit}."
        )
        lower_msg = (
            f"The question creator thinks the answer is unlikely to be below {lb} {unit}."
            if question.open_lower_bound
            else f"The answer cannot be below {lb} {unit}."
        )
        return upper_msg, lower_msg

    ##################################### CONDITIONAL #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, full_research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Reasoning
            {parent_info.reasoning}
            ## Child Reasoning
            {child_info.reasoning}
            ## Yes-Branch Reasoning
            {yes_info.reasoning}
            ## No-Branch Reasoning
            {no_info.reasoning}
            """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,
            child=child_info.prediction_value,
            prediction_yes=yes_info.prediction_value,
            prediction_no=no_info.prediction_value,
        )
        return ReasonedPrediction(reasoning=full_reasoning, prediction_value=full_prediction)

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            previous_forecast = previous_forecasts[-1]
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > datetime.now(timezone.utc)
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)
                return (
                    ReasonedPrediction(
                        prediction_value=PredictionAffirmed(),
                        reasoning=f"Existing forecast reaffirmed at {pretty_value}.",
                    ),
                    research,
                )

        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer
        qt = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {qt} Question — Prior Forecast
            Value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            Reasoning:
            ```
            {reasoning.reasoning}
            ```
            IMPORTANT: Do NOT re-forecast the {qt} question. Use as context only.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            CONDITIONAL QUESTION: Forecast the CHILD question ONLY, treating the
            parent's resolution as a fixed given fact. Never re-forecast the parent.
            """
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").propagate = False
    logging.getLogger("httpx").setLevel(logging.WARNING)

    arg_parser = argparse.ArgumentParser(description="Run SERAF v2")
    arg_parser.add_argument(
        "--mode",
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
    )
    args    = arg_parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    # ──────────────────────────────────────────────────────────────────────────
    # MODEL STRINGS (OpenRouter, March 2026)
    # Update these if OpenRouter changes slugs — all other code uses slots only.
    # ──────────────────────────────────────────────────────────────────────────
    seraf_bot = Seraf(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            "default": GeneralLlm(
                model="openrouter/openai/gpt-5.4",
                temperature=0.3,
                timeout=120,
                allowed_tries=3,
            ),
            "challenger": GeneralLlm(
                model="openrouter/openai/gpt-5.2",
                temperature=0.3,
                timeout=120,
                allowed_tries=3,
            ),
            "arbiter": GeneralLlm(
                model="openrouter/claude-sonnet-4.6",
                temperature=0.3,
                timeout=120,
                allowed_tries=3,
            ),
            "parser": GeneralLlm(
                model="openrouter/openai/gpt-5.2",
                temperature=0.0,
                timeout=60,
                allowed_tries=3,
            ),
        },
    )

    client = MetaculusClient()

    if run_mode == "tournament":
        seasonal_reports = asyncio.run(
            seraf_bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            seraf_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_reports + minibench_reports

    elif run_mode == "metaculus_cup":
        seraf_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            seraf_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )

    elif run_mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
        ]
        seraf_bot.skip_previously_forecasted_questions = False
        questions = [client.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
        forecast_reports = asyncio.run(
            seraf_bot.forecast_questions(questions, return_exceptions=True)
        )

    seraf_bot.log_report_summary(forecast_reports)
