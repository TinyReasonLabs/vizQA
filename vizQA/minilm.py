"""
ONNX inference module for MiniLM model.
"""

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from vizQA.logger import get_logger
from vizQA.parser import ParserVocabulary


# pylint: disable=too-many-instance-attributes
class MiniLM:
    """
    Handles MiniLM ONNX inference for semantic similarity and intent classification.
    """

    # Anchor word lists — used to build pre-computed embedding groups
    _COLOR_ANCHORS = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "gray"]
    _STATE_ANCHORS = ["disabled", "enabled", "visible", "invisible", "hidden", "checked", "unchecked", "active"]
    _POSITION_ANCHORS = [
        "top",
        "bottom",
        "left",
        "right",
        "center",
        "middle",
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
    ]
    _NEGATION_ANCHORS = [
        "element is not present",
        "element is no longer visible",
        "element has disappeared",
        "element was removed",
        "element is absent",
        "element is hidden from view",
        "element is no longer displayed",
        "element has vanished",
        "element is gone from the page",
        "element is not showing anymore",
        "element should be done",
        # state transitions
        "element disappears",
        "element vanishes",
        "element gets removed",
        "element closes",
        "element is done",
        "element is out of view",
        "element is not in view",
        "element is invisible",
        "element is not visible",
    ]

    _POSITIVE_ANCHORS = [
        "element is present",
        "element is visible",
        "element is displayed",
        "element appears on the page",
        "element becomes visible",
        "element is shown",
        "element exists on the screen",
        "element is active and visible",
        "element is currently displayed",
        # state transitions
        "element appears",
        "element shows up",
        "element becomes visible",
        "element opens",
        "element comes into view",
        "element is in view",
    ]

    _ACTION_ANCHOR_GROUPS = {
        "click": [
            "click the button",
            "click on the link",
            "tap the icon",
            "press the button",
            "hit submit",
            "click navigation item",
            "click sidebar link",
            "navigate to page",
            "go to section",
            "click on the menu item",
        ],
        "right-click": [
            "right-click the element",
            "right-click on the image",
            "context-click",
            "open context menu",
            "right click the link",
            "perform right-click",
        ],
        "type": ["type into the field", "enter text", "input the password", "fill in the form"],
        "hover": ["hover over the element", "move mouse to", "point at the icon"],
        "scroll": ["scroll down", "scroll to the bottom", "scroll the list", "drag scrollbar"],
        "select": ["select from the dropdown", "choose an option", "pick from the list"],
        "check": ["check the box", "tick the checkbox", "mark as done"],
        "drag": ["drag the element", "drag and drop", "pull the slider"],
        "clear": ["clear the field", "empty the input", "erase the text"],
        "wait": ["wait for the element", "pause for 2 seconds", "sleep", "wait until visible"],
    }
    _KEY_NAMES = {"enter", "escape", "esc", "tab", "backspace", "space", "delete", "return", "shift", "ctrl", "alt"}

    def __init__(self, model_dir: str):
        self._logger = get_logger()
        self.model_path = os.path.join(model_dir, "model.onnx")
        self.tokenizer_path = os.path.join(model_dir, "tokenizer.json")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {self.tokenizer_path}")

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)

        # Load ONNX model
        self.session = ort.InferenceSession(self.model_path)
        self.input_names = [i.name for i in self.session.get_inputs()]

        # Pre-compute semantic anchors for step decomposition
        self._action_anchors = self._compute_anchor_embeddings(
            ["click", "type", "enter", "press", "hover", "verify", "check", "ensure", "assert"]
        )
        self._target_anchors = self._compute_anchor_embeddings(
            ["button", "field", "input", "link", "icon", "text", "element", "box", "modal", "toast", "alert", "menu"]
        )
        self._conjunction_anchors = self._compute_anchor_embeddings(["and", "then", "after", "next", ",", "also"])

        # Pre-compute intent classification anchor groups
        self._intent_anchor_groups: Dict[str, np.ndarray] = {
            "color": self._compute_anchor_embeddings(self._COLOR_ANCHORS),
            "state": self._compute_anchor_embeddings(self._STATE_ANCHORS),
            "position": self._compute_anchor_embeddings(self._POSITION_ANCHORS),
            "negation": self._compute_anchor_embeddings(self._NEGATION_ANCHORS),
            "positive": self._compute_anchor_embeddings(self._POSITIVE_ANCHORS),
        }

        # Pre-compute action anchors
        self._action_groups: Dict[str, np.ndarray] = {
            name: self._compute_anchor_embeddings(anchors) for name, anchors in self._ACTION_ANCHOR_GROUPS.items()
        }

        # Pre-compute action synonyms for semantic dissection
        all_syns = []
        for ct, syn_list in ParserVocabulary.ACTION_VERBS.items():
            for s in syn_list:
                all_syns.append((s.lower(), ct))
        self._all_syns_sorted = sorted(all_syns, key=lambda x: len(x[0]), reverse=True)
        self._action_syn_set = set(s for s, _ in self._all_syns_sorted)

    def _compute_anchor_embeddings(self, anchors: List[str]) -> np.ndarray:
        """Runs the anchor words through the model to get their 384D representations."""
        embeddings = []
        for word in anchors:
            encoding = self.tokenizer.encode(word)
            inputs = {"input_ids": [encoding.ids], "attention_mask": [encoding.attention_mask]}
            if "token_type_ids" in self.input_names:
                inputs["token_type_ids"] = [encoding.type_ids]

            output = self.session.run(None, inputs)[0]  # shape: (1, seq, 384)
            # Pool by taking the mean across the sequence length
            pooled = np.mean(output[0], axis=0)
            embeddings.append(pooled)

        return np.array(embeddings)  # shape: (num_anchors, 384)

    def encode(self, text: str) -> np.ndarray:
        """Embed a single string into a 384D vector (mean-pooled)."""
        encoding = self.tokenizer.encode(text)
        inputs = {"input_ids": [encoding.ids], "attention_mask": [encoding.attention_mask]}
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = [encoding.type_ids]
        output = self.session.run(None, inputs)[0]  # (1, seq, 384)
        return np.mean(output[0], axis=0)  # (384,)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Returns cosine similarity between two 1D vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def best_anchor_similarity_centroid(self, vec: np.ndarray, anchor_matrix: np.ndarray) -> float:
        """
        Returns cosine similarity between vec and the centroid of anchor_matrix
        (replaces max pooling with centroid-based similarity).
        """
        # Normalize input vector
        norm_v = np.linalg.norm(vec)
        if norm_v == 0:
            return 0.0
        vec_n = vec / norm_v

        # Compute centroid of anchors
        centroid = np.mean(anchor_matrix, axis=0)

        # Normalize centroid
        norm_c = np.linalg.norm(centroid)
        if norm_c == 0:
            return 0.0
        centroid_n = centroid / norm_c

        # Cosine similarity
        return float(np.dot(vec_n, centroid_n))

    def best_anchor_similarity(self, vec: np.ndarray, anchor_matrix: np.ndarray) -> float:
        """Returns the highest cosine similarity between vec and any row in anchor_matrix."""
        norms_a = np.linalg.norm(anchor_matrix, axis=-1, keepdims=True)
        norms_a = np.where(norms_a == 0, 1e-10, norms_a)
        norm_v = np.linalg.norm(vec)
        if norm_v == 0:
            return 0.0
        sims = np.dot(anchor_matrix / norms_a, vec / norm_v)
        return float(np.max(sims))

    def classify_anchor_group(
        self,
        text: str,
        groups: Optional[Dict[str, np.ndarray]] = None,
        threshold: float = 0.50,
    ) -> Optional[str]:
        """
        Classifies text into one of the named anchor groups by cosine similarity.

        Returns the name of the best-matching group if its similarity exceeds *threshold*,
        otherwise returns None.  Uses the pre-computed intent anchor groups when *groups*
        is not provided.
        """
        if groups is None:
            groups = self._intent_anchor_groups

        vec = self.encode(text)
        best_group: Optional[str] = None
        best_sim = -1.0

        for group_name, anchor_matrix in groups.items():
            sim = self.best_anchor_similarity(vec, anchor_matrix)
            if sim > best_sim:
                best_sim = sim
                best_group = group_name

        if best_sim >= threshold:
            return best_group
        return None

    def is_negation(self, text: str, threshold: float = 0.4, logit_threshold: float = 0.7, delta: float = 0.02) -> bool:
        """
        Returns True if *text* semantically resembles a negation intent.

        This compares similarity against negation anchors and ensures the
        negation match is stronger than the positive match to avoid false positives.
        """
        vec = self.encode(text)
        sim_neg = self.best_anchor_similarity(vec, self._intent_anchor_groups["negation"])
        sim_pos = self.best_anchor_similarity(vec, self._intent_anchor_groups["positive"])
        self._logger.log_debug("is_negation", f"text={text!r}, sim_neg={sim_neg:.3f}, sim_pos={sim_pos:.3f}")
        # print(f"text={text!r}, sim_neg={sim_neg:.3f}, sim_pos={sim_pos:.3f}")
        # print(f"threshold={threshold:.3f}")
        self._logger.log_debug("is_negation", f"threshold={threshold:.3f}")

        # logit-style
        score = sim_neg - sim_pos
        prob = 1 / (1 + math.exp(-score * 5))  # k ~ 5–10
        # print(f"prob={prob:.3f}")
        if prob > logit_threshold and sim_neg >= 0.25:
            return True
        # return prob > threshold

        # margin-based
        margin = sim_neg - sim_pos
        # print(f"margin={margin:.3f}")
        if margin >= delta * 2.0 and sim_neg >= 0.25:
            return True
        return prob > logit_threshold and sim_neg >= threshold
        return prob > logit_threshold and margin >= delta and sim_neg >= threshold

        # absolute threshold
        # return sim_neg >= threshold and sim_neg > sim_pos

    def semantic_match(self, query: str, candidates: List[str], threshold: float = 0.7) -> List[int]:
        """Returns indices of candidates whose similarity to query exceeds threshold."""
        q_vec = self.encode(query)
        matched = []
        for i, cand in enumerate(candidates):
            if not cand:
                continue
            c_vec = self.encode(cand)
            sim = self.cosine_similarity(q_vec, c_vec)
            if sim >= threshold:
                matched.append(i)
        return matched

    def rank_candidates(self, query: str, candidates: List[str], threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Returns all candidates with their similarity score, sorted descending.
        Only candidates at or above *threshold* are included.

        Each entry is ``{"index": int, "text": str, "score": float}``.
        """
        q_vec = self.encode(query)
        results = []
        for i, cand in enumerate(candidates):
            if not cand:
                continue
            c_vec = self.encode(cand)
            sim = self.cosine_similarity(q_vec, c_vec)
            if sim >= threshold:
                results.append({"index": i, "text": cand, "score": sim})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Low-level / generative path (kept for completeness, not primary path)
    # ------------------------------------------------------------------

    def predict(self, prompt: str) -> List[Dict[str, str]]:
        """
        Runs inference on the prompt and returns decomposed steps.
        Expected output format: [{"type": "FIND", "value": "..."}, ...]
        """
        encoding = self.tokenizer.encode(prompt)
        input_ids = encoding.ids
        attention_mask = encoding.attention_mask

        inputs = {"input_ids": [input_ids], "attention_mask": [attention_mask]}
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = [encoding.type_ids]

        outputs = self.session.run(None, inputs)
        result = self._parse_outputs(outputs, prompt)

        if not isinstance(result, list):
            raise ValueError("Model output is not a list of steps")

        for step in result:
            if not all(k in step for k in ("type", "value")):
                raise ValueError(f"Malformed step in model output: {step}")

        return result

    def _parse_outputs(self, outputs: Any, prompt: str) -> List[Dict[str, str]]:
        """
        Parses raw ONNX outputs into structured steps.
        Handles both generative models (logits/token IDs) and encoder models (embeddings).
        """

        try:
            output_tensor = outputs[0]

            if len(output_tensor.shape) == 2:
                token_ids = output_tensor[0]
                decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            elif len(output_tensor.shape) == 3 and output_tensor.shape[-1] > 1000:
                token_ids = np.argmax(output_tensor[0], axis=-1)
                decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            elif len(output_tensor.shape) == 3:
                return self._semantic_dissection(prompt)
            else:
                raise ValueError(f"Unexpected output tensor shape: {output_tensor.shape}")

            if not decoded_text:
                raise ValueError("Model produced an empty response")

            try:
                steps = json.loads(decoded_text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Model output is not valid JSON: {decoded_text}") from e

            if not isinstance(steps, list):
                raise ValueError(f"Model output must be a list of steps, got: {type(steps).__name__}")

            for i, step in enumerate(steps):
                if not isinstance(step, dict) or "type" not in step or "value" not in step:
                    raise ValueError(f"Step {i} is malformed or missing keys: {step}")

            return steps

        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            raise RuntimeError(f"MiniLM Deserialization Error: {e}") from e

    # ── Semantic Dissection Helpers ──────────────────────────────────────────

    def _sd_protect(self, prompt: str) -> Tuple[str, List[str]]:
        """Quotes remain unchanged, but their contents are protected from regex splits."""
        quotes: List[str] = []

        def _repl(match):
            quotes.append(match.group(0))
            return f" __QUOTE_{len(quotes)-1}__ "

        protected = re.sub(r"(['\"])(.*?)\1", _repl, prompt)
        # Upfront noise stripping
        protected = re.sub(r"\bright\s+then\b", "then", protected, flags=re.I)
        protected = re.sub(
            r"^(?:\s*)(?:please\s*(?:navigate\s*ahead\s*)?|i\s*want\s*you\s*to\s*)(?:and\s*)?",
            "",
            protected,
            flags=re.I,
        )
        # Protect "press and hold" from 'and' splitting
        protected = re.sub(r"\bpress\s+and\s+hold\b", "press_and_hold", protected, flags=re.I)
        return protected, quotes

    def _sd_restore(self, text: str, quotes: List[str]) -> str:
        """Restores protected quote contents."""
        for i, q in enumerate(quotes):
            text = text.replace(f"__QUOTE_{i}__", q.strip())
        text = text.replace("press_and_hold", "press and hold")
        return text

    def _sd_clean_target(self, text: str) -> str:
        """Removes leading articles and prepositions (the, a, an, over, at, on, in, into, from, of)."""
        clean = re.sub(r"^(the|a|an|over|at|on|in|into|from|of)\s+", "", text, flags=re.I).strip()
        # Also strip trailing 'key' if after a recognized key name
        return clean

    def _sd_find_verb(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Returns (literal_syn, canonical_type) or (None, None)."""
        t = text.lower()
        for s, ct in self._all_syns_sorted:
            if s == "input" and re.search(r"\bthe\s+input\b", t):
                continue
            if re.search(rf"\b{re.escape(s)}\b", t):
                return s, ct
        return None, None

    def _sd_rhs_starts_with_verb(self, text: str) -> bool:
        """True if text starts with a recognized action verb."""
        t = re.sub(r"^\s*(the|a|an|please|over)\s+", "", text.lower()).strip()
        return any(re.match(rf"{re.escape(v)}\b", t) for v in self._action_syn_set)

    def _sd_get_clauses(self, protected: str) -> List[str]:
        """Splits the protected prompt into logical instruction clauses."""
        # High-level split: 'until' handling for VERIFY
        until_split = re.split(r"\buntil\b", protected, maxsplit=1, flags=re.I)
        if len(until_split) == 2:
            action_part = until_split[0].strip()
            if re.fullmatch(r"(pause|wait|sleep)", action_part.lower()):
                temp_clauses = ["VERIFY_FLAG " + until_split[1].strip()]
            else:
                temp_clauses = [
                    c.strip()
                    for c in re.split(r"\b(?:then|after|while|also)\b|->|=>", action_part, flags=re.I)
                    if c.strip()
                ]
                temp_clauses.append("VERIFY_FLAG " + until_split[1].strip())
        else:
            temp_clauses = [
                c.strip() for c in re.split(r"\b(?:then|after|while|also)\b|->|=>", protected, flags=re.I) if c.strip()
            ]

        # Smart 'and'/',' split — only when RHS STARTS with a recognized action verb
        clauses = []
        pattern_and = re.compile(r"(\band\b|,)", flags=re.I)
        for c in temp_clauses:
            tokens = pattern_and.split(c)
            current = tokens[0]
            for i in range(1, len(tokens), 2):
                rhs = tokens[i + 1]
                if self._sd_rhs_starts_with_verb(rhs):
                    if current.strip():
                        clauses.append(current.strip())
                    current = rhs
                else:
                    current += tokens[i] + rhs
            if current.strip():
                clauses.append(current.strip())
        return clauses

    def _sd_handle_verify(
        self, clause: str, real_clause: str, all_steps: List[Dict[str, str]]
    ) -> Tuple[bool, Optional[str]]:
        """Handles VERIFY patterns. Returns (handled, next_prev_target)."""
        lower_real = real_clause.lower()
        is_verify = (
            clause.startswith("VERIFY_FLAG ")
            or re.search(r"\b(verify|ensure|assert|make sure)\b", lower_real)
            or re.search(r"\bshould\b", lower_real)
        )
        if not is_verify:
            return False, None

        val = re.sub(r"^(VERIFY_FLAG\s+)", "", real_clause)
        val = re.sub(r"\b(verify|ensure|make sure|assert|that|the|a|an)\b", "", val, flags=re.I).strip()
        # Split VERIFY on 'and' to emit separate VERIFY steps
        for vp in re.split(r"\band\b", val, flags=re.I):
            vp = re.sub(r"\s+", " ", vp).strip()
            if vp:
                all_steps.append({"type": "VERIFY", "value": vp})
        return True, None

    def _sd_handle_bare_noun(
        self, real_clause: str, quotes: List[str], all_steps: List[Dict[str, str]]
    ) -> Tuple[bool, Optional[str]]:
        """Handles pattern where only a noun is provided (defaults to click). Returns (handled, next_prev_target)."""
        # This is called if no verb was found by find_verb
        noun = self._sd_clean_target(real_clause.lower())
        noun = self._sd_restore(noun, quotes)
        if not noun or noun.lower() in ["it", "them"]:
            raise ValueError(f"Could not identify a target element in '{real_clause}'")

        all_steps.append({"type": "FIND", "value": noun})
        all_steps.append({"type": "DO", "value": f"click {noun}"})
        return True, noun

    def _sd_handle_drag(
        self,
        canonical_type: str,
        target_area: str,
        real_clause: str,
        quotes: List[str],
        all_steps: List[Dict[str, str]],
    ) -> Tuple[bool, Optional[str]]:
        """Handles pattern 'drag [source] onto [dest]'. Returns (handled, next_prev_target)."""
        if canonical_type != "drag" or not re.search(r"\bonto\b", target_area, re.I):
            return False, None

        onto_parts = re.split(r"\bonto\b", target_area, maxsplit=1, flags=re.I)
        drag_tgt = self._sd_clean_target(self._sd_restore(onto_parts[0].strip(), quotes))
        drop_tgt = self._sd_clean_target(self._sd_restore(onto_parts[1].strip(), quotes))

        if not drag_tgt:
            raise ValueError(f"No source element identified for drag action in '{real_clause}'")
        if not drop_tgt:
            raise ValueError(f"No destination element identified for drop action in '{real_clause}'")

        all_steps.extend(
            [
                {"type": "FIND", "value": drag_tgt},
                {"type": "DO", "value": "drag"},
                {"type": "FIND", "value": drop_tgt},
                {"type": "DO", "value": "drop"},
            ]
        )
        return True, drop_tgt

    def _sd_handle_key_input(
        self,
        real_clause: str,
        target_area: str,
        canonical_type: str,
        prev_target: Optional[str],
        all_steps: List[Dict[str, str]],
    ) -> Tuple[bool, Optional[str]]:
        """Handles pattern 'press enter' or 'press [key]'. Returns (handled, next_prev_target)."""
        detected_key = None
        for k in self._KEY_NAMES:
            if re.search(rf"\b{re.escape(k)}\b", real_clause.lower()):
                detected_key = k
                break

        if not detected_key:
            return False, None

        # Robust key handling: "Enter" at start is an action, NOT a key press
        is_explicit_press = re.match(r"^\s*(press|hit|on|hit\s+the|press\s+the)\b", real_clause, flags=re.I)
        if not is_explicit_press and detected_key.lower() == "enter":
            return False, None

        if detected_key:
            # "press enter on the login button"
            element = re.sub(r"\b(press|hit|on)\b", "", real_clause, flags=re.I)
            element = re.sub(rf"\b{re.escape(detected_key)}\b", "", element, flags=re.I).strip()
            element = self._sd_clean_target(element)
            if element.lower().endswith(" key"):
                element = element[:-4].strip()

            if not element or element.lower() in ["it", "them"]:
                element = prev_target

            if not element:
                raise ValueError(f"No target element identified for key press '{detected_key}' in '{real_clause}'")

            all_steps.append({"type": "FIND", "value": element})
            all_steps.append({"type": "DO", "value": f"press {detected_key}"})
            return True, element

        return False, None

    def _sd_handle_type_enter(
        self,
        canonical_type: str,
        target_area: str,
        real_clause: str,
        quotes: List[str],
        prev_target: Optional[str],
        all_steps: List[Dict[str, str]],
    ) -> Tuple[bool, Optional[str]]:
        """Handles single or multi-field
        'type [payload] into [target]' or 'enter [payload]'.
        Returns (handled, next_prev_target).

        :param canonical_type: The canonical type of the action.
        :param target_area: The target area of the action.
        :param real_clause: The real clause of the action.
        :param quotes: The quotes of the action.
        :param prev_target: The previous target of the action.
        :param all_steps: The list of all steps.
        :return: A tuple of (handled, next_prev_target).
        """
        if canonical_type not in ["type", "enter"]:
            return False, None

        # ── Into/In/On split: distributive or multiple targets ──
        connector = None
        if re.search(r"\binto\b", target_area, re.I):
            connector = "into"
        elif re.search(r"\bin\b", target_area, re.I):
            connector = "in"

        if connector:
            and_parts = re.split(r"\band\b", target_area, flags=re.I)
            if len(and_parts) >= 2 and all(re.search(rf"\b{connector}\b", p, re.I) for p in and_parts):
                elem = None
                for part in and_parts:
                    part_split = re.split(rf"\b{connector}\b", part.strip(), maxsplit=1, flags=re.I)
                    if len(part_split) == 2:
                        pload = self._sd_restore(part_split[0].strip(), quotes)
                        elem = self._sd_clean_target(self._sd_restore(part_split[1].strip(), quotes))
                        if not elem:
                            raise ValueError(f"No target element identified for '{canonical_type}' action in '{part}'")
                        all_steps.append({"type": "FIND", "value": elem})
                        all_steps.append({"type": "DO", "value": f"{canonical_type} {pload}"})
                return True, elem

            # Single split: 'payload' in/into element
            part_split = re.split(rf"\b{connector}\b", target_area, maxsplit=1, flags=re.I)
            if len(part_split) == 2:
                pload = self._sd_restore(part_split[0].strip(), quotes)
                elem = self._sd_clean_target(self._sd_restore(part_split[1].strip(), quotes))
                if not elem or elem.lower() in ["it", "them"]:
                    elem = prev_target
                if not elem:
                    raise ValueError(f"No target element identified for '{canonical_type}' action in '{real_clause}'")

                all_steps.append({"type": "FIND", "value": elem})
                all_steps.append({"type": "DO", "value": f"{canonical_type} {pload}"})
                return True, elem

        # ── Multi-field fallback: "type first name john and last name doe" ──
        restored_target = self._sd_restore(target_area, quotes)
        and_parts = re.split(r"\band\b", restored_target, flags=re.I) if restored_target else []
        if len(and_parts) >= 2:
            last_elem = None
            for part in and_parts:
                part = part.strip()
                words = part.split()
                if len(words) >= 2:
                    last_elem = " ".join(words[:-1])
                    all_steps.append({"type": "FIND", "value": last_elem})
                    all_steps.append({"type": "DO", "value": f"{canonical_type} {words[-1]}"})
                elif words:
                    if not prev_target:
                        raise ValueError(
                            f"Ambiguous segment '{part}' in multi-field action '{restored_target}': No target specified."
                        )
                    last_elem = prev_target
                    all_steps.append({"type": "FIND", "value": last_elem})
                    all_steps.append({"type": "DO", "value": f"{canonical_type} {words[0]}"})
            return True, last_elem

        return False, None

    def _sd_handle_select(
        self,
        canonical_type: str,
        target_area: str,
        real_clause: str,
        quotes: List[str],
        prev_target: Optional[str],
        all_steps: List[Dict[str, str]],
    ) -> Tuple[bool, Optional[str]]:
        """Handles pattern 'select [option] from [target]'
        or distributive select.

        :param canonical_type: The canonical type of the action.
        :param target_area: The target area of the action.
        :param real_clause: The real clause of the action.
        :param quotes: The quotes of the action.
        :param prev_target: The previous target of the action.
        :param all_steps: The list of all steps.
        :return: A tuple of (handled, next_prev_target).
        """
        if canonical_type != "select":
            return False, None

        if re.search(r"\bfrom\b", target_area, re.I):
            from_split = re.split(r"\bfrom\b", target_area, maxsplit=1, flags=re.I)
            pload = self._sd_restore(from_split[0].strip(), quotes)
            elem_val = self._sd_clean_target(self._sd_restore(from_split[1].strip(), quotes))
            if not elem_val:
                raise ValueError(f"No target element identified for 'select' action in '{real_clause}'")
            all_steps.append({"type": "FIND", "value": elem_val})
            all_steps.append({"type": "DO", "value": f"select {pload}"})
            return True, elem_val

        if re.search(r"\band\b", target_area, re.I):
            # Distributive: 'apple' and 'banana' options
            and_parts = re.split(r"\band\b", target_area, flags=re.I)
            last_elem = None
            for part in and_parts:
                part_restored = self._sd_clean_target(self._sd_restore(part.strip(), quotes))
                if not part_restored:
                    raise ValueError(f"Could not identify a clear option to select in '{real_clause}'")

                # Note: this distributive logic currently assumes the target is prev_target
                tgt = prev_target or "element"
                all_steps.append({"type": "FIND", "value": tgt})
                all_steps.append({"type": "DO", "value": f"select {part_restored}"})
                last_elem = tgt
            return True, last_elem

        return False, None

    def _sd_handle_distributive_and(
        self,
        canonical_type: str,
        target_area: str,
        real_clause: str,
        quotes: List[str],
        saved_literal: str,
        all_steps: List[Dict[str, str]],
    ) -> Tuple[bool, Optional[str]]:
        """Handles distributive 'and' (e.g., 'click submit and cancel buttons'). Returns (handled, next_prev_target)."""
        if canonical_type in ["click", "right-click", "hover", "select", "check"] and re.search(
            r"\band\b", target_area, re.I
        ):
            and_parts = re.split(r"\band\b", target_area, flags=re.I)
            if not any(self._sd_rhs_starts_with_verb(p) for p in and_parts):
                restored_parts = [self._sd_clean_target(self._sd_restore(p.strip(), quotes)) for p in and_parts]
                last_words = restored_parts[-1].split()
                shared_suffix = last_words[-1] if last_words else ""
                for j, part in enumerate(restored_parts):
                    tgt = part.strip()
                    if j < len(restored_parts) - 1 and shared_suffix:
                        tgt = f"{tgt} {shared_suffix}"
                    if not tgt or tgt.lower() in ["it", "them"]:
                        raise ValueError(f"Could not identify a target element in '{real_clause}'")

                    all_steps.append({"type": "FIND", "value": tgt})
                    all_steps.append({"type": "DO", "value": saved_literal or canonical_type})
                return True, restored_parts[-1]

        return False, None

    def _sd_handle_press_and_hold(
        self, lower_real: str, prev_target: Optional[str], all_steps: List[Dict[str, str]]
    ) -> Tuple[bool, Optional[str]]:
        """Handles pattern 'press and hold [target]'. Returns (handled, next_prev_target)."""
        if "press and hold" in lower_real or "press_and_hold" in lower_real:
            rest = re.sub(r"\b(press\s+and\s+hold|press_and_hold)\b", "", lower_real, flags=re.I).strip()
            rest = self._sd_clean_target(rest)
            if not rest or rest.lower() in ["it", "them"]:
                rest = prev_target

            if not rest:
                raise ValueError(f"No target element identified for 'press and hold' in '{lower_real}'")

            all_steps.append({"type": "FIND", "value": rest})
            all_steps.append({"type": "DO", "value": "press-and-hold"})
            return True, rest

        return False, None

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def _semantic_dissection(self, prompt: str) -> List[Dict[str, str]]:
        """Modularized decomposition of a UI instruction into FIND/DO/VERIFY steps."""
        protected, quotes = self._sd_protect(prompt)
        clauses = self._sd_get_clauses(protected)

        all_steps: List[Dict[str, str]] = []
        prev_target = None

        for clause in clauses:
            real_clause = self._sd_restore(clause, quotes)
            lower_real = real_clause.lower()

            handled, nt = self._sd_handle_verify(clause, real_clause, all_steps)
            if handled:
                prev_target = nt
                continue

            literal_syn, canonical_type = self._sd_find_verb(real_clause)
            saved_literal = literal_syn

            if not canonical_type:
                handled, nt = self._sd_handle_bare_noun(real_clause, quotes, all_steps)
                if handled:
                    prev_target = nt
                    continue

            target_area = re.sub(rf"\b{re.escape(literal_syn)}\b", "", clause, count=1, flags=re.I).strip()

            handled, nt = self._sd_handle_drag(canonical_type, target_area, real_clause, quotes, all_steps)
            if handled:
                prev_target = nt
                continue

            handled, nt = self._sd_handle_key_input(real_clause, target_area, canonical_type, prev_target, all_steps)
            if handled:
                prev_target = nt
                continue

            handled, nt = self._sd_handle_type_enter(
                canonical_type, target_area, real_clause, quotes, prev_target, all_steps
            )
            if handled:
                prev_target = nt
                continue

            handled, nt = self._sd_handle_select(
                canonical_type, target_area, real_clause, quotes, prev_target, all_steps
            )
            if handled:
                prev_target = nt
                continue

            handled, nt = self._sd_handle_distributive_and(
                canonical_type, target_area, real_clause, quotes, saved_literal, all_steps
            )
            if handled:
                prev_target = nt
                continue

            handled, nt = self._sd_handle_press_and_hold(lower_real, prev_target, all_steps)
            if handled:
                prev_target = nt
                continue

            # 5. FINAL FALLBACK (standard verb + noun)
            target_val = self._sd_clean_target(self._sd_restore(target_area, quotes))
            if not target_val or target_val.lower() in ["it", "them"]:
                target_val = prev_target or "element"

            if not target_val:
                if canonical_type == "wait":
                    target_val = "page"
                else:
                    raise ValueError(
                        f"No target element identified for {saved_literal or canonical_type} in '{real_clause}'"
                    )

            final_verb = saved_literal or canonical_type
            if canonical_type == "wait":
                all_steps.append({"type": "FIND", "value": target_val})
                all_steps.append({"type": "DO", "value": f"wait {target_val}"})
            else:
                all_steps.append({"type": "FIND", "value": target_val})
                all_steps.append({"type": "DO", "value": final_verb})
            prev_target = target_val

        if not all_steps:
            raise ValueError(f"Could not decompose the instruction into any valid test steps: '{prompt}'")

        return all_steps
