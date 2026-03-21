"""
ONNX inference module for MiniLM model.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


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
        "the element should disappear",
        "the element is gone",
        "the element is no longer present",
        "the element vanished",
        "the element is hidden",
        "the element was removed",
        "the element is absent",
        "the element is not visible",
        "the element is done",
        "the spinner is done",
        "the loading is finished",
        "the modal is closed",
    ]
    _POSITIVE_ANCHORS = [
        "the element should appear",
        "the element is visible",
        "the element is present",
        "the element shows up",
        "the element is active",
        "the element exists",
        "the element is displayed",
        "the element is running",
        "the button appeared",
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

    def __init__(self, model_dir: str):
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

    def is_negation(self, text: str, threshold: float = 0.45) -> bool:
        """
        Returns True if *text* semantically resembles a negation intent.

        This compares similarity against negation anchors and ensures the
        negation match is stronger than the positive match to avoid false positives.
        """
        vec = self.encode(text)
        sim_neg = self.best_anchor_similarity(vec, self._intent_anchor_groups["negation"])
        sim_pos = self.best_anchor_similarity(vec, self._intent_anchor_groups["positive"])

        return sim_neg >= threshold and sim_neg > sim_pos

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

    def _semantic_dissection(self, prompt: str) -> List[Dict[str, str]]:
        """
        Vocabulary-grounded decomposition of a UI instruction into atomic FIND/DO/VERIFY steps.
        Uses ParserVocabulary for canonical action grounding and handles special sentence patterns.
        """
        from vizQA.parser import ParserVocabulary

        KEY_NAMES = {"enter", "escape", "esc", "tab", "backspace", "space", "delete", "return", "shift", "ctrl", "alt"}

        quotes: List[str] = []

        def _protect(match):
            quotes.append(match.group(0))
            return f" __QUOTE_{len(quotes)-1}__ "

        def _restore(text: str) -> str:
            for i, q in enumerate(quotes):
                text = text.replace(f"__QUOTE_{i}__", q.strip())
            return text

        protected = re.sub(r"(['\"])(.*?)\1", _protect, prompt)
        # Upfront noise stripping
        protected = re.sub(r"\bright\s+then\b", "then", protected, flags=re.I)
        protected = re.sub(
            r"^(?:\s*)(?:please\s*(?:navigate\s*ahead\s*)?|i\s*want\s*you\s*to\s*)(?:and\s*)?",
            "",
            protected,
            flags=re.I,
        )

        # Build verb lookup: sorted by length descending so multi-word verbs match first
        all_syns_sorted: List[tuple] = []
        for ct, syns in ParserVocabulary.ACTION_VERBS.items():
            for s in syns:
                all_syns_sorted.append((s.lower(), ct))
        all_syns_sorted.sort(key=lambda x: len(x[0]), reverse=True)
        action_syn_set = set(s for s, _ in all_syns_sorted)

        def find_verb(text: str):
            """Returns (literal_syn, canonical_type) or (None, None). Protects 'the input' from matching 'input' verb."""
            t = text.lower()
            for s, ct in all_syns_sorted:
                if s == "input" and re.search(r"\bthe\s+input\b", t):
                    continue
                if re.search(rf"\b{re.escape(s)}\b", t):
                    return s, ct
            return None, None

        def rhs_starts_with_verb(text: str) -> bool:
            """True if text starts with a recognized action verb (ignoring leading articles)."""
            t = re.sub(r"^\s*(the|a|an|please)\s+", "", text.lower()).strip()
            return any(re.match(rf"{re.escape(v)}\b", t) for v in action_syn_set)

        def strip_articles(text: str) -> str:
            text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.I).strip()
            return text

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
                if rhs_starts_with_verb(rhs):
                    if current.strip():
                        clauses.append(current.strip())
                    current = rhs
                else:
                    current += tokens[i] + rhs
            if current.strip():
                clauses.append(current.strip())

        all_steps: List[Dict[str, str]] = []
        prev_target = None  # For target inheritance when RHS clause has no explicit element

        for clause in clauses:
            real_clause = _restore(clause)
            lower_real = real_clause.lower()

            # ── VERIFY ──────────────────────────────────────────────────
            is_verify = (
                clause.startswith("VERIFY_FLAG ")
                or re.search(r"\b(verify|ensure|assert|make sure)\b", lower_real)
                or re.search(r"\bshould\b", lower_real)
            )
            if is_verify:
                val = re.sub(r"^(VERIFY_FLAG\s+)", "", real_clause)
                val = re.sub(r"\b(verify|ensure|make sure|assert|that|the|a|an)\b", "", val, flags=re.I).strip()
                # Split VERIFY on 'and' to emit separate VERIFY steps
                for vp in re.split(r"\band\b", val, flags=re.I):
                    vp = re.sub(r"\s+", " ", vp).strip()
                    if vp:
                        all_steps.append({"type": "VERIFY", "value": vp})
                prev_target = None
                continue

            # ── ACTION: find the verb ────────────────────────────────────
            literal_syn, canonical_type = find_verb(real_clause)
            saved_literal = literal_syn  # preserve before any reset

            # Bare noun (no verb) → click it
            if not canonical_type:
                noun = strip_articles(lower_real)
                noun = _restore(noun)
                if not noun or noun.lower() in ["it", "them"]:
                    raise ValueError(f"Could not identify a target element in '{real_clause}'")

                # Use the identified noun as the target instead of generic "element"
                all_steps.append({"type": "FIND", "value": noun})
                all_steps.append({"type": "DO", "value": f"click {noun}"})
                prev_target = noun
                continue

            # Remove verb from clause to obtain target area (work in protected form)
            target_area = re.sub(rf"\b{re.escape(literal_syn)}\b", "", clause, count=1, flags=re.I).strip()

            # ── DRAG + onto ── → split into drag+drop ───────────────────
            if canonical_type == "drag" and re.search(r"\bonto\b", target_area, re.I):
                onto_parts = re.split(r"\bonto\b", target_area, maxsplit=1, flags=re.I)
                drag_tgt = strip_articles(_restore(onto_parts[0]).strip())
                drop_tgt = strip_articles(_restore(onto_parts[1]).strip())
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
                prev_target = drop_tgt
                continue

            # ── KEY NAME detection ─────────────────────────────────────
            # Detect patterns like "Press Enter on the search box" or "Hit the Escape key"
            lower_target = target_area.lower()
            detected_key = None
            for key in sorted(KEY_NAMES, key=len, reverse=True):
                if re.search(rf"\b{re.escape(key)}\b", lower_target):
                    detected_key = key
                    break
            if detected_key and canonical_type in ["click", "type", "enter"]:
                rest = re.sub(rf"\b{re.escape(detected_key)}\b", "", lower_target, count=1, flags=re.I).strip()
                rest = re.sub(r"\bkey\b", "", rest, flags=re.I).strip()  # strip trailing 'key' word
                rest = re.sub(r"^(on|in|the|a|an)\s+", "", rest, flags=re.I).strip()
                element = _restore(rest) if rest else ""
                element = strip_articles(element)
                if not element or element.lower() in ["it", "them"]:
                    element = prev_target

                if not element:
                    raise ValueError(f"No target element identified for key press '{detected_key}' in '{real_clause}'")
                # Normalize verb to 'press' for key actions
                all_steps.append({"type": "FIND", "value": element})
                all_steps.append(
                    {
                        "type": "DO",
                        "value": (
                            f"press {detected_key} key"
                            if "key" in lower_real.replace(detected_key, "", 1).lower()
                            else f"press {detected_key}"
                        ),
                    }
                )
                prev_target = element
                continue

            # ── TYPE/ENTER: quoted payload + into/from ─────────────────
            payload = ""
            if canonical_type in ["type", "enter"]:
                q_tokens = re.findall(r"__QUOTE_\d+__", target_area)

                if q_tokens and re.search(r"\binto\b", target_area, re.I):
                    # Multiple or single "X into Y" patterns
                    and_into_parts = re.split(r"\band\b", target_area, flags=re.I)
                    if all(re.search(r"\binto\b", p, re.I) for p in and_into_parts):
                        for part in and_into_parts:
                            into_split = re.split(r"\binto\b", part.strip(), maxsplit=1, flags=re.I)
                            if len(into_split) == 2:
                                pload = _restore(into_split[0].strip())
                                elem = strip_articles(_restore(into_split[1].strip()))
                                if not elem:
                                    raise ValueError(
                                        f"No target element identified for '{saved_literal}' action in '{part}'"
                                    )
                                all_steps.append({"type": "FIND", "value": elem})
                                all_steps.append({"type": "DO", "value": f"{saved_literal} {pload}"})
                        prev_target = elem
                        continue
                    # Single into: 'payload' into element
                    into_split = re.split(r"\binto\b", target_area, maxsplit=1, flags=re.I)
                    if len(into_split) == 2:
                        pload = _restore(into_split[0].strip())
                        elem = strip_articles(_restore(into_split[1].strip()))
                        if not elem or elem.lower() in ["it", "them"]:
                            elem = prev_target
                        if not elem:
                            raise ValueError(
                                f"No target element identified for '{saved_literal}' action in '{real_clause}'"
                            )

                        all_steps.append({"type": "FIND", "value": elem})
                        all_steps.append({"type": "DO", "value": f"{saved_literal} {pload}"})
                        prev_target = elem
                        continue

                elif q_tokens:
                    # Quote is payload, remaining is element
                    first_q = q_tokens[0]
                    payload = first_q
                    target_area = target_area.replace(first_q, "").strip()

                elif re.search(r"\binto\b", target_area, re.I):
                    # Unquoted: "input mypassword into pass field"
                    into_split = re.split(r"\binto\b", target_area, maxsplit=1, flags=re.I)
                    if len(into_split) == 2:
                        pload = re.sub(
                            rf"\b{re.escape(literal_syn or canonical_type)}\b", "", into_split[0], count=1, flags=re.I
                        ).strip()
                        elem = strip_articles(_restore(into_split[1].strip()))
                        if not elem or elem.lower() in ["it", "them"]:
                            elem = prev_target
                        if not elem:
                            raise ValueError(
                                f"No target element identified for '{saved_literal}' action in '{real_clause}'"
                            )

                        all_steps.append({"type": "FIND", "value": elem})
                        all_steps.append({"type": "DO", "value": f"{saved_literal} {pload}"})
                        prev_target = elem
                        continue

            # ── SELECT: quoted payload with 'from' OR distributive 'and' ─
            if canonical_type == "select":
                q_tokens = re.findall(r"__QUOTE_\d+__", target_area)
                if q_tokens and re.search(r"\bfrom\b", target_area, re.I):
                    from_split = re.split(r"\bfrom\b", target_area, maxsplit=1, flags=re.I)
                    pload = _restore(from_split[0].strip())
                    elem_val = strip_articles(_restore(from_split[1].strip()))
                    if not elem_val:
                        raise ValueError(
                            f"No target element identified for '{saved_literal}' search in '{real_clause}'"
                        )
                    all_steps.append({"type": "FIND", "value": elem_val})
                    all_steps.append({"type": "DO", "value": f"{saved_literal} {pload}"})
                    prev_target = elem_val
                    continue
                elif q_tokens and re.search(r"\band\b", target_area, re.I):
                    # Distributive: 'apple' and 'banana' options
                    and_parts = re.split(r"\band\b", target_area, flags=re.I)
                    for part in and_parts:
                        part = part.strip()
                        part_restored = strip_articles(_restore(part))
                        if not part_restored:
                            raise ValueError(f"Could not identify a clear option to select in '{real_clause}'")
                        all_steps.append({"type": "FIND", "value": part_restored})
                        all_steps.append({"type": "DO", "value": saved_literal or canonical_type})
                    prev_target = part_restored
                    continue

            # ── Distributive 'and' for noun targets (click submit and cancel buttons) ──
            if canonical_type in ["click", "right-click", "hover", "select", "check"] and re.search(
                r"\band\b", target_area, re.I
            ):
                and_parts = re.split(r"\band\b", target_area, flags=re.I)
                if not any(rhs_starts_with_verb(p) for p in and_parts):
                    restored_parts = [strip_articles(_restore(p.strip())) for p in and_parts]
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
                    prev_target = restored_parts[-1]
                    continue

            # ── Press-and-hold special case ─────────────────────────────
            if "press and hold" in lower_real:
                rest = re.sub(r"\bpress\b", "", lower_real, count=1, flags=re.I)
                rest = re.sub(r"\band\s+hold\b", "", rest, flags=re.I).strip()
                rest = strip_articles(rest)
                if not rest or rest.lower() in ["it", "them"]:
                    rest = prev_target
                if not rest:
                    raise ValueError(f"No target element identified for 'press and hold' in '{real_clause}'")
                all_steps.append({"type": "FIND", "value": rest})
                all_steps.append({"type": "DO", "value": "press"})
                prev_target = rest
                continue

            # ── Strip verb from target area ─────────────────────────────
            if literal_syn:
                target_area = re.sub(rf"\b{re.escape(literal_syn)}\b", "", target_area, count=1, flags=re.I).strip()

            # Noise: strip leading prepositions and articles
            target_area = re.sub(
                r"^(into|onto|to|on|over|in|at|from|with|inside)\s+", "", target_area, flags=re.I
            ).strip()
            target_area = re.sub(r"^(the|a|an)\s+", "", target_area, flags=re.I).strip()

            restored_target = _restore(target_area)
            restored_payload = _restore(payload) if payload else ""

            target_val = restored_target if restored_target else ""
            if not target_val or target_val.lower() in ["it", "them"]:
                target_val = prev_target or ""

            if not target_val:
                # If wait action, it can default to 'the page' or 'anything' if it just wants to wait, but usually it needs a target
                if canonical_type == "wait":
                    target_val = "page"  # Special case for 'Wait 2 seconds'
                else:
                    raise ValueError(f"No target element identified for {final_verb} in '{real_clause}'")

            target_val = re.sub(r"\s+", " ", target_val).strip()
            target_val = re.sub(r"^[,.]|[,.]$", "", target_val).strip()

            final_verb = saved_literal or canonical_type

            if canonical_type == "wait":
                # Wait requires a clear target or it defaults to the previous one
                wait_tgt = target_val if target_val != "element" else prev_target
                if not wait_tgt:
                    raise ValueError(f"No target element identified for 'wait' action in '{real_clause}'")

                all_steps.append({"type": "FIND", "value": wait_tgt})
                all_steps.append({"type": "DO", "value": f"wait {target_val}"})
                prev_target = wait_tgt
                continue

            # ── Type/Enter payload building ─────────────────────────────
            if canonical_type in ["type", "enter"]:
                if restored_payload:
                    final_verb = f"{final_verb} {restored_payload}"
                else:
                    # Multi-field: "type first name john and last name doe"
                    and_parts = re.split(r"\band\b", restored_target, flags=re.I) if restored_target else []
                    if len(and_parts) >= 2:
                        for part in and_parts:
                            part = part.strip()
                            words = part.split()
                            if len(words) >= 2:
                                all_steps.append({"type": "FIND", "value": " ".join(words[:-1])})
                                all_steps.append({"type": "DO", "value": f"{final_verb} {words[-1]}"})
                            elif words:
                                if not prev_target:
                                    raise ValueError(
                                        f"Ambiguous segment '{part}' in multi-field action '{restored_target}': No target specified."
                                    )
                                all_steps.append({"type": "FIND", "value": prev_target})
                                all_steps.append({"type": "DO", "value": f"{final_verb} {words[0]}"})
                        prev_target = "element"  # mark as used
                        continue
                    else:
                        # Single field+value: last word = value, rest = field
                        parts = restored_target.split()
                        if len(parts) > 1 and restored_target != "element":
                            final_verb = f"{final_verb} {parts[-1]}"
                            target_val = " ".join(parts[:-1])
                        elif restored_target and restored_target != "element":
                            final_verb = f"{final_verb} {restored_target}"
                            if not prev_target:
                                raise ValueError(
                                    f"Ambiguous action '{final_verb}': No target element provided and no previous context."
                                )
                            target_val = prev_target

            # ── FIND-only action ────────────────────────────────────────
            if canonical_type == "find":
                if target_val != "element":
                    all_steps.append({"type": "FIND", "value": target_val})
                prev_target = target_val
                continue

            if target_val and target_val != "element":
                all_steps.append({"type": "FIND", "value": target_val})
            elif not prev_target:
                raise ValueError(
                    f"Ambiguous instruction: Could not identify target for '{real_clause}' and no previous context exists."
                )
            else:
                # Use previous target as fallback if current one is mission/generic
                all_steps.append({"type": "FIND", "value": prev_target})

            all_steps.append({"type": "DO", "value": final_verb})
            prev_target = target_val if target_val else prev_target

        if not all_steps:
            raise ValueError(f"Could not decompose the instruction into any valid test steps: '{prompt}'")

        return all_steps
