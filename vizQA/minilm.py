"""
ONNX inference module for MiniLM model.
"""

import os
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
        import json  # pylint: disable=import-outside-toplevel

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
        import json  # pylint: disable=import-outside-toplevel

        try:
            output_tensor = outputs[0]

            if len(output_tensor.shape) == 2:
                token_ids = output_tensor[0]
                decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            elif len(output_tensor.shape) == 3 and output_tensor.shape[-1] > 1000:
                token_ids = np.argmax(output_tensor[0], axis=-1)
                decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            elif len(output_tensor.shape) == 3:
                return self._semantic_dissection(output_tensor[0], prompt)
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

    def _semantic_dissection(self, token_embeddings: np.ndarray, prompt: str) -> List[Dict[str, str]]:
        """
        Uses embeddings and rule-based chunking to decompose a prompt into steps.
        Leverages MiniLM for robust intent classification of clauses.
        """
        import re

        # 1. Protect quotes to keep them as atomic units
        quotes: List[str] = []

        def _repl(match):
            quotes.append(match.group(0))
            return f" __QUOTE_{len(quotes)-1}__ "

        protected = re.sub(r"(['\"])(.*?)\1", _repl, prompt)

        # 2. Split into clauses using standard conjunctions
        split_pattern = re.compile(r"\b(?:and|then|after|while|also)\b|,|->|=>", re.I)
        clauses = [c.strip() for c in split_pattern.split(protected) if c.strip()]

        all_steps = []

        for clause in clauses:
            # Restore quotes for this specific clause to get better embedding
            real_clause = clause
            for i, q in enumerate(quotes):
                real_clause = real_clause.replace(f"__QUOTE_{i}__", q.strip())

            # 3. Determine intent of this chunk
            chunk_vec = self.encode(real_clause)

            is_verify = (
                "verify" in real_clause.lower()
                or "ensure" in real_clause.lower()
                or "should" in real_clause.lower()
                or "assert" in real_clause.lower()
            )

            if is_verify:
                # VERIFY path
                val = re.sub(r"\b(verify|ensure|assert|that|the|a|an)\b", "", real_clause, flags=re.I).strip()
                all_steps.append({"type": "VERIFY", "value": val})
                continue

            # 4. ACTION path (FIND + DO)
            # Use semantic grounding via anchor groups
            best_verb = self.classify_anchor_group(real_clause, groups=self._action_groups, threshold=0.45)
            if not best_verb:
                # Fallback to single verb similarity if group classification fails
                best_sim = -1.0
                action_words = list(self._ACTION_ANCHOR_GROUPS.keys())
                for v in action_words:
                    v_vec = self.encode(v)
                    s = self.cosine_similarity(chunk_vec, v_vec)
                    if s > best_sim:
                        best_sim = s
                        best_verb = v

            if not best_verb:
                best_verb = "interact"

            # Clean the chunk to find target/payload
            # Note: We use the protected version for easier regexing then restore
            target_area = clause
            # Remove ALL action words from target if they appear
            action_words = list(self._ACTION_ANCHOR_GROUPS.keys())
            for v in action_words:
                target_area = re.sub(rf"\b{v}\b", "", target_area, flags=re.I).strip()

            # Remove common instruction noise
            target_area = re.sub(
                r"\b(please navigate ahead and|i want you to|also|then)\b", "", target_area, flags=re.I
            ).strip()

            # Preposition stripping (can be anywhere if verb was removed)
            target_area = re.sub(
                r"\b(into|onto|to|on|over|in|at|from|with|inside)\b", "", target_area, flags=re.I
            ).strip()
            target_area = re.sub(r"\b(the|a|an)\b", "", target_area, flags=re.I).strip()

            # Restore quotes for target_area
            restored_target = target_area
            for i, q in enumerate(quotes):
                restored_target = restored_target.replace(f"__QUOTE_{i}__", q.strip())

            # Determine payload for 'type'/'enter'
            payload = ""
            if best_verb in ["type", "enter"]:
                # If there's a quoted string, it's the payload
                q_match = re.search(r"(['\"])(.*?)\1", restored_target)
                if q_match:
                    payload = q_match.group(0)
                    # Remove payload from target to isolate the element description
                    restored_target = restored_target.replace(payload, "").strip()
                else:
                    # heuristic: maybe the whole remainder is the payload if no locator logic?
                    # But for now let's stick to quotes
                    pass

            target_val = restored_target if restored_target else "element"
            # Final touch: remove extra spaces and characters
            target_val = re.sub(r"\s+", " ", target_val).strip()
            target_val = re.sub(r"^[,\.]|[,\.]$", "", target_val).strip()

            # Map specialized verbs
            final_verb = best_verb
            if best_verb in ["type", "enter"] and payload:
                final_verb = f"{best_verb} {payload}"

            if best_verb in ["type", "enter"] and not payload:
                # If no payload found in quotes, maybe the whole target IS the payload?
                # e.g. "type admin" -> FIND element, DO type admin
                if restored_target and restored_target != "element":
                    all_steps.append({"type": "FIND", "value": "element"})
                    all_steps.append({"type": "DO", "value": f"{best_verb} {restored_target}"})
                    continue

            all_steps.append({"type": "FIND", "value": target_val})
            all_steps.append({"type": "DO", "value": final_verb})

        if not all_steps:
            return [{"type": "FIND", "value": "element"}, {"type": "DO", "value": "interact"}]

        return all_steps
