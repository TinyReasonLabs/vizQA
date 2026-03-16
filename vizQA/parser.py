"""
Semantic parser for UI testing instructions.

Provides rule-based AST parsing of natural language instructions into atomic
FIND / DO / VERIFY nodes, with optional MiniLM-powered intent classification
for verification queries.
"""

import re
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional

from vizQA.logger import get_logger

if TYPE_CHECKING:
    from vizQA.minilm import MiniLM


class SemanticNode(NamedTuple):
    """Atomic semantic unit produced by the parser."""

    type: str
    value: str


# ---------------------------------------------------------------------------
# Intent classification helpers (used by parse_verify_intent)
# ---------------------------------------------------------------------------

# Keyword-list fallbacks when MiniLM is not available
_COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "gray", "grey"]
_STATES = ["disabled", "enabled", "checked", "unchecked", "visible", "invisible", "hidden", "displayed", "active"]
_POSITIONS = [
    "top left",
    "top right",
    "bottom left",
    "bottom right",
    "top",
    "bottom",
    "left",
    "right",
    "center",
    "middle",
]
# Regex fast-path for negation — catches explicit literals
_NEGATION_RE = re.compile(
    r"\b(not|no longer|should not|shouldn't|should not be|disappear(?:s|ed)?|gone|invisible|absent|done|finished|closed|close|removed|vanish(?:es|ed)?|gone)\b",
    re.IGNORECASE,
)


class SemanticParser:
    """
    Advanced Rule-Based Engine (AST Parser) for dissecting UI testing instructions.
    Optionally enhanced with MiniLM embeddings for robust intent classification.
    """

    # Core Action Verbs
    ACTION_VERBS = {
        "click": ["click", "tap", "hit", "press"],
        "right-click": ["right-click", "context-click", "right click"],
        "type": ["type", "enter"],
        "hover": ["hover", "move to", "point"],
        "select": ["select", "choose", "pick"],
        "check": ["check", "tick"],
        "drag": ["drag"],
        "drop": ["drop"],
        "scroll": ["scroll"],
        "clear": ["clear"],
        "wait": ["wait", "pause", "sleep"],
        "find": ["find", "locate"],
    }

    # Verification verbs
    VERIFY_VERBS = ["verify", "ensure", "assert", "check that", "make sure"]

    # Conjunctions (Split Points)
    SPLIT_PATTERN = re.compile(r"\b(?:and|then|after|while)\b|,|->")

    def __init__(self, minilm: Optional["MiniLM"] = None):
        """
        Initialise the parser.

        Parameters
        ----------
        minilm:
            Optional pre-loaded MiniLM instance.  When provided, intent
            classification (color, state, position, negation) uses semantic
            similarity instead of plain keyword lists.
        """
        self.minilm = minilm
        self._logger = get_logger()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, instruction: str) -> List[SemanticNode]:
        """Parses a full natural language instruction into a list of atomic SemanticNodes."""
        nodes: List[SemanticNode] = []

        # Broad splits on explicit flow arrows which almost always mean VERIFY comes next
        parts = re.split(r"->|=>", instruction)
        nodes.extend(self._parse_clause(parts[0]))

        for verification_part in parts[1:]:
            nodes.extend(self._parse_verify(verification_part))

        return nodes

    def parse_verify_intent(self, query: str) -> Dict[str, Any]:
        """
        Parses a verification query to extract specific intents.

        Returns a dict with keys: ``keyword``, ``color``, ``position``,
        ``state``, ``negated``, ``subject``.

        When a MiniLM instance is provided, colors / states / positions are
        detected via cosine similarity (robust to synonyms).  Negation is
        detected via a fast regex pass **and** a semantic slow-path so that
        paraphrases like "the overlay should vanish" are also caught.
        """
        intent: Dict[str, Any] = {
            "keyword": None,
            "color": None,
            "position": None,
            "state": None,
            "negated": False,
            "subject": query,
        }
        subject = query  # working copy — we strip matched tokens as we go

        # 1. Extract quoted keyword
        quote_match = re.search(r"(['\"])(.*?)\1", query)
        if quote_match:
            intent["keyword"] = quote_match.group(2)
            subject = subject.replace(quote_match.group(0), "")

        # 2. Negation/Positive — structural approach
        # Explicit literals first
        is_negated = False
        is_positive = False

        if _NEGATION_RE.search(query):
            is_negated = True
        elif re.search(r"\b(appear|visible|shows?|exists?|present|displayed|open)\b", query, re.I):
            is_positive = True

        if not is_negated and not is_positive and self.minilm:
            # Semantic fallback
            neg_sim = self.minilm.best_anchor_similarity(
                self.minilm.encode(query), self.minilm._intent_anchor_groups["negation"]
            )
            pos_sim = self.minilm.best_anchor_similarity(
                self.minilm.encode(query), self.minilm._intent_anchor_groups["positive"]
            )
            if neg_sim > 0.6 or pos_sim > 0.6:
                if neg_sim > pos_sim:
                    is_negated = True
                else:
                    is_positive = True

        intent["negated"] = is_negated
        # Strip negation literals from subject if found
        if is_negated:
            subject = _NEGATION_RE.sub("", subject)

        # 3. Color detection
        if self.minilm:
            color_group = {c: self.minilm._intent_anchor_groups["color"] for c in _COLORS}
            # Use classify across individual words so multi-word queries work
            for word in query.lower().split():
                if word in _COLORS:
                    intent["color"] = word
                    subject = re.sub(rf"\b{re.escape(word)}\b", "", subject, flags=re.IGNORECASE)
                    break
            if not intent["color"]:
                # Semantic fallback: classify the full query
                detected = self.minilm.classify_anchor_group(
                    query, {"color": self.minilm._intent_anchor_groups["color"]}, threshold=0.60
                )
                if detected == "color":
                    # Try to pin down which color via keyword list
                    for c in _COLORS:
                        if re.search(rf"\b{re.escape(c)}\b", query, re.IGNORECASE):
                            intent["color"] = c
                            subject = re.sub(rf"\b{re.escape(c)}\b", "", subject, flags=re.IGNORECASE)
                            break
        else:
            lower_q = query.lower()
            for c in _COLORS:
                if re.search(rf"\b{c}\b", lower_q):
                    intent["color"] = c
                    subject = re.sub(rf"\b{c}\b", "", subject, flags=re.IGNORECASE)
                    break

        # 4. Position detection
        pos_regex = r"\b(top left|top right|bottom left|bottom right|top|bottom|left|right|center|centered|middle)\b"
        pos_match = re.search(pos_regex, query.lower())
        if pos_match:
            val = pos_match.group(1).lower()
            if val == "centered":
                val = "center"
            intent["position"] = val.replace(" ", "-")
            subject = re.sub(pos_regex, "", subject, flags=re.IGNORECASE)
        elif self.minilm:
            detected = self.minilm.classify_anchor_group(
                query, {"position": self.minilm._intent_anchor_groups["position"]}, threshold=0.55
            )
            if detected == "position":
                # Semantic match found but no explicit word; leave position as None
                # (we don't want to hallucinate which position)
                pass

        # 5. State detection
        if self.minilm:
            for s in _STATES:
                if re.search(rf"\b{s}\b", query.lower()):
                    intent["state"] = s
                    subject = re.sub(rf"\b{s}\b", "", subject, flags=re.IGNORECASE)
                    break
            if not intent["state"]:
                detected = self.minilm.classify_anchor_group(
                    query, {"state": self.minilm._intent_anchor_groups["state"]}, threshold=0.58
                )
                if detected == "state":
                    # Semantic match but no literal — leave state as None to avoid false positives
                    pass
        else:
            for s in _STATES:
                if re.search(rf"\b{s}\b", query.lower()):
                    intent["state"] = s
                    subject = re.sub(rf"\b{s}\b", "", subject, flags=re.IGNORECASE)
                    break

        # 6. Clean subject
        subject = re.sub(
            r"\b(should appear|should close|should occur|is|at the|in the|on the|of the|off the|the|a|an|located|aligned|should be|should|of|on|center of|screen|be|been|was|were|has|have|had)\b",
            "",
            subject,
            flags=re.IGNORECASE,
        )
        subject = re.sub(r"\s+", " ", subject).strip()
        intent["subject"] = subject
        return intent

    def filter_elements_by_intent(
        self,
        intent: Dict[str, Any],
        elements: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Filters a list of perception elements based on a parsed intent dict.

        Filtering is applied in priority order: semantics first, then color,
        then state.  A secondary filter that would produce an empty list is
        **silently skipped** — the previous non-empty list is kept instead,
        so semantics always take priority over positional / colour refinements.

        Parameters
        ----------
        intent:
            Dict produced by :meth:`parse_verify_intent`.
        elements:
            Raw perception element dicts with ``text``, ``label``, ``name``
            fields.

        Returns
        -------
        List[Dict[str, Any]]
            Filtered (and possibly re-ranked) element list.
        """
        if not elements:
            return []

        # Build candidate strings for semantic / substring matching
        # prioritize "placeholder" especially for input fields
        candidates = [
            " ".join(filter(None, [el.get("placeholder") or el.get("text"), el.get("label"), el.get("name")]))
            for el in elements
        ]

        query = intent.get("keyword") or intent.get("subject") or ""

        # --- Semantic / substring baseline ---
        if query:
            if self.minilm:
                # Primary high-confidence match
                matched_idxs = set(self.minilm.semantic_match(query, candidates, threshold=0.65))
                # Fallback borderline match (used if no high-confidence exists and we have other intent markers)
                if not matched_idxs and (intent.get("color") or intent.get("position")):
                    matched_idxs = set(self.minilm.semantic_match(query, candidates, threshold=0.55))

                base_filtered = [el for i, el in enumerate(elements) if i in matched_idxs]
            else:
                q_lower = query.lower()
                base_filtered = [
                    el
                    for el in elements
                    if q_lower in (el.get("placeholder") or "").lower()
                    or q_lower in (el.get("text") or "").lower()
                    or q_lower in (el.get("label") or "").lower()
                    or q_lower in (el.get("name") or "").lower()
                ]
            current = base_filtered
        else:
            current = elements

        # --- Color filter (non-destructive) ---
        if intent.get("color"):
            c = intent["color"].lower()
            color_filtered = [
                el for el in current if c in str(el.get("color", "")).lower() or c in str(el.get("style", "")).lower()
            ]
            if color_filtered:
                current = color_filtered
            # else: keep current (color filter would empty the list — skip it)

        # --- State filter (non-destructive) ---
        if intent.get("state"):
            s = intent["state"].lower()
            state_filtered = [
                el
                for el in current
                if s in str(el.get("state", "")).lower() or s in str(el.get("attributes", "")).lower()
            ]
            if state_filtered:
                current = state_filtered
            # else: keep current

        # --- Position filter (non-destructive) ---
        if intent.get("position"):
            p = intent["position"]  # e.g. "bottom-right"
            pos_filtered = []
            for el in current:
                loc = el.get("location")  # [y, x, w, h] normalized
                if not loc:
                    continue
                y, x = loc[0], loc[1]

                match = False
                if p == "top" and y < 0.33:
                    match = True
                elif p == "bottom" and y > 0.66:
                    match = True
                elif p == "left" and x < 0.33:
                    match = True
                elif p == "right" and x > 0.66:
                    match = True
                elif p == "top-left" and y < 0.4 and x < 0.4:
                    match = True
                elif p == "top-right" and y < 0.4 and x > 0.6:
                    match = True
                elif p == "bottom-left" and y > 0.6 and x < 0.4:
                    match = True
                elif p == "bottom-right" and y > 0.6 and x > 0.6:
                    match = True
                elif p == "center" and 0.2 < x < 0.8 and 0.2 < y < 0.8:
                    match = True

                if match:
                    pos_filtered.append(el)

            if pos_filtered:
                current = pos_filtered

        return current

    def normalize_subject(self, subject: str) -> str:
        """
        Normalizes a subject string for consistent historical lookup.

        Uses the internal intent parser's 'subject' extraction.
        """
        if not subject:
            return ""
        return self.parse_verify_intent(subject)["subject"].lower().strip()

    def resolve_historical_target(
        self, intent: Dict[str, Any], history_metadata: Dict[str, Any]
    ) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
        """
        Resolves a historical target and its associated perception elements
        by matching the current intent against history.

        Returns (target, before_elements).
        """
        subj = intent.get("subject") or intent.get("keyword") or ""
        if not subj:
            return None, []

        norm_subj = self.normalize_subject(subj)

        # 1. Direct match first (optimisation)
        history = history_metadata.get("history", {})
        if norm_subj in history:
            entry = history[norm_subj]
            return entry.get("target"), entry.get("elements", [])

        # 2. Semantic lookup if direct match fails
        # history["history"] is expected to be Dict[norm_subj, {target, elements}]
        if history and self.minilm:
            history_keys = list(history.keys())
            matched_idxs = self.minilm.semantic_match(norm_subj, history_keys, threshold=0.75)
            if matched_idxs:
                best_key = history_keys[matched_idxs[0]]
                self._logger.log_debug("parser", f"Resolved historical subject '{norm_subj}' -> '{best_key}'")
                entry = history[best_key]
                return entry.get("target"), entry.get("elements", [])

        return None, []

    def verify_negation(
        self,
        after_elements: List[Dict[str, Any]],
        intent_or_subject: Any,
        before_elements: Optional[List[Dict[str, Any]]] = None,
        target: Optional[Dict[str, Any]] = None,
        history_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Checks whether an element that was present before an action has
        disappeared afterwards.

        If a 'target' is provided, it attempts to find that specific element
        in 'after_elements' by comparing semantic attributes.

        If 'history_metadata' is provided, it attempts to resolve the target
        from history if not explicitly passed.
        """
        # 1. Normalize intent
        if isinstance(intent_or_subject, dict):
            intent = intent_or_subject
        else:
            intent = self.parse_verify_intent(str(intent_or_subject))

        if not intent.get("keyword") and not intent.get("subject"):
            return False

        # 2. Resolve target from history if needed
        if not target and history_metadata:
            target, before_elements = self.resolve_historical_target(intent, history_metadata)

        # 3. Identity-based check if target is provided
        if target:
            # Does the target actually match the intent?
            pos_intent = intent.copy()
            pos_intent["negated"] = False
            matches_intent = self.filter_elements_by_intent(pos_intent, [target])

            if matches_intent:
                target_text = (target.get("text") or "").lower().strip()
                target_label = (target.get("label") or "").lower().strip()
                target_name = (target.get("name") or "").lower().strip()
                target_placeholder = (target.get("placeholder") or "").lower().strip()
                self._logger.log_debug("verify_negation", f"target={target}")

                for el in after_elements:
                    if (
                        (el.get("text") or "").lower().strip() == target_text
                        and (el.get("label") or "").lower().strip() == target_label
                        and (el.get("name") or "").lower().strip() == target_name
                        and (el.get("placeholder") or "").lower().strip() == target_placeholder
                    ):
                        return False
                return True

        # 4. Fallback: Collective check
        pos_intent = intent.copy()
        pos_intent["negated"] = False
        after_matches = self.filter_elements_by_intent(pos_intent, after_elements)
        self._logger.log_debug("verify_negation", f"after_matches={after_matches}")
        return len(after_matches) == 0

    # ------------------------------------------------------------------
    # Internal parsing helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _parse_verify(self, clause: str) -> List[SemanticNode]:
        """Extracts a strict verification state."""
        clause = clause.strip()
        if not clause:
            return []

        boilerplate = r"^(verify( that)?|assert( that)?|ensure( that)?|make sure( that)?|check that)\b"
        cleaned = re.sub(boilerplate, "", clause, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^(the|a|an)\b", "", cleaned, flags=re.IGNORECASE).strip()

        return [SemanticNode(type="VERIFY", value=cleaned)]

    def _parse_clause(self, clause: str) -> List[SemanticNode]:
        """Parses an instruction sequence, breaking it by standard conjunctions."""
        nodes: List[SemanticNode] = []
        chunks = [
            c.strip()
            for c in self.SPLIT_PATTERN.split(clause)
            if c.strip() and c.strip().lower() not in ["and", "then", "after", "while", ","]
        ]

        current_target = "element"
        current_action = None

        for i, chunk in enumerate(chunks):
            is_verify = any(chunk.lower().startswith(v) for v in self.VERIFY_VERBS)
            if is_verify:
                nodes.extend(self._parse_verify(chunk))
                continue

            chunk_nodes, new_target, new_action = self._parse_atomic_action(chunk, current_target, current_action)

            # Distributive property: "Click A and B"
            # If this is an 'and' join, and we have an implicit action being used,
            # and the previous node was a DO with the same action, we might need to
            # re-insert the action for this target.
            # Actually _parse_atomic_action already adds the DO node if it uses implicit_action.

            nodes.extend(chunk_nodes)
            if new_target and new_target != "element":
                current_target = new_target
            if new_action:
                current_action = new_action

        return nodes

    def _parse_atomic_action(  # pylint: disable=too-many-branches,too-many-statements
        self, chunk: str, implicit_target: str, implicit_action: Optional[str] = None
    ) -> tuple:
        """Takes a single continuous phrase and extracts FIND and DO."""
        quotes: List[str] = []

        def _repl(match):
            quotes.append(match.group(0))
            return f"__QUOTE_{len(quotes)-1}__"

        chunk_protected = re.sub(r"(['\"])(.*?)\1", _repl, chunk)
        lower_chunk = chunk_protected.lower()

        action_verb = None
        action_type = None

        for ax_type, synonyms in self.ACTION_VERBS.items():
            for syn in sorted(synonyms, key=len, reverse=True):
                if re.search(r"\b" + re.escape(syn) + r"\b", lower_chunk):
                    if (
                        syn == "type"
                        and "clear" in lower_chunk
                        and lower_chunk.index("clear") < lower_chunk.index("type")
                    ):
                        continue
                    action_verb = syn
                    action_type = ax_type
                    break
            if action_verb:
                break

        if not action_verb and implicit_action:
            action_verb = implicit_action
            for ax_type, synonyms in self.ACTION_VERBS.items():
                if action_verb in synonyms:
                    action_type = ax_type
                    break

        if not action_verb:
            if lower_chunk.strip() == "submit":
                return (
                    [SemanticNode(type="FIND", value="element"), SemanticNode(type="DO", value="click Submit")],
                    "element",
                    "click",
                )
            elif "submit" in lower_chunk:
                return (
                    [
                        SemanticNode(type="FIND", value=chunk_protected.strip()),
                        SemanticNode(type="DO", value="click Submit"),
                    ],
                    chunk_protected.strip(),
                    "click",
                )

            restored = chunk_protected
            for i, q in enumerate(quotes):
                restored = restored.replace(f"__QUOTE_{i}__", q)
            # Re-apply quote protection to the restored string if it has quotes
            return [], restored.strip(), None

        payload = ""
        chunk_no_payload = chunk_protected

        if action_type in ["type", "enter"]:
            quote_match_type = re.search(r"__QUOTE_(\d+)__", chunk_no_payload)
            if quote_match_type:
                q_idx = int(quote_match_type.group(1))
                payload = quotes[q_idx]
                chunk_no_payload = chunk_no_payload.replace(f"__QUOTE_{q_idx}__", "")
            else:
                if re.search(rf"\b{action_verb}\b", lower_chunk):
                    implicit_match = re.search(
                        rf"\b{action_verb}\s+([a-zA-Z0-9_@.-]+)(?:\s+(in|into|on|to)\b|$)", lower_chunk
                    )
                else:
                    implicit_match = re.search(r"^([a-zA-Z0-9_@.-]+)(?:\s+(in|into|on|to)\b|$)", lower_chunk.strip())

                if implicit_match:
                    payload = implicit_match.group(1)
                    chunk_no_payload = re.sub(re.escape(payload), "", chunk_protected, count=1, flags=re.IGNORECASE)
                else:
                    implicit_match = re.search(r"(.*?)\s+([a-zA-Z0-9_@.-]+)$", lower_chunk)
                    if implicit_match:
                        payload = implicit_match.group(2)
                        chunk_no_payload = chunk_protected[: implicit_match.end(1)]
        elif action_type == "click":
            key_match = re.search(r"\b(enter|escape|esc|return|tab|space)\b", lower_chunk)
            if key_match and "key" not in lower_chunk:
                payload = key_match.group(1).capitalize()
                chunk_no_payload = re.sub(
                    r"\b" + re.escape(key_match.group(1)) + r"\b", "", chunk_protected, flags=re.IGNORECASE
                )
            elif key_match and "key" in lower_chunk:
                payload = key_match.group(1).capitalize() + " key"
                chunk_no_payload = re.sub(
                    r"\b" + re.escape(key_match.group(1)) + r"\s+key\b", "", chunk_protected, flags=re.IGNORECASE
                )

        target_str = re.sub(r"\b" + re.escape(action_verb) + r"\b", "", chunk_no_payload, flags=re.IGNORECASE, count=1)

        # Special logic for multi-step or complex relational actions
        if action_type == "drag" and (" onto " in target_str.lower() or " to " in target_str.lower()):
            drag_split = re.split(r"\b(?:onto|to)\b", target_str, flags=re.IGNORECASE, maxsplit=1)
            if len(drag_split) == 2:
                source = drag_split[0].strip()
                dest = drag_split[1].strip()
                # Clean noise from both
                for noise in ["the ", "a ", "an "]:
                    if source.lower().startswith(noise):
                        source = source[len(noise) :]
                    if dest.lower().startswith(noise):
                        dest = dest[len(noise) :]
                # Restore quotes
                for i, q in enumerate(quotes):
                    source = source.replace(f"__QUOTE_{i}__", q)
                    dest = dest.replace(f"__QUOTE_{i}__", q)
                return (
                    [
                        SemanticNode(type="FIND", value=source),
                        SemanticNode(type="DO", value="drag"),
                        SemanticNode(type="FIND", value=dest),
                        SemanticNode(type="DO", value="drop"),
                    ],
                    dest,
                    "drag",
                )

        if action_type == "select" and " from " in target_str.lower():
            select_split = re.split(r"\bfrom\b", target_str, flags=re.IGNORECASE, maxsplit=1)
            if len(select_split) == 2:
                payload_val = select_split[0].strip()
                container = select_split[1].strip()
                # Clean noise
                for noise in ["the ", "a ", "an "]:
                    if container.lower().startswith(noise):
                        container = container[len(noise) :]
                # Restore quotes
                for i, q in enumerate(quotes):
                    payload_val = payload_val.replace(f"__QUOTE_{i}__", q)
                    container = container.replace(f"__QUOTE_{i}__", q)
                return (
                    [
                        SemanticNode(type="FIND", value=container),
                        SemanticNode(type="DO", value=f"select {payload_val}"),
                    ],
                    container,
                    "select",
                )

        target_str = re.sub(r"\b(please navigate ahead and|i want you to)\b", "", target_str, flags=re.IGNORECASE)
        target_str = re.sub(r"\b(click on|click|type|enter|into)\b", "", target_str, flags=re.IGNORECASE)
        # Leading preposition stripping - MUST be anchored to start or after noise words
        target_str = target_str.strip()
        target_str = re.sub(r"^\b(into|onto|to|on|over|in|at|from)\b", "", target_str, flags=re.IGNORECASE)
        target_str = target_str.strip()
        target_str = re.sub(r"^(the|a|an)\b", "", target_str, flags=re.IGNORECASE)

        # Action-specific noise cleaning
        if action_type == "scroll":
            target_str = re.sub(r"\bto the\b", "", target_str, flags=re.IGNORECASE)
            target_str = re.sub(r"\bof the\b", "", target_str, flags=re.IGNORECASE)

        target_str = re.sub(r"\s+", " ", target_str).strip()
        target_str = re.sub(r"^[,\.]|[,\.]$", "", target_str).strip()

        for i, q in enumerate(quotes):
            target_str = target_str.replace(f"__QUOTE_{i}__", q)
            payload = payload.replace(f"__QUOTE_{i}__", q)

        nodes: List[SemanticNode] = []
        if target_str.lower() in ["it", "them", ""]:
            target_str = implicit_target if implicit_target else "element"

        if action_type == "wait":
            do_val = chunk.lower().strip()
            for i, q in enumerate(quotes):
                do_val = do_val.replace(f"__quote_{i}__", q.lower())

            if "until" in do_val:
                verify_text = re.sub(rf"^{action_verb}\s+until\s+", "", do_val).strip()
                return self._parse_verify(verify_text), "element", None
            return (
                [SemanticNode(type="FIND", value="element"), SemanticNode(type="DO", value=do_val)],
                "element",
                action_verb,
            )

        if action_type == "find":
            return [], target_str, None

        if target_str:
            nodes.append(SemanticNode(type="FIND", value=target_str))

        do_val = action_verb.lower()
        if payload:
            if action_type == "click":
                do_val = f"press {payload}"
            else:
                do_val += f" {payload}"
        elif action_type in ["type", "enter"]:
            do_val = f"type {target_str}"
            nodes = [SemanticNode(type="FIND", value="element")]

        nodes.append(SemanticNode(type="DO", value=do_val))

        return nodes, target_str, action_verb
