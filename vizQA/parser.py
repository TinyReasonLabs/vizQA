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

from vizQA.ranking import RankingEngine


class SemanticNode(NamedTuple):
    """Atomic semantic unit produced by the parser."""

    type: str
    value: str


# ---------------------------------------------------------------------------
# Intent classification helpers (used by parse_verify_intent)
# ---------------------------------------------------------------------------


class ParserVocabulary:
    """Central repository for UI grounding vocabulary."""

    ACTION_VERBS = {
        "click": ["click", "tap", "hit", "press"],
        "right-click": ["right-click", "context-click", "right click", "perform right-click", "context click"],
        "type": ["type", "enter", "input", "fill"],
        "hover": ["hover", "move to", "point"],
        "select": ["select", "choose", "pick"],
        "check": ["check", "tick"],
        "drag": ["drag"],
        "drop": ["drop"],
        "scroll": ["scroll"],
        "clear": ["clear", "empty"],
        "wait": ["wait", "pause", "sleep"],
        "find": ["find", "locate"],
    }

    VERIFY_VERBS = ["verify", "ensure", "assert", "check that", "make sure"]
    VERIFY_BOILERPLATE = [
        "should",
        "must",
        "appear",
        "visible",
        "shows",
        "exists",
        "present",
        "displayed",
        "open",
        "pause until",
    ]

    COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "gray", "grey"]
    STATES = ["disabled", "enabled", "checked", "unchecked", "visible", "invisible", "hidden", "displayed", "active"]
    POSITIONS = [
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
    # pylint: disable=line-too-long
    NEGATION_RE = re.compile(
        r"\b(not|no longer|should not|shouldn't|should not be|disappear(?:s|ed)?|gone|invisible|absent|done|finished|closed|close|removed|vanish(?:es|ed)?|gone)\b",
        re.IGNORECASE,
    )


class SemanticParser:
    """
    Advanced Rule-Based Engine (AST Parser) for dissecting UI testing instructions.
    Optionally enhanced with MiniLM embeddings for robust intent classification.
    """

    def __init__(
        self,
        minilm: Optional["MiniLM"] = None,
        use_advanced_ranking: bool = False,
        intent_threshold: float = 0.6,
        action_threshold: float = 0.52,
        semantic_match_threshold: float = 0.70,
    ):
        """
        Initialise the parser.
        """
        self.minilm = minilm
        self.use_advanced_ranking = use_advanced_ranking
        self.intent_threshold = intent_threshold
        self.action_threshold = action_threshold
        self.semantic_match_threshold = semantic_match_threshold
        self._ranking_engine = RankingEngine(minilm) if minilm else None
        self._logger = get_logger()

        # Cache length-sorted action synonyms for stable regex fallback
        self._action_synonyms_ordered = []
        for ax_type, synonyms in ParserVocabulary.ACTION_VERBS.items():
            for syn in synonyms:
                self._action_synonyms_ordered.append((syn, ax_type))
        self._action_synonyms_ordered.sort(key=lambda x: len(x[0]), reverse=True)

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

    # pylint: disable=protected-access
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

        if ParserVocabulary.NEGATION_RE.search(query):
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
            if neg_sim > self.intent_threshold or pos_sim > self.intent_threshold:
                if neg_sim > pos_sim:
                    is_negated = True
                else:
                    is_positive = True

        intent["negated"] = is_negated
        # Strip negation literals from subject if found
        if is_negated:
            subject = ParserVocabulary.NEGATION_RE.sub("", subject)

        # Strip positive boilerplate from subject
        subject = re.sub(r"\b(appears?|visible|shows?|exists?|present|displayed|opened?)\b", "", subject, flags=re.I)

        # 3. Color detection
        if self.minilm:
            _color_group = {c: self.minilm._intent_anchor_groups["color"] for c in ParserVocabulary.COLORS}
            # Use classify across individual words so multi-word queries work
            for word in query.lower().split():
                if word in ParserVocabulary.COLORS:
                    intent["color"] = word
                    subject = re.sub(rf"\b{re.escape(word)}\b", "", subject, flags=re.IGNORECASE)
                    break
            if not intent["color"]:
                # Semantic fallback: classify the full query
                detected = self.minilm.classify_anchor_group(
                    query, {"color": self.minilm._intent_anchor_groups["color"]}, threshold=self.intent_threshold
                )
                if detected == "color":
                    # Try to pin down which color via keyword list
                    for c in ParserVocabulary.COLORS:
                        if re.search(rf"\b{re.escape(c)}\b", query, re.IGNORECASE):
                            intent["color"] = c
                            subject = re.sub(rf"\b{re.escape(c)}\b", "", subject, flags=re.IGNORECASE)
                            break
        else:
            lower_q = query.lower()
            for c in ParserVocabulary.COLORS:
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
                query,
                {"position": self.minilm._intent_anchor_groups["position"]},
                threshold=self.intent_threshold - 0.05,
            )
            if detected == "position":
                # Semantic match found but no explicit word; leave position as None
                # (we don't want to hallucinate which position)
                pass

        # 5. State detection
        if self.minilm:
            for s in ParserVocabulary.STATES:
                if re.search(rf"\b{s}\b", query.lower()):
                    intent["state"] = s
                    subject = re.sub(rf"\b{s}\b", "", subject, flags=re.IGNORECASE)
                    break
            if not intent["state"]:
                detected = self.minilm.classify_anchor_group(
                    query, {"state": self.minilm._intent_anchor_groups["state"]}, threshold=self.intent_threshold - 0.02
                )
                if detected == "state":
                    # Semantic match but no literal — leave state as None to avoid false positives
                    pass
        else:
            for s in ParserVocabulary.STATES:
                if re.search(rf"\b{s}\b", query.lower()):
                    intent["state"] = s
                    subject = re.sub(rf"\b{s}\b", "", subject, flags=re.IGNORECASE)
                    break

        # 6. Clean subject
        # pylint: disable=line-too-long
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

        # --- Advanced Ranking Pipeline (Phase 1-3) ---
        if self.use_advanced_ranking and self._ranking_engine:
            query = intent.get("keyword") or intent.get("subject") or ""
            return self._ranking_engine.rank(query, intent, elements)

        # Build candidate strings for semantic / substring matching
        # prioritize "placeholder" especially for input fields
        candidates = [
            " ".join(filter(None, [el.get("placeholder") or el.get("text"), el.get("label"), el.get("name")]))
            for el in elements
        ]

        query = intent.get("keyword") or intent.get("subject") or ""

        # --- Semantic / substring baseline ---
        if query:
            kw = intent.get("keyword")

            if self.minilm:
                # Primary high-confidence match
                matched_idxs = set(
                    self.minilm.semantic_match(query, candidates, threshold=self.semantic_match_threshold)
                )
                # Fallback borderline match
                if not matched_idxs and (intent.get("color") or intent.get("position")):
                    matched_idxs = set(
                        self.minilm.semantic_match(query, candidates, threshold=self.semantic_match_threshold - 0.10)
                    )

                base_filtered = [el for i, el in enumerate(elements) if i in matched_idxs]

                if not base_filtered:
                    q_lower = query.lower()
                    kw_lower = kw.lower() if kw else None
                    base_filtered = []
                    for el in elements:
                        txt_fields = [
                            (el.get("placeholder") or "").lower(),
                            (el.get("text") or "").lower(),
                        ]
                        tech_fields = [
                            (el.get("label") or "").lower(),
                            (el.get("name") or "").lower(),
                        ]

                        if kw_lower:
                            # 1. Exact match on any field (highest priority)
                            if any(kw_lower == t for t in txt_fields + tech_fields):
                                base_filtered.append(el)
                                continue

                            # 2. Word boundary match on text fields
                            if any(re.search(rf"\b{re.escape(kw_lower)}\b", t) for t in txt_fields):
                                base_filtered.append(el)
                                continue

                            # 3. Technical field match (handle underscores/hyphens as boundaries)
                            # Only if NOT a quoted keyword (per user feedback "Unless quoted")
                            # Actually, if it's a quoted keyword, we still want it to match "logout" in "logout_btn"
                            # because that's often how users refer to technical elements.
                            # But we should be careful about partial matches.
                            for t in tech_fields:
                                normalized_tech = re.sub(r"[_-]", " ", t)
                                if re.search(rf"\b{re.escape(kw_lower)}\b", normalized_tech):
                                    base_filtered.append(el)
                                    break
                            continue

                        if any(q_lower in t for t in txt_fields + tech_fields):
                            base_filtered.append(el)
            else:
                q_lower = query.lower()
                kw_lower = kw.lower() if kw else None
                base_filtered = []
                for el in elements:
                    txt_fields = [
                        (el.get("placeholder") or "").lower(),
                        (el.get("text") or "").lower(),
                    ]
                    tech_fields = [
                        (el.get("label") or "").lower(),
                        (el.get("name") or "").lower(),
                    ]
                    if kw_lower:
                        if any(kw_lower == t or re.search(rf"\b{re.escape(kw_lower)}\b", t) for t in txt_fields):
                            base_filtered.append(el)
                            continue

                        found_tech = False
                        for t in tech_fields:
                            normalized_tech = re.sub(r"[_-]", " ", t)
                            if kw_lower == t or re.search(rf"\b{re.escape(kw_lower)}\b", normalized_tech):
                                base_filtered.append(el)
                                found_tech = True
                                break
                        if found_tech:
                            continue
                        continue

                    if any(q_lower in t for t in txt_fields + tech_fields):
                        base_filtered.append(el)

            # --- PRECISION PREFERENCE ---
            # If we have multiple matches and one (or more) are EXACT matches for our query/keyword,
            # we should prioritize those to avoid "Add" matching "Address Line 1" when "Add Item" exists.
            if len(base_filtered) > 1:
                target_q = (kw or query).lower()
                exact_matches = []
                for el in base_filtered:
                    txts = [(el.get(k) or "").lower() for k in ["text", "placeholder", "label", "name"]]
                    if any(t == target_q for t in txts):
                        exact_matches.append(el)

                if exact_matches:
                    base_filtered = exact_matches

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
            matched_idxs = self.minilm.semantic_match(
                norm_subj, history_keys, threshold=self.semantic_match_threshold + 0.05
            )
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
        _before_elements: Optional[List[Dict[str, Any]]] = None,
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
            target, _before_elements = self.resolve_historical_target(intent, history_metadata)

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

        boilerplate = (
            r"^(verify( that)?|assert( that)?|ensure( that)?|make sure( that)?|check that|pause until( the)?)\b"
        )
        cleaned = re.sub(boilerplate, "", clause, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^(the|a|an)\b", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^(loader disappears)\b", r"\1", cleaned, flags=re.IGNORECASE).strip()  # Specific cleanup

        # Split multiple verifications if "and" is present
        # e.g. "is red and aligned right" -> ["is red", "aligned right"]
        if re.search(r"\band\b", cleaned, re.I):
            # Split if "and" is followed by a state, position, or another subject/boilerplate
            # pylint: disable=line-too-long
            parts = re.split(
                r"\s+and\s+(?='|\"|the|a|an|it|is|should|must|at|in|on|color|state|visible|hidden|displayed|present|aligned|centered|top|bottom|left|right)",
                cleaned,
                flags=re.I,
            )
            if len(parts) > 1:
                return [SemanticNode(type="VERIFY", value=p.strip()) for p in parts if p.strip()]

        return [SemanticNode(type="VERIFY", value=cleaned)]

    def _parse_clause(self, clause: str) -> List[SemanticNode]:
        """Parses an instruction sequence, breaking it by standard conjunctions."""
        # Clean potential double-defined nodes or noise
        nodes: List[SemanticNode] = []

        # 1. Broad split on arrows
        parts = re.split(r"->|=>", clause)

        main_clauses = self._split_complex_sequence(parts[0])

        current_target = "element"
        current_action = None

        for i, chunk in enumerate(main_clauses):
            lower_chunk = chunk.lower()
            is_verify = (
                any(lower_chunk.startswith(v) for v in ParserVocabulary.VERIFY_VERBS)
                or any(v in lower_chunk for v in ParserVocabulary.VERIFY_BOILERPLATE)
                # Inheritance: if previous chunk was verify and current has no new action verb
                or (
                    i > 0
                    and nodes
                    and nodes[-1].type == "VERIFY"
                    and not any(
                        re.search(rf"\b{re.escape(syn)}\b", lower_chunk) for syn, _ in self._action_synonyms_ordered
                    )
                )
            )

            if is_verify:
                nodes.extend(self._parse_verify(chunk))
                # Set a generic verification flag for next chunks
                continue

            chunk_nodes, new_target, new_action = self._parse_atomic_action(chunk, current_target, current_action)

            # --- DISTRIBUTIVE TARGET HEURISTIC ---
            # "Click the submit and cancel buttons" -> "submit button", "cancel buttons"
            if i < len(main_clauses) - 1:
                next_chunk = main_clauses[i + 1].lower()
                # Check if next chunk ends with a known plural noun
                words_next = next_chunk.split()
                if words_next:
                    suffix = words_next[-1]
                    if suffix in ["buttons", "fields", "icons", "inputs", "links", "checkboxes", "labels"]:
                        singular = suffix[:-1]
                        for j, node in enumerate(chunk_nodes):
                            if node.type == "FIND" and singular not in node.value.lower() and node.value != "element":
                                # Update the node value directly (SemanticNode is a namedtuple, so we recreate)
                                chunk_nodes[j] = SemanticNode(type="FIND", value=f"{node.value} {singular}")

            nodes.extend(chunk_nodes)

            if new_target and new_target != "element":
                current_target = new_target
            if new_action:
                current_action = new_action

        # 2. Join the tail (after ->)
        for verify_part in parts[1:]:
            nodes.extend(self._parse_verify(verify_part))

        return nodes

    def _split_complex_sequence(self, text: str) -> List[str]:
        """Splits a sequence while protecting atomic phrases."""
        # 0. Protect "press and hold" and similar phrases
        protected_text = re.sub(r"\b(press|select|click|tap)\s+and\s+hold\b", r"\1_AND_HOLD", text, flags=re.I)

        # 1. Broad split on major conjunctions/punctuation
        splits = re.split(r"\bthen\b|\bafter\b|\bwhile\b|,|->", protected_text, flags=re.I)

        final_chunks = []
        for s in splits:
            s_clean = s.strip()
            if not s_clean:
                continue

            # 2. Split 'and' if followed by a verb, article, quoted string, or specific descriptors
            # Prioritize splitting when different objects or subjects are being described.
            # We use a narrower list to avoid splitting compound attributes (like red and aligned)
            # pylint: disable=line-too-long
            sub_splits = re.split(
                r"\band\b\s+(?='|\"|the|a|an|click|tap|type|enter|input|fill|select|press|hover|into|onto|last|first|clear|scroll|right|left|context|submit|cancel|delete|save|sign|log|password|username|email)",
                s_clean,
                flags=re.I,
            )

            for sub in sub_splits:
                res = sub.strip().replace("_AND_HOLD", " and hold")
                if res:
                    final_chunks.append(res)

        return final_chunks

    def _parse_atomic_action(  # pylint: disable=too-many-branches,too-many-statements, too-many-return-statements
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

        # 0. High-priority explicit verbs (Clear/Empty)
        first_word = lower_chunk.split()[0] if lower_chunk.split() else ""
        if first_word in ["clear", "empty"]:
            action_verb = first_word
            action_type = "clear"

        # 1. Semantic Action Classification (Prioritize MiniLM)
        if not action_type and self.minilm:
            # Restore quotes for semantic matching to get full context
            semantic_query = chunk
            # pylint: disable=protected-access
            detected = self.minilm.classify_anchor_group(
                semantic_query, groups=self.minilm._action_groups, threshold=self.action_threshold
            )
            if detected:
                action_type = detected
                self._logger.log_debug(0, f"[Parser] Semantic classified '{semantic_query}' as '{action_type}'")
                # We still need a concrete verb to strip from the string
                # Find the longest synonym of this type that exists in the chunk
                synonyms = ParserVocabulary.ACTION_VERBS.get(action_type, [])
                for syn in sorted(synonyms, key=len, reverse=True):
                    # Avoid matching "input" as verb if it's "the input"
                    pattern = rf"\b{re.escape(syn)}\b"
                    if syn == "input":
                        pattern = r"(?<!the\s)\binput\b"

                    if re.search(pattern, lower_chunk):
                        action_verb = syn
                        break
                if not action_verb:
                    # Fallback to the first synonym if no literal match (semantic match)
                    action_verb = synonyms[0] if synonyms else action_type

                # --- ACTION REFINEMENT ---
                # Check for explicit synonyms in the text to override semantic mismatch
                # specifically for right-click which often gets matched as click
                refined = False
                for syn_type, syns in ParserVocabulary.ACTION_VERBS.items():
                    if syn_type == action_type:
                        continue
                    for syn in syns:
                        if re.search(rf"\b{re.escape(syn)}\b", lower_chunk):
                            # pylint: disable=line-too-long
                            self._logger.log_debug(
                                0,
                                f"[Parser] Refined action from '{action_type}' to '{syn_type}' because of literal '{syn}' in chunk",
                            )
                            action_type = syn_type
                            action_verb = syn
                            refined = True
                            break
                    if refined:
                        break

        # 2. Rule-based fallback (Stable matching via length-sorted synonyms)
        if not action_type:
            for syn, ax_type in self._action_synonyms_ordered:
                # Avoid matching "input" as verb if it's "the input"
                pattern = rf"\b{re.escape(syn)}\b"
                if syn == "input":
                    pattern = r"(?<!the\s)\binput\b"

                if re.search(pattern, lower_chunk):
                    # Check for "type" vs "clear" override
                    if (
                        syn == "type"
                        and "clear" in lower_chunk
                        and lower_chunk.index("clear") < lower_chunk.index("type")
                    ):
                        continue
                    action_verb = syn
                    action_type = ax_type
                    break

        if not action_verb and implicit_action:
            # Only use implicit action if the chunk doesn't look like it already contains a target
            # e.g. if we have "Click A and B", chunk "B" uses implicit "click".
            # Avoid duplicate actions if the chunk already seems to have one.
            words = lower_chunk.split()
            # Heuristic: inherit action if chunk is short OR contains prepositions like "into" or "to"
            has_verb = any(w in [s for s, _ in self._action_synonyms_ordered] for w in words)
            if not has_verb:
                # If it's a long chunk, only inherit if it has a preposition that implies a target
                if len(words) <= 3 or any(w in ["into", "to", "on", "in"] for w in words):
                    action_verb = implicit_action
                    action_type = next(
                        (ax for ax, syns in ParserVocabulary.ACTION_VERBS.items() if action_verb in syns), None
                    )

        if not action_verb:
            if lower_chunk.strip() == "submit":
                return (
                    [SemanticNode(type="FIND", value="element"), SemanticNode(type="DO", value="click Submit")],
                    "element",
                    "click",
                )
            if "submit" in lower_chunk:
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
        target_str = re.sub(r"\s+", " ", target_str).strip()
        target_str = re.sub(r"^\b(into|onto|to|on|over|in|at|from)\b", "", target_str, flags=re.IGNORECASE)
        target_str = target_str.strip()
        target_str = re.sub(r"^(the|a|an)\b", "", target_str, flags=re.IGNORECASE)
        target_str = target_str.strip()

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
            if target_str.lower() in ["it", "them", ""]:
                nodes.append(SemanticNode(type="FIND", value=implicit_target if implicit_target else "element"))
            else:
                nodes.append(SemanticNode(type="FIND", value=target_str))

        # Use canonical type for test compatibility, fallback to matched literal
        do_name = action_type if action_type else (action_verb or "interact")
        do_val = do_name.lower()

        if payload:
            if action_type == "click":
                do_val = f"press {payload}"
            else:
                do_val = f"{do_val} {payload}"
                if action_type in ["type", "enter"]:
                    return (
                        [
                            SemanticNode(
                                type="FIND", value=target_str if target_str and target_str != "element" else "element"
                            ),
                            SemanticNode(type="DO", value=f"{do_val}"),
                        ],
                        target_str,
                        action_verb,
                    )
        elif action_type in ["type", "enter"]:
            # Special legacy multi-node return for type/enter when no payload extracted
            return (
                [SemanticNode(type="FIND", value="element"), SemanticNode(type="DO", value=f"{do_val} {target_str}")],
                target_str,
                action_verb,
            )

        nodes.append(SemanticNode(type="DO", value=do_val))

        return nodes, target_str, action_verb
