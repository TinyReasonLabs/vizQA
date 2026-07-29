"""
Semantic parser for UI testing instructions.

Provides rule-based AST parsing of natural language instructions into atomic
FIND / DO / VERIFY nodes, with optional MiniLM-powered intent classification
for verification queries.
"""

import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from vizQA.app.config import CONFIG
from vizQA.app.logger import get_logger
from vizQA.reasoning.clause_splitting import split_verify_conjunctions
from vizQA.reasoning.intent import Intent, IntentAttributes
from vizQA.reasoning.language import LanguagePack, alternation_pattern, default_language_pack, match_prefixed_payload
from vizQA.reasoning.model_protocols import SemanticModel
from vizQA.reasoning.query_semantics import lexical_term_score
from vizQA.reasoning.ranking import RankingEngine


class SemanticNode(NamedTuple):
    """Atomic semantic unit produced by the parser."""

    type: str
    value: str


# pylint: disable=too-many-instance-attributes
class SemanticParser:
    """
    Advanced Rule-Based Engine (AST Parser) for dissecting UI testing instructions.
    Optionally enhanced with MiniLM embeddings for robust intent classification.
    """

    def __init__(
        self,
        logger: Optional[Any] = None,
        language_pack: Optional[LanguagePack] = None,
        semantic_provider: Optional[SemanticModel] = None,
    ):
        """Initialise the parser."""
        self.language_pack = language_pack or default_language_pack()
        self.semantic_provider = semantic_provider
        self.config = CONFIG
        self._ranking_engine = (
            RankingEngine(self.semantic_provider, language_pack=self.language_pack) if self.semantic_provider else None
        )
        self._logger = logger or get_logger()
        self._negation_re = self.language_pack.negation_regex
        self._verify_subject_noise_re = re.compile(
            rf"\b(?:{alternation_pattern(self.language_pack.verify_subject_noise)})\b",
            re.IGNORECASE,
        )
        self._verify_trigger_re = re.compile(
            rf"\b(?:{alternation_pattern(self.language_pack.verify_trigger_terms)})\b",
            re.IGNORECASE,
        )
        self._verify_query_prefix_re = self._prefix_pattern(self.language_pack.verify_query_prefixes)
        self._verify_prefix_re = self._prefix_pattern(self.language_pack.verify_prefixes)
        self._article_prefix_re = self._prefix_pattern(self.language_pack.articles)
        self._leading_preposition_prefix_re = self._prefix_pattern(self.language_pack.leading_prepositions)
        self._target_cleanup_re = re.compile(
            rf"\b(?:{alternation_pattern(self.language_pack.target_cleanup_phrases)})\b",
            re.IGNORECASE,
        )
        coordination_pattern = alternation_pattern(self.language_pack.coordination_terms)
        hold_modifier_pattern = alternation_pattern(self.language_pack.hold_modifier_terms)
        self._hold_phrase_re = re.compile(
            rf"\b({alternation_pattern(self.language_pack.hold_action_verbs)})\s+"
            rf"(?:{coordination_pattern})\s+(?:{hold_modifier_pattern})\b",
            re.IGNORECASE,
        )
        followers_pattern = alternation_pattern(self.language_pack.sequence_split_followers)
        self._coordination_split_re = re.compile(
            rf"\b(?:{coordination_pattern})\b\s+(?='|\"|{followers_pattern})",
            re.IGNORECASE,
        )
        self._wait_condition_prefix_re = re.compile(
            rf"^(?:{alternation_pattern(self.language_pack.wait_verbs)})\s+"
            rf"(?:{alternation_pattern(self.language_pack.wait_condition_terms)})\s+",
            re.IGNORECASE,
        )
        self._position_patterns = self._build_position_patterns()
        self._noun_action_guard_patterns = {
            synonym: [re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE) for phrase in phrases]
            for synonym, phrases in self.language_pack.noun_action_guards.items()
        }
        self._action_anchor_groups = {
            action_name: spec.anchors for action_name, spec in self.language_pack.actions.items() if spec.anchors
        }

        # Cache length-sorted action synonyms for stable regex fallback
        self._action_synonyms_ordered = []
        for ax_type, spec in self.language_pack.actions.items():
            for syn in spec.synonyms:
                self._action_synonyms_ordered.append((syn, ax_type))
        self._action_synonyms_ordered.sort(key=lambda x: len(x[0]), reverse=True)

    @staticmethod
    def _prefix_pattern(prefixes: List[str]) -> re.Pattern[str]:
        return re.compile(rf"^(?:{alternation_pattern(prefixes)})\b\s*", re.IGNORECASE)

    def _build_position_patterns(self) -> List[Tuple[re.Pattern[str], str]]:
        """Build regex patterns for canonical positions and aliases."""
        patterns: List[Tuple[re.Pattern[str], str]] = []
        seen: set[str] = set()

        def _canonicalize(value: str) -> str:
            return value.strip().lower().replace(" ", "-")

        def _pattern_for(term: str) -> re.Pattern[str]:
            escaped = re.escape(term.strip()).replace(r"\ ", r"[-\s]+")
            return re.compile(rf"\b{escaped}\b", re.IGNORECASE)

        position_mappings = list(self.language_pack.position_terms.items())
        position_mappings.extend((canonical, [term]) for term, canonical in self.language_pack.position_aliases.items())
        localized_positions = [(canonical, term) for canonical, terms in position_mappings for term in terms]
        for canonical, term in sorted(localized_positions, key=lambda item: len(item[1]), reverse=True):
            normalized = term.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            patterns.append((_pattern_for(normalized), _canonicalize(canonical)))
        return patterns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, instruction: str) -> List[SemanticNode]:
        """Parses a full natural language instruction into a list of atomic SemanticNodes."""
        direct_nodes = self.parse_direct_action(instruction)
        if direct_nodes is not None:
            return direct_nodes

        nodes: List[SemanticNode] = []

        # Broad splits on explicit flow arrows which almost always mean VERIFY comes next
        parts = re.split(r"->|=>", instruction)
        nodes.extend(self._parse_clause(parts[0]))

        for verification_part in parts[1:]:
            nodes.extend(self._parse_verify(verification_part))

        return nodes

    def parse_direct_action(self, instruction: str) -> Optional[List[SemanticNode]]:
        """Parse explicit direct commands that do not require visual target grounding."""
        payload = match_prefixed_payload(instruction, self.language_pack.keypress_prefixes)
        if payload is not None:
            return [SemanticNode("DO", f"press-key {payload}".strip())]
        return None

    def parse_verification(self, assertion: str) -> List[SemanticNode]:
        """Parse an expectation as verification without injecting language text."""
        return self._parse_verify(assertion)

    def parse_verify_intent(self, query: str) -> Intent:
        """
        Parses a verification query to extract specific intents.

        Returns a dict with keys: ``keyword``, ``color``, ``position``,
        ``state``, ``negated``, ``subject``.

        When a MiniLM instance is provided, colors / states / positions are
        detected via cosine similarity (robust to synonyms).  Negation is
        detected via a fast regex pass **and** a semantic slow-path so that
        paraphrases like "the overlay should vanish" are also caught.
        """
        query = self.normalize_verify_query(query)
        intent = Intent(subject=query)
        subject = query

        # 1. Extract quoted keyword
        quote_match = re.search(r"(['\"])(.*?)\1", query)
        if quote_match:
            intent = Intent(
                keyword=quote_match.group(2),
                subject=intent.subject,
                negated=intent.negated,
                attributes=intent.attributes,
                source=intent.source,
            )
            subject = subject.replace(quote_match.group(0), "")

        # 2. Extract specific attributes
        negated, subject = self._extract_negation(query, subject)
        color, subject = self._extract_color(query, subject)
        position, subject = self._extract_position(query, subject)
        state, subject = self._extract_state(query, subject)

        # 3. Clean up final subject
        subject = self._verify_subject_noise_re.sub("", subject)
        return Intent(
            keyword=intent.keyword,
            subject=re.sub(r"\s+", " ", subject).strip(),
            negated=negated,
            attributes=IntentAttributes(color=color, position=position, state=state),
            source="rule",
        )

    def has_specific_target_subject(self, query_or_intent: Any) -> bool:
        """Returns whether the parsed target is semantically more specific than generic page scope."""
        intent = (
            query_or_intent if isinstance(query_or_intent, Intent) else self.parse_verify_intent(str(query_or_intent))
        )
        target_text = intent.query_text.strip()
        if not target_text:
            return False
        if not self.semantic_provider:
            raise RuntimeError("SemanticParser.has_specific_target_subject requires a MiniLM-backed parser.")

        generic_matches = self.semantic_provider.semantic_match(
            target_text,
            self.language_pack.generic_scope_terms,
            threshold=self.config.semantic_match_threshold,
        )
        return not bool(generic_matches)

    def normalize_verify_query(self, query: str) -> str:
        """Strips generic wait/scroll prefixes before intent parsing."""
        normalized = query.strip()
        normalized = self._verify_query_prefix_re.sub("", normalized)
        return normalized.strip()

    def _extract_negation(self, query: str, subject: str) -> Tuple[bool, str]:
        """Extracts negation intent."""
        is_negated = bool(self._negation_re.search(query))
        if not is_negated and self.semantic_provider:
            is_negated = self.semantic_provider.is_negation(query)

        if is_negated:
            subject = self._negation_re.sub("", subject)
        return is_negated, subject

    def _extract_color(self, query: str, subject: str) -> Tuple[Optional[str], str]:
        """Extracts color intent."""
        for word in query.lower().split():
            if word in self.language_pack.colors:
                return word, re.sub(rf"\b{re.escape(word)}\b", "", subject, flags=re.I)

        if self.semantic_provider:
            color = self.semantic_provider.classify_anchor_group(query, threshold=self.config.intent_threshold)
            if color:  # It returned a group name like 'color' or specific match
                for c in self.language_pack.colors:
                    if re.search(rf"\b{re.escape(c)}\b", query, re.I):
                        return c, re.sub(rf"\b{re.escape(c)}\b", "", subject, flags=re.I)
        return None, subject

    def _extract_position(self, query: str, subject: str) -> Tuple[Optional[str], str]:
        """Extracts position intent."""
        for pattern, canonical in self._position_patterns:
            if pattern.search(query):
                return canonical, pattern.sub("", subject, count=1).strip()
        return None, subject

    def _extract_state(self, query: str, subject: str) -> Tuple[Optional[str], str]:
        """Extracts state intent."""
        for s in self.language_pack.states:
            if re.search(rf"\b{re.escape(s)}\b", query.lower()):
                return s, re.sub(rf"\b{re.escape(s)}\b", "", subject, flags=re.I)
        return None, subject

    def filter_elements_by_intent(self, intent: Intent, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filters a list of perception elements based on a parsed intent dict."""
        if not elements:
            return []

        if self.config.use_advanced_ranking and self._ranking_engine:
            query = intent.query_text
            return self._ranking_engine.rank(query, intent, elements)

        query = intent.query_text
        current = self._filter_by_query(query, intent, elements) if query else elements

        # Non-destructive attribute filters
        for attr in ["color", "state"]:
            val = getattr(intent, attr)
            if val:
                val = val.lower()
                filtered = [
                    el
                    for el in current
                    if val in str(el.get(attr, "")).lower()
                    or val in str(el.get("style" if attr == "color" else "attributes", "")).lower()
                ]
                if filtered:
                    current = filtered

        if intent.position:
            pos_filtered = self._filter_by_position(intent.position, current)
            if pos_filtered:
                current = pos_filtered

        return current

    def filter_target_candidates(self, intent: Intent, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter candidates for direct target grounding with stricter matching."""
        if not candidates:
            return []

        query = intent.query_text.strip()
        if not query:
            return []

        phrase_matches = [
            candidate for candidate in candidates if self._phrase_match(query, self._candidate_match_text(candidate))
        ]
        if phrase_matches:
            return self._prioritize_exact_matches(query, intent.keyword, phrase_matches)

        strict_intent = intent.with_threshold(min(0.95, self.config.semantic_match_threshold + 0.05))
        filtered = self.filter_elements_by_intent(strict_intent, candidates)
        if filtered:
            return filtered

        return self._overlap_grounding_fallback(query, candidates)

    def _filter_by_query(self, query: str, intent: Intent, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Core semantic and substring filtering logic."""
        candidates = [
            " ".join(filter(None, [el.get("placeholder") or el.get("text"), el.get("label"), el.get("name")]))
            for el in elements
        ]

        if self.semantic_provider:
            matched_idxs = set(
                self.semantic_provider.semantic_match(query, candidates, threshold=self.config.semantic_match_threshold)
            )
            if not matched_idxs and (intent.color or intent.position):
                matched_idxs = set(
                    self.semantic_provider.semantic_match(
                        query, candidates, threshold=self.config.semantic_match_threshold - 0.10
                    )
                )
            base_filtered = [el for i, el in enumerate(elements) if i in matched_idxs]
        else:
            base_filtered = []

        if not base_filtered:
            base_filtered = self._substring_fallback(query, intent.keyword, elements)

        return self._prioritize_exact_matches(query, intent.keyword, base_filtered)

    def _substring_fallback(
        self, query: str, kw: Optional[str], elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fallback substring matching when semantic search fails or is disabled.

        :param query: The query or subject string.
        :param kw: Optional strict keyword.
        :param elements: List of perception elements.
        :return: Filtered list of elements.
        """
        q_low, kw_low = query.lower(), kw.lower() if kw else None
        results = []
        for el in elements:
            txts = [str(el.get(k) or "").lower() for k in ["placeholder", "text", "label", "name"]]
            if kw_low:
                # Boundary check that respects snake_case/camelCase
                # We treat underscore or transition to uppercase (or non-alpha) as a boundary.
                # Simplest: check start/end or non-alpha around it.
                matched = False
                for t in txts:
                    if kw_low == t:
                        matched = True
                        break
                    # Search with logic that allows underscore/non-word as boundary
                    pattern = rf"(?:^|[^a-zA-Z0-9]){re.escape(kw_low)}(?:$|[^a-zA-Z0-9])"
                    if re.search(pattern, t):
                        matched = True
                        break
                if matched:
                    results.append(el)
            elif any(q_low in t for t in txts):
                results.append(el)
        return results

    def _prioritize_exact_matches(
        self, query: str, kw: Optional[str], elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Narrows results to exact matches if any exist."""
        if len(elements) <= 1:
            return elements
        target = (kw or query).lower()
        exact = [
            el
            for el in elements
            if any(str(el.get(k) or "").lower() == target for k in ["text", "placeholder", "label", "name"])
        ]
        return exact if exact else elements

    def _overlap_grounding_fallback(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback target grounding using generic lexical overlap, not stop-word pruning."""
        terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) >= 3]
        if len(terms) < 2:
            return []

        ranked_candidates = []
        for index, candidate in enumerate(candidates):
            candidate_text = self._candidate_match_text(candidate)
            if not candidate_text:
                continue

            matched_terms = [
                (term_index, len(term))
                for term_index, term in enumerate(terms)
                if lexical_term_score(term, candidate_text) > 0.0
            ]
            if not matched_terms:
                continue

            coverage = len(matched_terms) / len(terms)
            if coverage < 0.5:
                continue

            total_length = sum(term_length for _, term_length in matched_terms)
            first_match_index = min(term_index for term_index, _ in matched_terms)
            ranked_candidates.append((coverage, total_length, -first_match_index, -index, candidate))

        if not ranked_candidates:
            return []

        ranked_candidates.sort(reverse=True)
        best_signature = ranked_candidates[0][:4]
        return [candidate for *signature, candidate in ranked_candidates if tuple(signature) == best_signature]

    def _filter_by_position(self, p: str, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filters elements by spatial position."""
        filtered = []
        for el in elements:
            loc = el.get("location")  # [y, x, w, h]
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
                filtered.append(el)
        return filtered

    @staticmethod
    def _candidate_match_text(candidate: Dict[str, Any]) -> str:
        """Build a searchable string for one perception candidate."""
        parts = [candidate.get(key) for key in ("placeholder", "text", "label", "name")]
        return " ".join(str(part).strip() for part in parts if part)

    @staticmethod
    def _phrase_match(query: str, candidate_text: str) -> bool:
        """Return whether the candidate contains the requested phrase."""
        q = re.sub(r"['\"]", "", query.lower()).strip()
        c = re.sub(r"['\"]", "", candidate_text.lower()).strip()
        return bool(q and c and q in c)

    def normalize_subject(self, subject: str) -> str:
        """
        Normalizes a subject string for consistent historical lookup.

        Uses the internal intent parser's 'subject' extraction.
        """
        if not subject:
            return ""
        intent = self.parse_verify_intent(subject)
        return intent.query_text.lower().strip()

    def resolve_historical_target(
        self, intent: Intent, history_metadata: Dict[str, Any]
    ) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
        """
        Resolves a historical target and its associated perception elements
        by matching the current intent against history.

        Returns (target, before_elements).
        """
        subj = intent.normalized_subject
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
        if history and self.semantic_provider:
            history_keys = list(history.keys())
            matched_idxs = self.semantic_provider.semantic_match(
                norm_subj, history_keys, threshold=self.config.semantic_match_threshold + 0.05
            )
            if matched_idxs:
                best_key = history_keys[matched_idxs[0]]
                self._logger.log_debug("parser", f"Resolved historical subject '{norm_subj}' -> '{best_key}'")
                entry = history[best_key]
                return entry.get("target"), entry.get("elements", [])

        return None, []

    # pylint: disable=too-many-arguments, too-many-positional-arguments, unused-argument
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

        :param after_elements: Perception elements from the AFTER state.
        :param intent_or_subject: Either a parsed intent dict or a query string.
        :param before_elements: Optional perception elements from the BEFORE state.
        :param target: Specific element to track if already identified.
        :param history_metadata: Optional history data for subject resolution.
        :return: True if the element is effectively 'gone'.
        """
        # 1. Normalize intent
        if isinstance(intent_or_subject, Intent):
            intent = intent_or_subject
        else:
            intent = self.parse_verify_intent(str(intent_or_subject))

        if not intent.keyword and not intent.subject:
            return False

        # 2. Resolve target from history if needed
        if not target and history_metadata:
            target, before_elements = self.resolve_historical_target(intent, history_metadata)

        # 3. Identity-based check if target is provided
        if target:
            # Does the target actually match the intent?
            pos_intent = intent.with_negated(False)
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
        pos_intent = intent.with_negated(False)
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

        cleaned = self._verify_prefix_re.sub("", clause).strip()
        cleaned = self._article_prefix_re.sub("", cleaned).strip()

        quotes: List[str] = []

        def _protect_quote(match: re.Match[str]) -> str:
            quotes.append(match.group(0))
            return f"__QUOTE_{len(quotes)-1}__"

        protected = re.sub(r"(['\"])(.*?)\1", _protect_quote, cleaned)
        parts = split_verify_conjunctions(protected, self.language_pack)
        if len(parts) > 1:
            restored_parts = []
            for part in parts:
                restored = part
                for i, quote in enumerate(quotes):
                    restored = restored.replace(f"__QUOTE_{i}__", quote)
                restored = self._article_prefix_re.sub("", restored).strip()
                restored_parts.append(SemanticNode(type="VERIFY", value=restored))
            return restored_parts

        return [SemanticNode(type="VERIFY", value=cleaned)]

    def _parse_clause(self, clause: str) -> List[SemanticNode]:
        """Parses an instruction sequence, breaking it by standard conjunctions."""
        nodes: List[SemanticNode] = []
        parts = re.split(r"->|=>", clause)
        main_clauses = self._split_complex_sequence(parts[0])

        current_target = "element"
        current_action = None

        for i, chunk in enumerate(main_clauses):
            res_nodes, new_target, new_action = self._process_sequence_chunk(
                chunk, i, main_clauses, nodes, current_target, current_action
            )
            nodes.extend(res_nodes)
            if new_target and new_target != "element":
                current_target = new_target
            if new_action:
                current_action = new_action

        for verify_part in parts[1:]:
            nodes.extend(self._parse_verify(verify_part))

        return nodes

    # pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments
    def _process_sequence_chunk(
        self,
        chunk: str,
        idx: int,
        clauses: List[str],
        nodes: List[SemanticNode],
        implicit_target: str,
        implicit_action: Optional[str],
    ) -> tuple[List[SemanticNode], str, Optional[str]]:
        """Processes a single chunk in a sequence, handling inheritance and distributive targets."""
        lower_chunk = chunk.lower()
        is_verify = (
            any(lower_chunk.startswith(v) for v in self.language_pack.verify_verbs)
            or any(v in lower_chunk for v in self.language_pack.verify_boilerplate)
            or bool(self._verify_trigger_re.search(lower_chunk))
            or (
                idx > 0
                and nodes
                and nodes[-1].type == "VERIFY"
                and not any(
                    re.search(rf"\b{re.escape(syn)}\b", lower_chunk) for syn, _ in self._action_synonyms_ordered
                )
            )
        )

        if is_verify:
            return self._parse_verify(chunk), "element", None

        chunk_nodes, new_target, new_action = self._parse_atomic_action(chunk, implicit_target, implicit_action)

        # Distributive target heuristic
        if idx < len(clauses) - 1:
            next_chunk = clauses[idx + 1].lower()
            words_next = next_chunk.split()
            if words_next:
                suffix = words_next[-1]
                plural_nouns = self.language_pack.distributive_plural_nouns
                if suffix in plural_nouns:
                    singular = suffix[:-1]
                    for j, node in enumerate(chunk_nodes):
                        if node.type == "FIND" and singular not in node.value.lower() and node.value != "element":
                            chunk_nodes[j] = SemanticNode(type="FIND", value=f"{node.value} {singular}")

        return chunk_nodes, new_target or implicit_target, new_action or implicit_action

    def _split_complex_sequence(self, text: str) -> List[str]:
        """Splits a sequence while protecting atomic phrases."""
        # 0. Protect hold gestures and similar atomic phrases
        protected_hold_suffixes: List[str] = []

        def _protect_hold(match: re.Match[str]) -> str:
            protected_hold_suffixes.append(match.group(0)[len(match.group(1)) :])
            return f"{match.group(1)}__HOLD_{len(protected_hold_suffixes) - 1}__"

        protected_text = self._hold_phrase_re.sub(_protect_hold, text)

        # 1. Broad split on major conjunctions/punctuation
        clause_splitters = alternation_pattern(self.language_pack.clause_splitters)
        punctuation = "|".join(re.escape(term) for term in self.language_pack.coordination_punctuation if term)
        splitter_pattern = rf"\b(?:{clause_splitters})\b|->|=>"
        if punctuation:
            splitter_pattern = f"{splitter_pattern}|{punctuation}"
        splits = re.split(splitter_pattern, protected_text, flags=re.I)

        final_chunks = []
        for s in splits:
            s_clean = s.strip()
            if not s_clean:
                continue

            # 2. Split 'and' if followed by a verb, article, quoted string, or specific descriptors
            # Prioritize splitting when different objects or subjects are being described.
            # We use a narrower list to avoid splitting compound attributes (like red and aligned)
            # pylint: disable=line-too-long
            sub_splits = self._coordination_split_re.split(s_clean)

            for sub in sub_splits:
                res = sub.strip()
                for index, suffix in enumerate(protected_hold_suffixes):
                    res = res.replace(f"__HOLD_{index}__", suffix)
                if res:
                    final_chunks.append(res)

        return final_chunks

    def _parse_atomic_action(self, chunk: str, implicit_target: str, implicit_action: Optional[str] = None) -> tuple:
        """Takes a single continuous phrase and extracts FIND and DO."""
        quotes: List[str] = []

        def _repl(m):
            quotes.append(m.group(0))
            return f"__QUOTE_{len(quotes)-1}__"

        protected = re.sub(r"(['\"])(.*?)\1", _repl, chunk)
        action_verb, action_type = self._get_action_info(protected, chunk)

        if not action_verb and implicit_action:
            action_verb, action_type = self._inherit_action(protected, implicit_action)

        if not action_verb:
            return self._handle_no_verb(protected, quotes)

        # Extraction phase
        payload, chunk_no_pload = self._extract_payload(protected, action_verb, action_type, quotes)
        target_str = self._clean_target_string(chunk_no_pload, action_verb, action_type)

        # Relational phase
        relational = self._handle_relational_actions(target_str, action_type, quotes)
        if relational:
            return relational

        # Final assembly
        final_target = self._resolve_final_target(target_str, implicit_target)
        action_info = (final_target, action_type, action_verb)
        return self._assemble_nodes(action_info, payload, quotes, chunk)

    def _get_action_info(self, protected: str, original: str) -> Tuple[Optional[str], Optional[str]]:
        """Determines action verb and type."""
        lower = protected.lower()
        first = lower.split()[0] if lower.split() else ""
        if first in self.language_pack.actions.get("clear").synonyms:
            return first, "clear"

        if self.semantic_provider:
            dtype = self.semantic_provider.classify_anchor_group(
                original, groups=self._action_anchor_groups, threshold=self.config.action_threshold
            )
            if dtype:
                synonym_values = self.language_pack.actions[dtype].synonyms
                for syn in sorted(synonym_values, key=len, reverse=True):
                    pattern = rf"\b{re.escape(syn)}\b"
                    if re.search(pattern, lower) and not self._matches_noun_action_guard(syn, lower):
                        return syn, dtype
                first_word = re.findall(r"[a-zA-Z-]+", lower)
                return (first_word[0] if first_word else synonym_values[0]), dtype

        for syn, atype in self._action_synonyms_ordered:
            pattern = rf"\b{re.escape(syn)}\b"
            if re.search(pattern, lower) and not self._matches_noun_action_guard(syn, lower):
                clear_synonyms = self.language_pack.actions.get("clear").synonyms
                if atype == "type" and any(
                    clear in lower and lower.index(clear) < lower.index(syn) for clear in clear_synonyms
                ):
                    continue
                return syn, atype
        return None, None

    def _matches_noun_action_guard(self, synonym: str, text: str) -> bool:
        """Return whether a synonym appears in a configured noun-only phrase."""
        guard_patterns = self._noun_action_guard_patterns.get(synonym.lower(), [])
        return any(pattern.search(text) for pattern in guard_patterns)

    def _inherit_action(self, protected: str, implicit: str) -> Tuple[Optional[str], Optional[str]]:
        """Attempts to inherit action from previous context."""
        words = protected.lower().split()
        if len(words) <= 3 or any(w in self.language_pack.leading_prepositions for w in words):
            atype = next((at for at, spec in self.language_pack.actions.items() if implicit in spec.synonyms), None)
            return implicit, atype
        return None, None

    def _handle_no_verb(self, protected: str, quotes: List[str]) -> tuple:
        """Handles cases where no action verb is found."""
        low = protected.lower().strip()
        if low in self.language_pack.bare_click_targets:
            return (
                [SemanticNode("FIND", "element"), SemanticNode("DO", f"click {protected.strip()}")],
                "element",
                "click",
            )
        if any(target in low for target in self.language_pack.bare_click_targets):
            return (
                [SemanticNode("FIND", protected.strip()), SemanticNode("DO", f"click {protected.strip()}")],
                protected.strip(),
                "click",
            )

        res = protected
        for i, q in enumerate(quotes):
            res = res.replace(f"__QUOTE_{i}__", q)
        return [], res.strip(), None

    def _extract_payload(self, protected: str, verb: str, atype: str, quotes: List[str]) -> Tuple[str, str]:
        """Extracts payload (like text to type) from the chunk."""
        if atype not in ["type", "enter", "click"]:
            return "", protected

        lower = protected.lower()
        if atype in ["type", "enter"]:
            m = re.search(r"__QUOTE_(\d+)__", protected)
            if m:
                q_idx = int(m.group(1))
                return quotes[q_idx], protected.replace(m.group(0), "")

            p_m = re.search(rf"\b{re.escape(verb)}\s+([a-zA-Z0-9_@.-]+)", lower) or re.search(
                r"^([a-zA-Z0-9_@.-]+)", lower.strip()
            )
            if p_m:
                payload = p_m.group(p_m.lastindex)
                return payload, re.sub(re.escape(payload), "", protected, count=1, flags=re.I)
        return "", protected

    def _handle_relational_actions(self, target: str, atype: str, quotes: List[str]) -> Optional[tuple]:
        """Handles complex actions like drag-onto or select-from."""
        drag_connectors = alternation_pattern(self.language_pack.drag_target_connectors)
        select_connectors = alternation_pattern(self.language_pack.select_source_connectors)
        if atype == "drag" and re.search(rf"\b(?:{drag_connectors})\b", target, flags=re.I):
            parts = re.split(rf"\b(?:{drag_connectors})\b", target, flags=re.I, maxsplit=1)
            if len(parts) == 2:
                src, dst = self._clean_noise(parts[0], quotes), self._clean_noise(parts[1], quotes)
                return (
                    [
                        SemanticNode("FIND", src),
                        SemanticNode("DO", "drag"),
                        SemanticNode("FIND", dst),
                        SemanticNode("DO", "drop"),
                    ],
                    dst,
                    "drag",
                )
        if atype == "select" and re.search(rf"\b(?:{select_connectors})\b", target, flags=re.I):
            parts = re.split(rf"\b(?:{select_connectors})\b", target, flags=re.I, maxsplit=1)
            if len(parts) == 2:
                opt, container = self._clean_noise(parts[0], quotes), self._clean_noise(parts[1], quotes)
                return [SemanticNode("FIND", container), SemanticNode("DO", f"select {opt}")], container, "select"
        return None

    def _clean_noise(self, s: str, quotes: List[str]) -> str:
        """Helper to clean articles and restore quotes."""
        res = self._article_prefix_re.sub("", s.strip()).strip()
        for i, q in enumerate(quotes):
            res = res.replace(f"__QUOTE_{i}__", q)
        return res

    def _clean_target_string(self, s: str, verb: str, atype: Optional[str]) -> str:
        """Strips verbs and prepositions from target string."""
        s = re.sub(r"\b" + re.escape(verb) + r"\b", "", s, flags=re.I, count=1)
        s = self._target_cleanup_re.sub("", s)
        s = self._leading_preposition_prefix_re.sub("", s.strip()).strip()
        s = self._article_prefix_re.sub("", s).strip()
        return re.sub(r"^[,\.]|[,\.]$|\s+", " ", s).strip()

    def _resolve_final_target(self, target: str, implicit: str) -> str:
        """Resolves target, using implicit if needed."""
        if target.lower() in [*self.language_pack.pronouns, ""]:
            return implicit or "element"
        return target

    def _assemble_nodes(self, action_info: tuple, payload: str, quotes: List[str], chunk: str) -> tuple:
        """Final assembly of SemanticNodes."""
        target, atype, verb = action_info
        if atype == "wait":
            val = chunk.lower().strip()
            for i, q in enumerate(quotes):
                val = val.replace(f"__quote_{i}__", q.lower())
            if self._wait_condition_prefix_re.search(val):
                return self._parse_verify(self._wait_condition_prefix_re.sub("", val, count=1)), "element", None
            return [SemanticNode("DO", val)], "element", verb
        if atype == "scroll":
            scroll_target = self._clean_target_string(chunk, verb, atype)
            return [SemanticNode("DO", chunk.lower().strip())], scroll_target or "element", verb
        if atype == "find":
            return [], target, None

        nodes = [SemanticNode("FIND", target)]
        canonical_synonyms = (
            self.language_pack.actions.get(atype).synonyms if atype in self.language_pack.actions else []
        )
        do_name = verb if verb in canonical_synonyms else (atype or verb or "interact")
        do_val = f"{do_name} {payload}" if payload else do_name

        if atype in ["type", "enter"] and not payload:
            return [SemanticNode("FIND", "element"), SemanticNode("DO", f"{do_val} {target}")], target, verb

        nodes.append(SemanticNode("DO", do_val.lower()))
        return nodes, target, verb
