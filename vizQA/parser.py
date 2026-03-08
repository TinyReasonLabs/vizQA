import re
from typing import List, NamedTuple


class SemanticNode(NamedTuple):
    type: str
    value: str


class SemanticParser:
    """
    Advanced Rule-Based Engine (AST Parser) for dissecting UI testing instructions.
    Replaces the LLM/Embedding approach for perfect determinism.
    """

    # Core Action Verbs
    ACTION_VERBS = {
        "click": ["click", "tap", "hit", "press"],
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
    # We use regex to carefully split clauses
    SPLIT_PATTERN = re.compile(r"\b(?:and|then|after|while)\b|,|->")

    def parse(self, instruction: str) -> List[SemanticNode]:
        """Parses a full natural language instruction into a list of atomic SemanticNodes."""
        nodes = []

        # 1. Broad Splits
        # Split by explicit flow arrows "->" which almost always mean VERIFY comes next
        parts = re.split(r"->|=>", instruction)
        nodes.extend(self._parse_clause(parts[0]))

        for verification_part in parts[1:]:
            nodes.extend(self._parse_verify(verification_part))

        return nodes

    def _parse_verify(self, clause: str) -> List[SemanticNode]:
        """Extracts a strict verification state."""
        clause = clause.strip()
        if not clause:
            return []

        # Strip common verification boilerplate
        boilerplate = r"^(verify( that)?|assert( that)?|ensure( that)?|make sure( that)?|check that)\b"
        cleaned = re.sub(boilerplate, "", clause, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^(the|a|an)\b", "", cleaned, flags=re.IGNORECASE).strip()

        return [SemanticNode(type="VERIFY", value=cleaned)]

    def _parse_clause(self, clause: str) -> List[SemanticNode]:
        """Parses an instruction sequence, breaking it by standard conjunctions."""
        nodes = []
        # Split on conjunctions
        chunks = [
            c.strip()
            for c in self.SPLIT_PATTERN.split(clause)
            if c.strip() and c.strip().lower() not in ["and", "then", "after", "while", ","]
        ]

        current_target = "element"  # Implicit context
        current_action = None  # Implicit action for missing verbs

        for chunk in chunks:
            # If the chunk is entirely a verification
            is_verify = any(chunk.lower().startswith(v) for v in self.VERIFY_VERBS)
            if is_verify:
                nodes.extend(self._parse_verify(chunk))
                continue

            chunk_nodes, new_target, new_action = self._parse_atomic_action(chunk, current_target, current_action)
            nodes.extend(chunk_nodes)
            if new_target and new_target != "element":
                current_target = new_target
            if new_action:
                current_action = new_action

        return nodes

    def _parse_atomic_action(
        self, chunk: str, implicit_target: str, implicit_action: str = None
    ) -> tuple[List[SemanticNode], str, str]:
        """Takes a single continuous phrase and extracts FIND and DO."""
        # 1. Protect Quotes
        quotes = []

        def _repl(match):
            quotes.append(match.group(0))
            return f"__QUOTE_{len(quotes)-1}__"

        chunk_protected = re.sub(r"(['\"])(.*?)\1", _repl, chunk)
        lower_chunk = chunk_protected.lower()

        action_verb = None
        action_type = None

        # Identify Action
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
            # Fallback
            if "submit" in lower_chunk:
                return (
                    [SemanticNode(type="FIND", value="element"), SemanticNode(type="DO", value="click Submit")],
                    "element",
                    "click",
                )

            restored = chunk_protected
            for i, q in enumerate(quotes):
                restored = restored.replace(f"__QUOTE_{i}__", q)
            return [], restored.strip(), None

        target = ""
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

        # Target Isolation
        target_str = re.sub(r"\b" + re.escape(action_verb) + r"\b", "", chunk_no_payload, flags=re.IGNORECASE, count=1)

        target_str = re.sub(r"\b(please navigate ahead and|i want you to)\b", "", target_str, flags=re.IGNORECASE)
        target_str = re.sub(r"\b(click on|click)\b", "", target_str, flags=re.IGNORECASE)

        target_str = re.sub(r"\b(into|onto|to|on)\b", "", target_str, count=1, flags=re.IGNORECASE)
        target_str = re.sub(r"^(the|a|an)\b", "", target_str.strip(), flags=re.IGNORECASE)

        target_str = re.sub(r"\s+", " ", target_str).strip()
        target_str = re.sub(r"^[,\.]|[,\.]$", "", target_str).strip()

        for i, q in enumerate(quotes):
            target_str = target_str.replace(f"__QUOTE_{i}__", q)
            payload = payload.replace(f"__QUOTE_{i}__", q)

        nodes = []
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

    def parse_verify_intent(self, query: str) -> dict:
        """Parses a verification query to extract specific intents like keyword, color, position, and state."""
        intent = {
            "keyword": None,
            "color": None,
            "position": None,
            "state": None,
            "subject": query
        }
        
        # 1. Extract keyword (quotes)
        quote_match = re.search(r"(['\"])(.*?)\1", query)
        if quote_match:
            intent["keyword"] = quote_match.group(2)
            intent["subject"] = intent["subject"].replace(quote_match.group(0), "")
            
        # 2. Extract color (simple list)
        lower_q = query.lower()
        colors = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "gray", "grey"]
        for c in colors:
            if re.search(rf"\b{c}\b", lower_q):
                intent["color"] = c
                intent["subject"] = re.sub(rf"\b{c}\b", "", intent["subject"], flags=re.IGNORECASE)
                break
                
        # 3. Extract position
        pos_regex = r"\b(top left|top right|bottom left|bottom right|top|bottom|left|right|center|middle)\b"
        pos_match = re.search(pos_regex, lower_q)
        if pos_match:
            intent["position"] = pos_match.group(1).replace(" ", "-")
            intent["subject"] = re.sub(pos_regex, "", intent["subject"], flags=re.IGNORECASE)
            
        # 4. Extract state
        states = ["disabled", "enabled", "checked", "unchecked", "visible", "invisible", "hidden", "displayed"]
        for s in states:
            if re.search(rf"\b{s}\b", lower_q):
                intent["state"] = s
                intent["subject"] = re.sub(rf"\b{s}\b", "", intent["subject"], flags=re.IGNORECASE)
                break
                
        # Clean subject
        intent["subject"] = re.sub(r"\b(should appear|should close|should occur|is|at the|in the|on the|the|a|an|located|aligned)\b", "", intent["subject"], flags=re.IGNORECASE)
        intent["subject"] = re.sub(r'\s+', ' ', intent["subject"]).strip()
        
        return intent
