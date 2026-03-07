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
        "type": ["type", "enter", "input", "fill"],
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
        lower_chunk = chunk.lower()

        action_verb = None
        action_type = None

        # Identify Action
        # We sort synonyms by length (descending) to match longest phrase first (e.g. "move to" before a shorter substring)
        for ax_type, synonyms in self.ACTION_VERBS.items():
            for syn in sorted(synonyms, key=len, reverse=True):
                if re.search(r"\b" + re.escape(syn) + r"\b", lower_chunk):
                    # Check if 'type' is incorrectly matched in an earlier chunk that was meant to be 'clear'
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
            # Maybe it's just a noun (fallback)
            # Check edge cases like implicit verbs if standard VERB NOUN pattern is missing
            if "submit" in lower_chunk:
                return (
                    [SemanticNode(type="FIND", value="element"), SemanticNode(type="DO", value="click Submit")],
                    "element",
                    "click",
                )
            return [], implicit_target, None

        # We found an action. Let's isolate the target and the payload.
        target = ""
        payload = ""

        # 1. Payload Extraction (Explicit Quotes)
        quote_match = re.search(r"(['\"])(.*?)\1", chunk)
        if quote_match and action_type in ["type", "enter", "input"]:
            payload = quote_match.group(2)
            # Replace payload temporarily so we don't treat it as target
            chunk_no_payload = chunk[: quote_match.start()] + chunk[quote_match.end() :]
        else:
            quote_match = None
            # Implicit payload for "type"
            chunk_no_payload = chunk
            if action_type in ["type", "enter", "input"]:
                # 1. Try with preposition: "type admin into the username field"
                implicit_match = re.search(rf"\b{action_verb}\s+([a-zA-Z0-9_@.-]+)\s+(in|into|on|to)\b", lower_chunk)
                if implicit_match:
                    payload = implicit_match.group(1)
                    chunk_no_payload = re.sub(re.escape(payload), "", chunk, count=1, flags=re.IGNORECASE)
                else:
                    # 2. Try implicit suffix: "type first name john" -> target "first name", payload "john"
                    implicit_match = re.search(rf"\b{action_verb}\s+(.*?)\s+([a-zA-Z0-9_@.-]+)$", lower_chunk)
                    if implicit_match:
                        payload = implicit_match.group(2)
                        # We extract payload from end of string
                        chunk_no_payload = chunk[: implicit_match.end(1)]
            elif action_type == "click":
                # Handle "press Enter", "hit Escape"
                key_match = re.search(r"\b(enter|escape|esc|return|tab|space)\b", lower_chunk)
                if key_match and "key" not in lower_chunk:
                    payload = key_match.group(1).capitalize()
                    chunk_no_payload = re.sub(
                        r"\b" + re.escape(key_match.group(1)) + r"\b", "", chunk, flags=re.IGNORECASE
                    )
                elif key_match and "key" in lower_chunk:
                    payload = key_match.group(1).capitalize() + " key"
                    chunk_no_payload = re.sub(
                        r"\b" + re.escape(key_match.group(1)) + r"\s+key\b", "", chunk, flags=re.IGNORECASE
                    )

        # Strip Action Verb exclusively (but not if it's inside quotes)
        target_str = re.sub(r"\b" + re.escape(action_verb) + r"\b", "", chunk_no_payload, flags=re.IGNORECASE, count=1)

        # Strip exact targeted Boilerplate at boundaries or specific phrases
        target_str = re.sub(r"\b(please navigate ahead and|i want you to)\b", "", target_str, flags=re.IGNORECASE)
        target_str = re.sub(
            r"\b(click on|click)\b", "", target_str, flags=re.IGNORECASE
        )  # In case action_verb was something else or we had "click on"

        # Remove leading/trailing stop words but KEEP internal ones like "in the header"
        # We'll just remove 'the', 'a', 'an' cleanly.
        target_str = re.sub(
            r"\b(into|onto|to|on)\b", "", target_str, count=1, flags=re.IGNORECASE
        )  # usually precedes the target for type/drag
        target_str = re.sub(r"\b(the|a|an)\b", "", target_str, flags=re.IGNORECASE)

        target_str = re.sub(r"\s+", " ", target_str).strip()

        # Remove trailing/leading punctuation
        target_str = re.sub(r"^[,\.]|[,\.]$", "", target_str).strip()

        if not target_str:
            target_str = implicit_target

        nodes = []

        # Handle 'it' or empty target by using implicit target
        if target_str.lower() in ["it", "them", ""]:
            target_str = implicit_target if implicit_target else "element"

        # Special logic for wait commands
        if action_type == "wait":
            do_val = chunk.lower().strip()
            if "until" in do_val:
                verify_text = re.sub(rf"^{action_verb}\s+until\s+", "", do_val).strip()
                return self._parse_verify(verify_text), "element", None
            return (
                [SemanticNode(type="FIND", value="element"), SemanticNode(type="DO", value=do_val)],
                "element",
                action_verb,
            )

        # Special logic for find command (no-op)
        if action_type == "find":
            return [], target_str, None

        if target_str:
            nodes.append(SemanticNode(type="FIND", value=target_str))

        # Reconstruct DO string
        do_val = action_verb.lower()
        if payload:
            if action_type == "click":
                do_val = f"press {payload}"
            elif quote_match:
                do_val += f" {quote_match.group(1)}{payload}{quote_match.group(1)}"
            else:
                do_val += f" {payload}"
        elif action_type == "type" and quote_match is None and not payload:
            # Fallback if we completely missed the implicit payload
            implicit_all = re.sub(r"\b" + re.escape(action_verb) + r"\b", "", chunk, flags=re.IGNORECASE).strip()
            if implicit_all:
                do_val = f"type {implicit_all}"
                nodes = [SemanticNode(type="FIND", value="element")]  # Target is ambiguous

        nodes.append(SemanticNode(type="DO", value=do_val))

        return nodes, target_str, action_verb
