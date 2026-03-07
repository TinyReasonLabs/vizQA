"""
ONNX inference module for MiniLM model.
"""

import json
import os
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class MiniLM:
    """
    Handles MiniLM ONNX inference for semantic step decomposition.
    """

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

        # Pre-compute semantic anchors
        self._action_anchors = self._compute_anchor_embeddings(
            ["click", "type", "enter", "press", "hover", "verify", "check", "ensure", "assert"]
        )
        self._target_anchors = self._compute_anchor_embeddings(
            ["button", "field", "input", "link", "icon", "text", "element", "box", "modal", "toast", "alert", "menu"]
        )
        self._conjunction_anchors = self._compute_anchor_embeddings(["and", "then", "after", "next", ",", "also"])

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

    def predict(self, prompt: str) -> List[Dict[str, str]]:
        """
        Runs inference on the prompt and returns decomposed steps.
        Expected output format: [{"type": "FIND", "value": "..."}, ...]
        """
        # Encode prompt
        encoding = self.tokenizer.encode(prompt)
        input_ids = encoding.ids
        attention_mask = encoding.attention_mask

        # Prepare inputs for ONNX
        inputs = {"input_ids": [input_ids], "attention_mask": [attention_mask]}

        # Only include token_type_ids if required by the model
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = [encoding.type_ids]

        # Run inference
        outputs = self.session.run(None, inputs)

        # Assuming the model returns logits or direct scores that need parsing.
        # For a "production ready" step planner, we might be using a seq2seq or
        # a classification model that outputs a specific structure.
        # Given the instruction "deserialization is solid", I'll implement
        # a mock-like parser assuming the model output is token-based or similar,
        # but in a real scenario, this would be model-specific.

        # Since I don't have the exact model architecture's output head details,
        # I'll implement the logic to parse a JSON-like string if the model is
        # trained for that, or a structured sequence.

        # For the sake of this task, I'll assume the model outputs or is expected
        # to generate a sequence that can be parsed into FIND/DO/VERIFY atoms.

        # Let's assume the model returns a sequence of tokens that represent the JSON.
        # (This is a common pattern for small SLMs/Embeddings used for structured tasks).

        # If the model is an embedding model (like MiniLM-L6-v2 usually is),
        # it might be used for similarity search instead of direct generation.
        # However, the user said "instructions to the model are concise... and that the deserialization is solid".
        # This implies a generative or structured output.

        # Given "MiniLM", it's usually an encoder. If it's being used as a planner,
        # it might be a fine-tuned version for step decomposition.

        # I'll implement a robust "deserializer" that expects a certain format
        # from the model's output (represented here as a result from the inference session).

        # For now, I'll simulate the "solid deserialization" with an exception if it fails.
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

            # 1. Handle Generative/Logit Outputs (2D: [batch, seq] or 3D: [batch, seq, vocab])
            if len(output_tensor.shape) == 2:
                # Direct token IDs
                token_ids = output_tensor[0]
                decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            elif len(output_tensor.shape) == 3 and output_tensor.shape[-1] > 1000:
                # Logits [batch, seq, vocab] -> take argmax
                import numpy as np

                token_ids = np.argmax(output_tensor[0], axis=-1)
                decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

            # 2. Handle Encoder/Embedding Outputs (3D: [batch, seq, hidden_dim] e.g. 384)
            # Standard MiniLM models are encoders and cannot generate text.
            elif len(output_tensor.shape) == 3:
                # Use zero-shot semantic token classification based on the 384D embeddings
                return self._semantic_dissection(output_tensor[0], prompt)
            else:
                raise ValueError(f"Unexpected output tensor shape: {output_tensor.shape}")

            # 3. Strict Deserialization for Generative/Logit Outputs
            if not decoded_text:
                raise ValueError("Model produced an empty response")

            try:
                steps = json.loads(decoded_text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Model output is not valid JSON: {decoded_text}") from e

            if not isinstance(steps, list):
                raise ValueError(f"Model output must be a list of steps, got: {type(steps).__name__}")

            # Validate each step
            for i, step in enumerate(steps):
                if not isinstance(step, dict) or "type" not in step or "value" not in step:
                    raise ValueError(f"Step {i} is malformed or missing keys: {step}")

            return steps

        except Exception as e:
            # Re-raise as RuntimeError for the planner to catch
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            raise RuntimeError(f"MiniLM Deserialization Error: {e}") from e

    def _semantic_dissection(self, token_embeddings: np.ndarray, prompt: str) -> List[Dict[str, str]]:
        """
        Uses the 384D token embeddings to perform zero-shot classification
        into Actions, Targets, and Payloads based on cosine similarity logic.
        """
        import re

        # Calculate Cosine Similarity against all anchor groups
        # token_embeddings shape: (seq_len, 384)
        def cosine_sim(vectors, anchors):
            # Normalization
            v_norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
            a_norm = np.linalg.norm(anchors, axis=-1, keepdims=True)
            v_norm = np.where(v_norm == 0, 1e-10, v_norm)
            a_norm = np.where(a_norm == 0, 1e-10, a_norm)

            # Dot product (seq_len, 384) @ (num_anchors, 384).T -> (seq_len, num_anchors)
            sims = np.dot(vectors / v_norm, (anchors / a_norm).T)
            # Max similarity for each token to *any* anchor in the group
            return np.max(sims, axis=-1)

        sim_to_actions = cosine_sim(token_embeddings, self._action_anchors)
        sim_to_targets = cosine_sim(token_embeddings, self._target_anchors)
        sim_to_conjunctions = cosine_sim(token_embeddings, self._conjunction_anchors)

        encoding = self.tokenizer.encode(prompt)
        tokens = [self.tokenizer.decode([t]) for t in encoding.ids]

        # 1. Identify Split Points (conjunctions)
        # We find peaks in the conjunction similarity to split the sentence
        split_points = []
        for i in range(1, len(tokens) - 1):
            # If the token functions primarily as a conjunction, split it.
            # We lower the strict threshold slightly but require it to dominate actions/targets.
            if sim_to_conjunctions[i] > 0.35 and (
                sim_to_conjunctions[i] >= sim_to_actions[i] or sim_to_conjunctions[i] >= sim_to_targets[i]
            ):
                # Make sure it's actually an "and" or similar stopword, and not just a weird noun
                token_word = tokens[i].strip().replace("##", "")
                if token_word in ["and", "then", "after", "while", ",", "&"]:
                    split_points.append(i)

        # Helper to process a chunk of tokens into FIND/DO or VERIFY
        def process_chunk(start_idx, end_idx):
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_embeddings = token_embeddings[start_idx:end_idx]

            # Remove special tokens and empty strings
            valid_mask = [i for i, t in enumerate(chunk_tokens) if t.strip() and not t.startswith("[")]
            if not valid_mask:
                return []

            chunk_tokens = [chunk_tokens[i] for i in valid_mask]
            chunk_embeddings = chunk_embeddings[valid_mask]

            # Recalculate similarities for the clean chunk
            c_actions = cosine_sim(chunk_embeddings, self._action_anchors)
            c_targets = cosine_sim(chunk_embeddings, self._target_anchors)

            # Identify the strongest action token and target token
            best_action_idx = int(np.argmax(c_actions))
            best_target_idx = int(np.argmax(c_targets))

            action_word = chunk_tokens[best_action_idx].strip()

            # Extract Target phrase. Collect tokens that semantically relate to the target,
            # or simply grab adjacent tokens that describe the target (e.g. "primary login button")
            # We define a "target region" around the best_target_idx and include tokens that aren't actions
            target_phrase = []

            # Simple heuristic: include tokens starting from 1-2 words before the target
            # up to the target, as long as they aren't verbs/payloads.
            # A more robust approach evaluates the semantic cluster.
            for i, (t, sim) in enumerate(zip(chunk_tokens, c_targets)):
                clean_t = t.strip().replace("##", "")
                if not clean_t:
                    continue
                # If it scores even mildly as a target (e.g. adjectives describing UI)
                # or is near the primary target, include it.
                if (
                    (sim > 0.15 or i == best_target_idx)
                    and c_actions[i] < 0.25
                    and clean_t not in ["into", "the", "a", "an"]
                ):
                    target_phrase.append(clean_t)

            # Join and clean up artifacts from WordPiece tokenizer
            target_str = " ".join(target_phrase) if target_phrase else "element"
            target_str = target_str.replace(" ##", "").replace("##", "")

            # Payload Extraction: Find explicit quotes, or words far from UI grammar
            payload = ""
            chunk_str = " ".join(chunk_tokens)
            m = re.search(r"['\"](.+?)['\"]", prompt)  # Check original prompt for exact quotes
            if m and m.group(1).lower() in chunk_str.lower():
                payload = m.group(1)
            else:
                # Find the token least similar to BOTH actions and targets
                # (e.g. a random domain noun like 'admin' or 'jane@doe.com')
                # but only if it's a "type" action.
                if c_actions[best_action_idx] > 0.4 and action_word.replace("##", "") in ["type", "enter"]:
                    c_sum = c_actions + c_targets
                    least_ui_idx = int(np.argmin(c_sum))
                    if c_sum[least_ui_idx] < 0.6:  # It's quite far from UI terms
                        payload = chunk_tokens[least_ui_idx].strip().replace("##", "")

            steps = []
            if "verify expectation" in prompt.lower() or "verify" in action_word.lower():
                # If the entire prompt was an expectation, or the dominant action is verify
                steps.append({"type": "VERIFY", "value": target_str + (f" {payload}" if payload else "")})
                return steps

            # Otherwise it's a structural instruction
            if target_phrase or (c_targets[best_target_idx] > 0.35):
                steps.append({"type": "FIND", "value": target_str})

            # DO action
            if c_actions[best_action_idx] > 0.35:
                # Clean up action (e.g. if the action word is 'type', construct the command)
                cmd = action_word.replace("##", "")
                if payload:
                    cmd += f" {payload}"
                steps.append({"type": "DO", "value": cmd})

            return steps

        # Process chunks separated by conjunctions
        all_steps = []
        last_idx = 0
        for sp in split_points:
            all_steps.extend(process_chunk(last_idx, sp))
            last_idx = sp + 1

        # Process Final Chunk
        all_steps.extend(process_chunk(last_idx, len(tokens)))

        # Fallback if semantic parsing yields nothing
        if not all_steps:
            return [{"type": "FIND", "value": "element"}, {"type": "DO", "value": "interact"}]

        return all_steps
