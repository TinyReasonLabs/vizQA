# UI Perception & Search API Summary

This API provides a visual perception layer for UI-driven workflows, enabling LLM agents and automation scripts to interpret and interact with graphical user interfaces via screenshots.

## Core Capabilities

### 1. Visual Perception (`POST /v1/perceive`)
The primary entry point for analyzing a UI. It converts a raw image into a structured, LLM-friendly representation.

*   **Input**:
    *   `file`: A screenshot of the UI (PNG/JPG).
    *   `query` (Optional): A natural language description of what to find (e.g., "login button").
    *   `mode`: Search mode (Default: `HYBRID`).
    *   `config`: Parameters like `lod_threshold`, `similarity_threshold`, and `top_k`.
*   **Result**:
    *   A unique `session_id` to track the state.
    *   Viewport metadata (width/height).
    *   A collection of detected UI elements with bounding boxes, labels, and salience scores.
    *   Optional YAML snapshot for direct prompt injection.

### 2. Contextual Search (`POST /v1/search`)
Enables follow-up queries on a previously analyzed UI without reprocessing the image.

*   **Input**:
    *   `session_id`: The ID returned by the initial perceive request.
    *   `query`: A search term or intent (e.g., "help icon").
*   **Result**:
    *   A ranked list of UI elements matching the query based on semantic and keyword similarity.

### 3. Reliability & Monitoring (`GET /health`)
*   Returns system status and GPU availability (`torch.cuda.is_available`).

## Integration Highlights
*   **Session-Based**: State is maintained via `session_id`, allowing multi-turn interactions with the same UI state.
*   **Multi-Modal**: Combines computer vision (element detection) with NLP (semantic search).
*   **LLM Optimized**: Outputs are designed to be consumed by Large Language Models, supporting both JSON and YAML formats.
