"""
Client for interacting with the perception API.
"""

from typing import Any, Dict, Optional

import httpx


class PerceptionClient:
    """
    A client for sending requests to the Vision Perception API.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def perceive(self, image_path: str, query: Optional[str] = None) -> Dict[str, Any]:
        """Send a screenshot to the perception API."""
        async with httpx.AsyncClient() as client:
            with open(image_path, "rb") as file:
                files = {"file": file}
                data = {}
                if query:
                    data["query"] = query

                response = await client.post(f"{self.base_url}/v1/perceive", files=files, data=data)
                response.raise_for_status()
                return response.json()

    async def search(self, session_id: str, query: str) -> Dict[str, Any]:
        """Perform a contextual search on an existing session."""
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.base_url}/v1/search", json={"session_id": session_id, "query": query})
            response.raise_for_status()
            return response.json()
