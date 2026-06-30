"""
Client for interacting with the perception API.
"""

from typing import Any, Dict, Optional

import httpx

from vizQA.app.config import PERCEPTION_BACKEND
from vizQA.app.exceptions import PerceptionServiceError
from vizQA.app.logger import get_logger


class PerceptionClient:
    """
    A client for sending requests to the Vision Perception API.
    """

    def __init__(self, base_url: Optional[str] = None, logger: Optional[Any] = None):
        self.base_url = base_url or PERCEPTION_BACKEND
        self.session_id = None
        self._scoped_session_ids: Dict[str, str] = {}
        self._logger = logger or get_logger()

    async def perceive(
        self, image_path: str, query: Optional[str] = None, session_scope: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a screenshot to the perception API."""
        async with httpx.AsyncClient() as client:
            with open(image_path, "rb") as file:
                files = {"file": file}
                data = {
                    "similarity_threshold": 0.5,
                }
                if query:
                    data["query"] = query
                if session_scope and session_scope in self._scoped_session_ids:
                    data["session_id"] = self._scoped_session_ids[session_scope]

                try:
                    response = await client.post(f"{self.base_url}/v1/perceive", files=files, data=data)
                    response.raise_for_status()
                    payload = response.json()
                    self.session_id = payload.get("session_id")
                    if session_scope and self.session_id:
                        self._scoped_session_ids[session_scope] = self.session_id
                    return payload
                except (httpx.ConnectError, httpx.TimeoutException) as e:
                    self._logger.log_exception("client", e)
                    raise PerceptionServiceError(
                        f"Could not connect to the visual perception service (at {self.base_url}).",
                        internal_detail=str(e),
                    ) from e
                except httpx.HTTPStatusError as e:
                    self._logger.log_exception("client", e)
                    raise PerceptionServiceError(
                        "The visual perception service returned an error.",
                        internal_detail=f"HTTP Status {e.response.status_code}: {e.response.text}",
                    ) from e
                except Exception as e:
                    self._logger.log_exception("client", e)
                    raise PerceptionServiceError(
                        "An unexpected error occurred while communicating with the visual perception service.",
                        internal_detail=str(e),
                    ) from e

    async def search(self, session_id: str, query: str) -> Dict[str, Any]:
        """Perform a contextual search on an existing session."""
        if not session_id:
            session_id = self.session_id
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/v1/search", json={"session_id": session_id, "query": query, "mode": "semantic"}
                )
                response.raise_for_status()
                return response.json()
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                self._logger.log_exception("client", e)
                raise PerceptionServiceError(
                    f"Could not connect to the visual perception service for search (at {self.base_url}).",
                    internal_detail=str(e),
                ) from e
            except httpx.HTTPStatusError as e:
                self._logger.log_exception("client", e)
                raise PerceptionServiceError(
                    "The visual perception service search failed.",
                    internal_detail=f"HTTP Status {e.response.status_code}: {e.response.text}",
                ) from e
            except Exception as e:
                self._logger.log_exception("client", e)
                raise PerceptionServiceError(
                    "An unexpected error occurred during visual perception search.", internal_detail=str(e)
                ) from e
