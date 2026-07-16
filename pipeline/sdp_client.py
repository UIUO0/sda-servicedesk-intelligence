"""Thin, read-only HTTP client for the ServiceDesk Plus API v3.

Every method issues a GET request only — this client never creates, updates or
deletes anything on the server. It centralises authentication, URL-encoded
``input_data`` construction, pagination and retry/back-off handling so the
extractor stays simple.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import quote

import requests

import config

logger = logging.getLogger(__name__)

# Transient statuses worth retrying with back-off.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class SDPClient:
    """Minimal GET-only wrapper around the SDP API v3."""

    def __init__(
        self,
        base_url: str = config.BASE_URL,
        authtoken: str = config.AUTHTOKEN,
        *,
        verify_ssl: bool = config.VERIFY_SSL,
        row_count: int = config.ROW_COUNT,
        rate_limit_sleep: float = config.RATE_LIMIT_SLEEP,
        timeout: int = config.TIMEOUT,
        max_retries: int = config.MAX_RETRIES,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.row_count = row_count
        self.rate_limit_sleep = rate_limit_sleep
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl

        self.session = requests.Session()
        self.session.headers.update(
            {"authtoken": authtoken, "Accept": "application/vnd.manageengine.sdp.v3+json"}
        )
        if not verify_ssl:
            # On-premise servers commonly use self-signed certs; silence the noise.
            requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

    # -- low level ---------------------------------------------------------
    def _request(self, path: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """GET ``{base_url}{path}`` with retry/back-off. Returns parsed JSON."""
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(
                    url, params=params, timeout=self.timeout, verify=self.verify_ssl
                )
            except requests.RequestException as exc:  # network-level failure
                last_exc = exc
                wait = self._backoff(attempt)
                logger.warning("GET %s failed (%s); retry %d/%d in %.1fs",
                               url, exc, attempt, self.max_retries, wait)
                time.sleep(wait)
                continue

            if resp.status_code in (401, 403):
                raise PermissionError(
                    f"Authentication/authorization failed ({resp.status_code}) for {url}. "
                    "Check SDP_AUTHTOKEN and the technician's permissions."
                )

            if resp.status_code in _RETRYABLE_STATUS:
                wait = self._retry_after(resp) or self._backoff(attempt)
                logger.warning("GET %s -> %d; retry %d/%d in %.1fs",
                               url, resp.status_code, attempt, self.max_retries, wait)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            if self.rate_limit_sleep:
                time.sleep(self.rate_limit_sleep)
            return resp.json()

        raise RuntimeError(f"GET {url} failed after {self.max_retries} retries") from last_exc

    @staticmethod
    def _backoff(attempt: int) -> float:
        """Exponential back-off capped at 30s."""
        return min(2.0 ** attempt, 30.0)

    @staticmethod
    def _retry_after(resp: requests.Response) -> Optional[float]:
        value = resp.headers.get("Retry-After")
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _input_data(list_info: Dict[str, Any]) -> Dict[str, str]:
        """Build the URL-safe ``input_data`` query param the list API expects."""
        payload = json.dumps({"list_info": list_info}, separators=(",", ":"))
        return {"input_data": quote(payload)}

    # -- public API --------------------------------------------------------
    def list_records(
        self, module: str, *, start_index: int = 1, row_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """Fetch one page of ``module`` (e.g. "requests"). Returns the raw JSON body."""
        list_info = {
            "row_count": row_count or self.row_count,
            "start_index": start_index,
            "get_total_count": True,
        }
        return self._request(f"/api/v3/{module}", params=self._input_data(list_info))

    def paginate(self, module: str, *, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Yield each raw page dict until ``has_more_rows`` is false.

        ``limit`` caps the number of *records* pulled (handy for smoke tests).
        The data list key in SDP responses matches the module name (e.g.
        "requests", "problems"), so callers extract it themselves.
        """
        start_index = 1
        pulled = 0
        while True:
            row_count = self.row_count
            if limit is not None:
                row_count = min(row_count, limit - pulled)
                if row_count <= 0:
                    return
            page = self.list_records(module, start_index=start_index, row_count=row_count)
            yield page

            list_info = page.get("list_info") or {}
            records = self._records_from(page, module)
            pulled += len(records)

            if limit is not None and pulled >= limit:
                return
            if not list_info.get("has_more_rows"):
                return
            start_index = (list_info.get("start_index") or start_index) + len(records)

    def get_detail(self, module: str, record_id: str) -> Dict[str, Any]:
        """GET a single record's full body (includes description/resolution)."""
        return self._request(f"/api/v3/{module}/{record_id}")

    def get_subresource(self, module: str, record_id: str, sub: str) -> Dict[str, Any]:
        """GET a sub-resource of a record, e.g. ``notes`` or ``_attachments``."""
        return self._request(f"/api/v3/{module}/{record_id}/{sub}")

    @staticmethod
    def _records_from(page: Dict[str, Any], module: str) -> List[Dict[str, Any]]:
        """Extract the list of records from a page body.

        SDP keys the list by module name; fall back to the first list value.
        """
        if module in page and isinstance(page[module], list):
            return page[module]
        for value in page.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
        return []
