#!/usr/bin/env python3
"""
PhishGuard – VirusTotal v3 Client (VT_Client.py)

Fetches reputation scores for URLs, domains, and IP addresses.
Rate-limit (HTTP 429) handling: fixed retry cap to avoid infinite loop.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import base64
import logging
import time
from typing import Dict, List, Optional, Union

# ── Third-party ───────────────────────────────────────────────────────────────
import requests

logger = logging.getLogger("VT_Client")

# Max retries on a 429 before giving up on that artifact
_MAX_RETRIES = 5


class VT_Client:
    """
    Lightweight VirusTotal v3 reputation client.
    Handles URL encoding, rate-limiting, and missing artifacts gracefully.
    """

    BASE_URL = "https://www.virustotal.com/api/v3"

    def __init__(self, api_key: Optional[str] = None, delay_seconds: int = 15):
        self.headers       = {"x-apikey": api_key or ""}
        self.delay         = delay_seconds
        self._has_key      = bool(api_key)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _encode_url(url: str) -> str:
        """VT v3 requires URLs base64-url-safe encoded without padding."""
        return base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")

    def _get_reputation(self, endpoint: str, identifier: str) -> Union[int, str]:
        """
        GET /v3/{endpoint}/{identifier} and return the 'reputation' integer.
        Retries on 429 up to _MAX_RETRIES times; returns an error string otherwise.
        """
        url = f"{self.BASE_URL}/{endpoint}/{identifier}"
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.get(url, headers=self.headers, timeout=15)
            except requests.RequestException as exc:
                return f"Network Error: {exc}"

            if resp.status_code == 200:
                return (
                    resp.json()
                    .get("data", {})
                    .get("attributes", {})
                    .get("reputation", 0)
                )
            if resp.status_code == 429:
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Sleeping %ds …",
                    attempt, _MAX_RETRIES, self.delay,
                )
                time.sleep(self.delay)
                continue
            if resp.status_code == 404:
                return "Not Found"
            if resp.status_code in (401, 403):
                return "Auth/Permission Error"
            return f"API Error {resp.status_code}"

        return f"Rate limit exceeded after {_MAX_RETRIES} retries"

    # ── Public interface ──────────────────────────────────────────────────────

    def get_reputations(
        self,
        urls:    Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        ips:     Optional[List[str]] = None,
    ) -> Dict[str, Union[int, str]]:
        """
        Query VT for all provided artifacts.  Any category may be None / empty;
        missing categories are silently skipped.
        Returns {artifact: reputation_score_or_error_string}.
        """
        if not self._has_key:
            logger.debug("No VT API key — skipping reputation lookup.")
            return {}

        urls    = [u for u in (urls    or []) if u]
        domains = [d for d in (domains or []) if d]
        ips     = [i for i in (ips     or []) if i]

        if not any([urls, domains, ips]):
            logger.debug("No artifacts to query.")
            return {}

        results: Dict[str, Union[int, str]] = {}

        for url in urls:
            rep = self._get_reputation("urls", self._encode_url(url))
            results[url] = rep
            logger.info("URL  %-60s → %s", url[:60], rep)
            time.sleep(self.delay)

        for domain in domains:
            rep = self._get_reputation("domains", domain)
            results[domain] = rep
            logger.info("DOM  %-60s → %s", domain[:60], rep)
            time.sleep(self.delay)

        for ip in ips:
            rep = self._get_reputation("ip_addresses", ip)
            results[ip] = rep
            logger.info("IP   %-60s → %s", ip[:60], rep)
            time.sleep(self.delay)

        return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI / smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    API_KEY = "PUT_YOUR_KEY_HERE"
    client  = VT_Client(api_key=API_KEY, delay_seconds=0)
    print(client.get_reputations(ips=["8.8.8.8", "1.1.1.1"]))
