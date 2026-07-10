"""Mock market API responses."""

from __future__ import annotations


def mock_market_response():
    """Return a minimal market response object."""

    class Response:
        def read(self):
            return b"{\"prices\": [2400, 2450, 2500, 2550]}"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    return Response()

