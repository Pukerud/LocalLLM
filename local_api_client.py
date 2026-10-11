"""Proxy-free, redirect-rejecting client for the local Strata HTTP API."""
from urllib.parse import urlsplit
import urllib.request


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


def open_local_api(request, timeout):
    url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
    parsed = urlsplit(url)
    if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1':
        raise ValueError('Local Strata API requests must use plain HTTP to 127.0.0.1')
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _RejectRedirects())
    return opener.open(request, timeout=timeout)
