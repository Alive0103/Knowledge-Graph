"""
Elasticsearch client configuration.

The original project hard-coded a remote Alibaba Cloud endpoint. That is not
usable after the remote service goes down, so this module now defaults to a
local single-node Elasticsearch instance and lets the runtime override all
connection details through environment variables.
"""

from __future__ import annotations

import os

from elasticsearch import Elasticsearch


DEFAULT_ES_URL = "http://localhost:9200"
DEFAULT_ES_INDEX_NAME = "data2"


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


ES_URL = os.getenv("KG_ES_URL", DEFAULT_ES_URL)
ES_USERNAME = os.getenv("KG_ES_USERNAME", "")
ES_PASSWORD = os.getenv("KG_ES_PASSWORD", "")
ES_INDEX_NAME = os.getenv("KG_ES_INDEX_NAME", DEFAULT_ES_INDEX_NAME)
ES_CA_CERTS = os.getenv("KG_ES_CA_CERTS", "")
ES_REQUEST_TIMEOUT = int(os.getenv("KG_ES_REQUEST_TIMEOUT", "30"))
ES_VERIFY_CERTS = _read_bool_env("KG_ES_VERIFY_CERTS", ES_URL.startswith("https://"))
ES_COMPAT_MODE = _read_bool_env("KG_ES_COMPAT_MODE", "aliyuncs.com" in ES_URL)


def create_es_client() -> Elasticsearch:
    kwargs: dict[str, object] = {
        "request_timeout": ES_REQUEST_TIMEOUT,
    }

    if ES_USERNAME:
        kwargs["basic_auth"] = (ES_USERNAME, ES_PASSWORD)
    if ES_CA_CERTS:
        kwargs["ca_certs"] = ES_CA_CERTS
    if ES_URL.startswith("https://"):
        kwargs["verify_certs"] = ES_VERIFY_CERTS
    if ES_COMPAT_MODE:
        kwargs["headers"] = {
            "accept": "application/vnd.elasticsearch+json;compatible-with=8",
            "content-type": "application/vnd.elasticsearch+json;compatible-with=8",
        }
        kwargs["http_compress"] = True

    return Elasticsearch(ES_URL, **kwargs)


def get_es_runtime_config() -> dict[str, object]:
    return {
        "url": ES_URL,
        "index_name": ES_INDEX_NAME,
        "has_auth": bool(ES_USERNAME),
        "verify_certs": ES_VERIFY_CERTS,
        "compat_mode": ES_COMPAT_MODE,
        "request_timeout": ES_REQUEST_TIMEOUT,
        "ca_certs": ES_CA_CERTS,
    }


es = create_es_client()
