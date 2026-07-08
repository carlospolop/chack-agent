from __future__ import annotations

import json
import os
import re
from typing import Any, Optional
from uuid import uuid4

try:
    from agents import function_tool
except ImportError:
    function_tool = None

import requests

from .config import ToolsConfig
from .research_artifacts import record_research_json_artifact, research_artifacts_root
from .serpapi_keys import (
    is_serpapi_rate_limited,
    note_serpapi_response_error,
    usable_serpapi_keys,
)
from .telemetry import run_with_tool_logging


_FDA_PRODUCT_TYPES = {"food", "device", "drug"}


def _run_logged(tool: str, tool_input: dict, func):
    try:
        return run_with_tool_logging(tool, tool_input, func)
    except Exception as exc:
        return f"ERROR: {tool} failed ({exc})"


def _set_param_descriptions(tool, descriptions: dict[str, str]):
    schema = getattr(tool, "params_json_schema", None)
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(properties, dict):
        for name, description in descriptions.items():
            if isinstance(properties.get(name), dict):
                properties[name]["description"] = description
    current = str(getattr(tool, "description", "") or "").strip()
    if current and "Output:" not in current:
        tool.description = (
            f"{current}\n\n"
            "Parameters: Use the endpoint-specific parameter descriptions in the schema to provide product IDs, image URLs, product categories, result limits, filters, and timeouts.\n"
            "Output: Returns a compact SUCCESS/ERROR text report with product, recall, vulnerability, food-label, or visual-match records. "
            "When raw endpoint data is preserved, the output includes an Artifact JSON path."
        )
    return tool


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _short(text: str, max_chars: int = 320) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(part for part in (_as_text(item) for item in value) if part)
    if isinstance(value, dict):
        for key in ("title", "name", "value", "label", "code", "id", "link", "url", "description"):
            text = _as_text(value.get(key))
            if text:
                return text
    return ""


def _safe_filename(value: str, fallback: str = "product-data") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return text[:120] or fallback


def _artifact_dir(kind: str) -> str:
    root = research_artifacts_root()
    base = os.path.join(root, kind) if root else os.path.join("/tmp", "chack-product", kind)
    os.makedirs(base, exist_ok=True)
    return base


def _write_json_artifact(kind: str, label: str, payload: Any) -> str:
    output_dir = _artifact_dir(kind)
    path = os.path.join(output_dir, f"{_safe_filename(label)}_{uuid4().hex}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    record_research_json_artifact(path, payload, provenance=f"{kind}:{label}", kind=kind, label=label)
    return path


def _maybe(params: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        params[key] = text


class ProductSearchTool:
    def __init__(self, config: ToolsConfig):
        self.config = config

    def _max_results(self, requested: Optional[int], default_limit: int = 10) -> int:
        cfg_limit = _coerce_int(getattr(self.config, "product_max_results", default_limit), default_limit)
        cfg_limit = _clamp(cfg_limit, 1, 50)
        if requested is None:
            return cfg_limit
        return _clamp(_coerce_int(requested, cfg_limit), 1, 50)

    def _serpapi_request(self, params: dict[str, Any], timeout_seconds: int = 20) -> Any:
        api_keys = usable_serpapi_keys(os.environ.get("SERPAPI_API_KEY", ""))
        if not api_keys:
            return "ERROR: No usable SerpAPI key (not configured or all keys exhausted)."
        for idx, api_key in enumerate(api_keys):
            req_params = dict(params)
            req_params["api_key"] = api_key
            req_params["output"] = "json"
            try:
                response = requests.get("https://serpapi.com/search", params=req_params, timeout=timeout_seconds)
            except requests.exceptions.Timeout:
                return "ERROR: SerpAPI request timed out"
            except requests.exceptions.ConnectionError:
                return "ERROR: Failed to connect to SerpAPI"
            if response.status_code >= 400:
                body = (response.text or "").strip().replace("\n", " ")
                if len(body) > 220:
                    body = body[:217] + "..."
                note_serpapi_response_error(api_key, response.status_code, body)
                if is_serpapi_rate_limited(response.status_code, body) and idx < len(api_keys) - 1:
                    continue
                return f"ERROR: SerpAPI returned HTTP {response.status_code} ({body})"
            try:
                payload = response.json()
            except ValueError:
                return "ERROR: SerpAPI returned invalid JSON"
            if isinstance(payload, dict) and payload.get("error"):
                error_text = str(payload.get("error") or "")
                note_serpapi_response_error(api_key, response.status_code, error_text)
                if is_serpapi_rate_limited(response.status_code, error_text) and idx < len(api_keys) - 1:
                    continue
                return f"ERROR: SerpAPI error ({error_text})"
            return payload
        return "ERROR: All configured SerpAPI keys are rate limited."

    def search_open_food_facts_products(
        self,
        query: str,
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        limit = self._max_results(max_results)
        params = {
            "search_terms": query,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": limit,
        }
        try:
            response = requests.get(
                "https://world.openfoodfacts.org/cgi/search.pl",
                params=params,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Open Food Facts request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Open Food Facts"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Open Food Facts returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Open Food Facts returned invalid JSON"
        artifact = _write_json_artifact("open-food-facts", f"search_{query}", payload)
        products = payload.get("products") or []
        if not isinstance(products, list) or not products:
            return f"SUCCESS: No Open Food Facts products found for '{query}'.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: Open Food Facts products for '{query}' (top {min(len(products), limit)}):"]
        for idx, item in enumerate(products[:limit], start=1):
            if not isinstance(item, dict):
                continue
            name = item.get("product_name") or item.get("generic_name") or "(no name)"
            code = item.get("code") or item.get("_id") or ""
            lines.append(f"{idx}. {name} | brand: {_as_text(item.get('brands'))} | barcode: {code}")
            detail = []
            for key, label in [
                ("quantity", "quantity"),
                ("countries", "countries"),
                ("categories", "categories"),
                ("nutriscore_grade", "nutri-score"),
                ("nova_group", "NOVA"),
                ("allergens", "allergens"),
            ]:
                text = _as_text(item.get(key))
                if text:
                    detail.append(f"{label}: {text}")
            if detail:
                lines.append(f"   {' | '.join(detail)}")
            ingredients = _as_text(item.get("ingredients_text"))
            if ingredients:
                lines.append(f"   ingredients: {_short(ingredients, 260)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def get_open_food_facts_product(self, barcode: str, timeout_seconds: int = 20) -> str:
        barcode = re.sub(r"\D+", "", str(barcode or ""))
        if not barcode:
            return "ERROR: barcode cannot be empty"
        try:
            response = requests.get(
                f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json",
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: Open Food Facts request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to Open Food Facts"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: Open Food Facts returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: Open Food Facts returned invalid JSON"
        artifact = _write_json_artifact("open-food-facts", f"barcode_{barcode}", payload)
        product = payload.get("product") if isinstance(payload, dict) else {}
        if not isinstance(product, dict) or not product:
            return f"SUCCESS: No Open Food Facts product found for barcode '{barcode}'.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: Open Food Facts product for barcode '{barcode}':"]
        lines.append(f"Name: {_as_text(product.get('product_name') or product.get('generic_name'))}")
        for key, label in [
            ("brands", "Brands"),
            ("quantity", "Quantity"),
            ("categories", "Categories"),
            ("countries", "Countries"),
            ("nutriscore_grade", "Nutri-Score"),
            ("nova_group", "NOVA group"),
            ("allergens", "Allergens"),
            ("traces", "Traces"),
        ]:
            text = _as_text(product.get(key))
            if text:
                lines.append(f"{label}: {text}")
        ingredients = _as_text(product.get("ingredients_text"))
        if ingredients:
            lines.append(f"Ingredients: {_short(ingredients, 700)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_openfda_recalls(
        self,
        query: str,
        product_type: str = "food",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        query = str(query or "").strip()
        if not query:
            return "ERROR: query cannot be empty"
        product_type = str(product_type or "food").strip().lower()
        if product_type not in _FDA_PRODUCT_TYPES:
            return "ERROR: product_type must be one of food, device, drug"
        limit = self._max_results(max_results)
        params = {"search": f'product_description:"{query}"', "limit": limit}
        try:
            response = requests.get(
                f"https://api.fda.gov/{product_type}/enforcement.json",
                params=params,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: openFDA request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to openFDA"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: openFDA returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: openFDA returned invalid JSON"
        artifact = _write_json_artifact("openfda", f"{product_type}_{query}", payload)
        results = payload.get("results") or []
        if not isinstance(results, list) or not results:
            return f"SUCCESS: No openFDA {product_type} enforcement recalls found for '{query}'.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: openFDA {product_type} enforcement recalls for '{query}' (top {min(len(results), limit)}):"]
        for idx, item in enumerate(results[:limit], start=1):
            if not isinstance(item, dict):
                continue
            lines.append(
                f"{idx}. {item.get('recalling_firm') or ''} | class: {item.get('classification') or ''} | status: {item.get('status') or ''} | initiated: {item.get('recall_initiation_date') or ''}"
            )
            if item.get("product_description"):
                lines.append(f"   product: {_short(item['product_description'], 360)}")
            if item.get("reason_for_recall"):
                lines.append(f"   reason: {_short(item['reason_for_recall'], 360)}")
            if item.get("distribution_pattern"):
                lines.append(f"   distribution: {_short(item['distribution_pattern'], 240)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_nvd_cpe(
        self,
        keyword: str,
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        keyword = str(keyword or "").strip()
        if not keyword:
            return "ERROR: keyword cannot be empty"
        limit = self._max_results(max_results)
        params = {"keywordSearch": keyword, "resultsPerPage": limit}
        headers = {}
        api_key = os.environ.get("NVD_API_KEY", "").strip()
        if api_key:
            headers["apiKey"] = api_key
        try:
            response = requests.get(
                "https://services.nvd.nist.gov/rest/json/cpes/2.0",
                params=params,
                headers=headers,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: NVD CPE request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to NVD"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: NVD returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: NVD returned invalid JSON"
        artifact = _write_json_artifact("nvd-cpe", f"cpe_{keyword}", payload)
        products = payload.get("products") or []
        if not isinstance(products, list) or not products:
            return f"SUCCESS: No NVD CPE products found for '{keyword}'.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: NVD CPE product matches for '{keyword}' (top {min(len(products), limit)}):"]
        for idx, wrapper in enumerate(products[:limit], start=1):
            item = wrapper.get("cpe") if isinstance(wrapper, dict) else {}
            if not isinstance(item, dict):
                continue
            titles = item.get("titles") or []
            title = ""
            if isinstance(titles, list):
                for title_item in titles:
                    if isinstance(title_item, dict) and title_item.get("title"):
                        title = str(title_item["title"])
                        break
            lines.append(f"{idx}. {title or item.get('cpeName') or ''}")
            lines.append(f"   cpeName: {item.get('cpeName') or ''} | deprecated: {item.get('deprecated')}")
            refs = item.get("refs") or []
            if isinstance(refs, list) and refs:
                url = _as_text(refs[0])
                if url:
                    lines.append(f"   ref: {url}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_nvd_cve(
        self,
        keyword: str,
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        keyword = str(keyword or "").strip()
        if not keyword:
            return "ERROR: keyword cannot be empty"
        limit = self._max_results(max_results)
        params = {"keywordSearch": keyword, "resultsPerPage": limit}
        headers = {}
        api_key = os.environ.get("NVD_API_KEY", "").strip()
        if api_key:
            headers["apiKey"] = api_key
        try:
            response = requests.get(
                "https://services.nvd.nist.gov/rest/json/cves/2.0",
                params=params,
                headers=headers,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "ERROR: NVD CVE request timed out"
        except requests.exceptions.ConnectionError:
            return "ERROR: Failed to connect to NVD"
        except requests.exceptions.HTTPError as exc:
            return f"ERROR: NVD returned HTTP {exc.response.status_code}"
        except ValueError:
            return "ERROR: NVD returned invalid JSON"
        artifact = _write_json_artifact("nvd-cve", f"cve_{keyword}", payload)
        vulns = payload.get("vulnerabilities") or []
        if not isinstance(vulns, list) or not vulns:
            return f"SUCCESS: No NVD CVEs found for '{keyword}'.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: NVD CVEs for '{keyword}' (top {min(len(vulns), limit)}):"]
        for idx, wrapper in enumerate(vulns[:limit], start=1):
            cve = wrapper.get("cve") if isinstance(wrapper, dict) else {}
            if not isinstance(cve, dict):
                continue
            description = ""
            for desc in cve.get("descriptions") or []:
                if isinstance(desc, dict) and str(desc.get("lang") or "").lower() == "en":
                    description = str(desc.get("value") or "")
                    break
            severity = ""
            metrics = cve.get("metrics") or {}
            if isinstance(metrics, dict):
                for metric_key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                    values = metrics.get(metric_key) or []
                    if isinstance(values, list) and values:
                        data = values[0].get("cvssData") if isinstance(values[0], dict) else {}
                        if isinstance(data, dict):
                            severity = " ".join(str(part) for part in [data.get("baseSeverity"), data.get("baseScore")] if part not in (None, ""))
                            break
            lines.append(
                f"{idx}. {cve.get('id') or ''} | status: {cve.get('vulnStatus') or ''} | published: {cve.get('published') or ''} | modified: {cve.get('lastModified') or ''} | severity: {severity}"
            )
            if description:
                lines.append(f"   {_short(description, 500)}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)

    def search_google_lens_products(
        self,
        image_url: str,
        query: str = "",
        country: str = "",
        hl: str = "",
        search_type: str = "products",
        max_results: Optional[int] = None,
        timeout_seconds: int = 20,
    ) -> str:
        image_url = str(image_url or "").strip()
        if not image_url:
            return "ERROR: image_url cannot be empty"
        search_type = str(search_type or "products").strip().lower()
        if search_type not in {"all", "exact_matches", "products", "visual_matches"}:
            return "ERROR: search_type must be one of all, exact_matches, products, visual_matches"
        limit = self._max_results(max_results)
        params: dict[str, Any] = {
            "engine": "google_lens",
            "url": image_url,
            "type": search_type,
        }
        _maybe(params, "q", query)
        _maybe(params, "country", country)
        _maybe(params, "hl", hl)
        payload = self._serpapi_request(params, timeout_seconds=timeout_seconds)
        if isinstance(payload, str):
            return payload
        artifact = _write_json_artifact("google-lens", f"lens_{image_url}", payload)
        results = (
            payload.get("products")
            or payload.get("visual_matches")
            or payload.get("exact_matches")
            or payload.get("shopping_results")
            or []
        )
        if not isinstance(results, list) or not results:
            return f"SUCCESS: No Google Lens {search_type} results found for '{image_url}'.\nArtifact JSON: {artifact}"
        lines = [f"SUCCESS: Google Lens {search_type} results for '{image_url}' (top {min(len(results), limit)}):"]
        for idx, item in enumerate(results[:limit], start=1):
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("name") or "(no title)"
            price = _as_text(item.get("price") or item.get("extracted_price"))
            source = _as_text(item.get("source") or item.get("seller") or item.get("domain"))
            lines.append(f"{idx}. {title} | price: {price} | source: {source}")
            link = _as_text(item.get("link") or item.get("url") or item.get("product_link"))
            if link:
                lines.append(f"   {link}")
        lines.append(f"Artifact JSON: {artifact}")
        return "\n".join(lines)


def get_open_food_facts_search_tool(helper: ProductSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_open_food_facts_products")
    def search_open_food_facts_products(query: str, max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search Open Food Facts public product records by product, brand, or ingredient query."""
        tool_input = {"query": query, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_open_food_facts_products", tool_input, lambda: helper.search_open_food_facts_products(query, max_results, timeout_seconds))

    return _set_param_descriptions(search_open_food_facts_products, {
        "query": "Product name, brand, ingredient, barcode fragment, or nutrition-related term to search in Open Food Facts.",
        "max_results": "Maximum number of Open Food Facts product records to return.",
        "timeout_seconds": "Maximum seconds to wait for the Open Food Facts search.",
    })


def get_open_food_facts_product_tool(helper: ProductSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="get_open_food_facts_product")
    def get_open_food_facts_product(barcode: str, timeout_seconds: int = 20) -> str:
        """Fetch a single Open Food Facts product by barcode."""
        tool_input = {"barcode": barcode, "timeout_seconds": timeout_seconds}
        return _run_logged("get_open_food_facts_product", tool_input, lambda: helper.get_open_food_facts_product(barcode, timeout_seconds))

    return _set_param_descriptions(get_open_food_facts_product, {
        "barcode": "Exact product barcode/GTIN to fetch from Open Food Facts.",
        "timeout_seconds": "Maximum seconds to wait for the Open Food Facts product fetch.",
    })


def get_openfda_recalls_search_tool(helper: ProductSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_openfda_recalls")
    def search_openfda_recalls(query: str, product_type: str = "food", max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search openFDA enforcement recalls for food, device, or drug product descriptions."""
        tool_input = {"query": query, "product_type": product_type, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_openfda_recalls", tool_input, lambda: helper.search_openfda_recalls(query, product_type, max_results, timeout_seconds))

    return _set_param_descriptions(search_openfda_recalls, {
        "query": "Product, brand, company, reason, or description text to search in openFDA enforcement recalls.",
        "product_type": "openFDA enforcement product category to search: food, device, or drug.",
        "max_results": "Maximum number of openFDA recall records to return.",
        "timeout_seconds": "Maximum seconds to wait for the openFDA recall search.",
    })


def get_nvd_cpe_search_tool(helper: ProductSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_nvd_cpe_products")
    def search_nvd_cpe_products(keyword: str, max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search NVD CPE product identifiers for software, hardware, and platform products."""
        tool_input = {"keyword": keyword, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_nvd_cpe_products", tool_input, lambda: helper.search_nvd_cpe(keyword, max_results, timeout_seconds))

    return _set_param_descriptions(search_nvd_cpe_products, {
        "keyword": "Software, hardware, platform, vendor, or product keyword to search in NVD CPE names.",
        "max_results": "Maximum number of NVD CPE product identifiers to return.",
        "timeout_seconds": "Maximum seconds to wait for the NVD CPE search.",
    })


def get_nvd_cve_search_tool(helper: ProductSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_nvd_cve_vulnerabilities")
    def search_nvd_cve_vulnerabilities(keyword: str, max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search NVD CVE vulnerability records for product keywords."""
        tool_input = {"keyword": keyword, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_nvd_cve_vulnerabilities", tool_input, lambda: helper.search_nvd_cve(keyword, max_results, timeout_seconds))

    return _set_param_descriptions(search_nvd_cve_vulnerabilities, {
        "keyword": "Product, vendor, technology, CPE fragment, or vulnerability keyword to search in NVD CVE records.",
        "max_results": "Maximum number of NVD CVE vulnerability records to return.",
        "timeout_seconds": "Maximum seconds to wait for the NVD CVE search.",
    })


def get_google_lens_products_tool(helper: ProductSearchTool):
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")

    @function_tool(name_override="search_google_lens_products")
    def search_google_lens_products(image_url: str, query: str = "", country: str = "", hl: str = "", search_type: str = "products", max_results: Optional[int] = None, timeout_seconds: int = 20) -> str:
        """Search Google Lens via SerpAPI for product/visual matches from an image URL."""
        tool_input = {"image_url": image_url, "query": query, "country": country, "hl": hl, "search_type": search_type, "max_results": max_results, "timeout_seconds": timeout_seconds}
        return _run_logged("search_google_lens_products", tool_input, lambda: helper.search_google_lens_products(image_url, query, country, hl, search_type, max_results, timeout_seconds))

    return _set_param_descriptions(search_google_lens_products, {
        "image_url": "Publicly reachable image URL to submit to Google Lens through SerpAPI.",
        "query": "Optional text query to refine visual/product matches for the image.",
        "country": "Optional country code to localize Google Lens results, such as us or es.",
        "hl": "Optional Google interface language code for Lens results, such as en or es.",
        "search_type": "Google Lens result type to request, commonly products, visual_matches, exact_matches, or shopping.",
        "max_results": "Maximum number of Google Lens result items to return.",
        "timeout_seconds": "Maximum seconds to wait for the Google Lens request.",
    })
