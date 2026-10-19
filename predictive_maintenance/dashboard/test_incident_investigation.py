"""Contract regression checks for the dashboard's alert investigation UI.

The dashboard is a browser-only, dependency-free artifact.  These checks keep
the regression test equally small: they read the sibling HTML file, inspect its
markup with :mod:`html.parser`, and inspect the inline JavaScript with regular
expressions.  They intentionally do not execute the dashboard in a browser.
"""

from __future__ import annotations

import re
import unittest
from html.parser import HTMLParser
from pathlib import Path


INCIDENT_STORAGE_KEY = "axon_edge_incidents_threshold_v1"
DASHBOARD_PATH = Path(__file__).with_name("index.html")


class _Node:
    """Small DOM-like node used only for resilient markup assertions."""

    def __init__(self, tag: str, attrs: dict[str, str], parent: "_Node | None" = None):
        self.tag = tag
        self.attrs = attrs
        self.parent = parent
        self.children: list[_Node] = []
        self.text_parts: list[str] = []

    @property
    def text(self) -> str:
        return "".join(self.text_parts) + "".join(child.text for child in self.children)

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


class _DashboardParser(HTMLParser):
    """Parse enough HTML structure to scope control assertions to the feature."""

    _VOID_TAGS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("#document", {})
        self._stack = [self.root]

    def handle_starttag(self, tag, attrs):  # type: ignore[no-untyped-def]
        node = _Node(tag.lower(), {name.lower(): value or "" for name, value in attrs}, self._stack[-1])
        self._stack[-1].children.append(node)
        if node.tag not in self._VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(self, tag, attrs):  # type: ignore[no-untyped-def]
        node = _Node(tag.lower(), {name.lower(): value or "" for name, value in attrs}, self._stack[-1])
        self._stack[-1].children.append(node)

    def handle_endtag(self, tag):  # type: ignore[no-untyped-def]
        tag = tag.lower()
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                return

    def handle_data(self, data):  # type: ignore[no-untyped-def]
        self._stack[-1].text_parts.append(data)


def _attribute_blob(node: _Node) -> str:
    """Return searchable identifying attributes without depending on formatting."""

    values = [value for name, value in node.attrs.items() if name == "id" or name == "class" or name.startswith("data-") or name in {"role", "aria-label", "aria-controls", "name"}]
    return " ".join(values).lower()


def _is_tab_control(node: _Node) -> bool:
    attrs = node.attrs
    if attrs.get("role", "").lower() == "tab":
        return True
    if any(name in attrs for name in ("data-tab", "data-investigation-tab", "data-incident-tab")):
        return True
    if node.tag not in {"button", "a", "input"}:
        return False
    marker = _attribute_blob(node)
    return "tab" in marker and ("incident" in marker or "investigation" in marker or "tab" in marker)


def _is_detail_marker(node: _Node) -> bool:
    attrs = node.attrs
    marker = _attribute_blob(node)
    return (
        "detail" in marker
        or "detail" in attrs.get("data-view", "").lower()
        or "detail" in attrs.get("data-panel", "").lower()
    )


def _is_empty_state_marker(node: _Node) -> bool:
    attrs = node.attrs
    marker = _attribute_blob(node)
    if "empty" in marker or "empty-state" in marker or "empty_state" in marker:
        return True
    text = " ".join(node.text.split()).lower()
    return bool(
        re.search(r"\bno\s+(?:open\s+)?(?:incident|alert)s?\b", text)
        or re.search(r"\bselect\s+(?:an?|the)\s+(?:incident|alert)\b", text)
    )


def _is_export_control(node: _Node) -> bool:
    attrs = node.attrs
    marker = _attribute_blob(node)
    text = " ".join(node.text.split()).lower()
    if node.tag not in {"button", "a", "input"} and attrs.get("data-action", "").lower() not in {"export", "download"}:
        return False
    return "export" in marker or "download" in marker or "export" in text or "download" in text


def _script_text(html: str) -> str:
    """Extract inline script bodies while tolerating attributes/newlines."""

    return "\n".join(re.findall(r"<script\b[^>]*>(.*?)</script\s*>", html, flags=re.IGNORECASE | re.DOTALL))


def _function_names(script: str) -> set[str]:
    """Collect named declarations, including the common inline-arrow form."""

    names = set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", script))
    names.update(
        re.findall(
            r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:(?:async\s+)?function\b|(?:async\s+)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>)",
            script,
        )
    )
    return names


class IncidentInvestigationContractTest(unittest.TestCase):
    """Verify the static contract needed by the alert investigation feature."""

    @classmethod
    def setUpClass(cls) -> None:
        if not DASHBOARD_PATH.is_file():
            raise unittest.SkipTest(f"dashboard fixture not found: {DASHBOARD_PATH}")
        cls.html = DASHBOARD_PATH.read_text(encoding="utf-8")
        cls.script = _script_text(cls.html)
        cls.parser = _DashboardParser()
        cls.parser.feed(cls.html)
        cls.nodes = list(cls.parser.root.walk())

    def _investigation_scopes(self) -> list[_Node]:
        scopes = []
        for node in self.nodes:
            if node.tag != "section":
                continue
            marker = _attribute_blob(node)
            text = " ".join(node.text.split()).lower()
            if "investigat" in marker or re.search(r"\b(?:alert|incident)\s+investigation", text):
                scopes.append(node)
        return scopes

    def test_investigation_section_and_controls_are_present(self) -> None:
        scopes = self._investigation_scopes()
        self.assertTrue(
            scopes,
            "expected a <section> for Alert/Incident Investigation, identified by its heading or investigation marker",
        )

        scope_nodes = [node for scope in scopes for node in scope.walk()]
        tablists = [
            node
            for node in scope_nodes
            if node.attrs.get("role", "").lower() == "tablist"
            or node.attrs.get("id", "").lower() == "incidenttabs"
        ]
        self.assertTrue(
            tablists,
            "investigation section should expose a tab list; individual tabs are rendered from incident records",
        )
        self.assertRegex(
            self.script,
            r"setAttribute\(\s*['\"]role['\"]\s*,\s*['\"]tab['\"]",
            "incident records should render as accessible tab controls",
        )
        self.assertTrue(
            any(_is_detail_marker(node) for node in scope_nodes),
            "investigation section should expose an incident detail control/panel",
        )
        self.assertTrue(
            any(_is_empty_state_marker(node) for node in scope_nodes),
            "investigation section should expose a marked empty state",
        )
        self.assertTrue(
            any(_is_export_control(node) for node in scope_nodes),
            "investigation section should expose an export/download control",
        )

    def test_incident_storage_key_and_existing_hooks_are_preserved(self) -> None:
        self.assertIn(
            INCIDENT_STORAGE_KEY,
            self.script,
            "the existing incident localStorage key must remain unchanged",
        )
        self.assertRegex(
            self.script,
            r"\blocalStorage\s*\.\s*(?:getItem|setItem)\s*\(",
            "incident persistence should continue to use localStorage",
        )
        for name in ("openIncident", "saveIncidents", "loadIncidents", "renderLedger", "renderTimeline"):
            self.assertRegex(
                self.script,
                rf"\bfunction\s+{re.escape(name)}\s*\(",
                f"expected incident hook function {name} to remain present",
            )

    def test_distinct_alerts_are_not_collapsed_into_one_active_ticket(self) -> None:
        open_body = self.script.split("function openIncident", 1)[1].split("function advanceIncident", 1)[0]
        self.assertNotRegex(
            open_body,
            r"if\s*\(\s*activeIncident\s*\(\s*\)\s*\)\s*return\s+activeIncident",
            "distinct threshold crossings must create separate persistent investigation records",
        )

    def test_investigation_render_select_and_export_hooks_exist(self) -> None:
        names = {name.lower() for name in _function_names(self.script)}
        for name in (
            "renderincidentinvestigation",
            "selectincident",
            "downloadselectedincidentreport",
        ):
            self.assertIn(name, names, f"expected investigation hook {name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
