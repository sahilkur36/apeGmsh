"""Regression test: viewer SelectionState must log callback failures, not swallow."""
import logging

from apeGmsh.viewers.core.selection import SelectionState


def test_on_changed_failure_is_logged(caplog):
    sel = SelectionState()

    def bad_cb():
        raise RuntimeError("boom")

    sel.on_changed.append(bad_cb)
    with caplog.at_level(logging.ERROR, logger="apeGmsh.viewer.selection"):
        sel.pick((3, 1))

    assert any(
        "boom" in r.message or "boom" in str(r.exc_info)
        for r in caplog.records
    ), f"expected 'boom' in logs; got: {[r.message for r in caplog.records]}"
