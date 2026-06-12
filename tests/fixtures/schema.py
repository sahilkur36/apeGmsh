"""Single source of truth for schema versions used in test fixtures.

Tests stamping ``/meta/schema_version`` or ``/meta/opensees_schema_version``
in synthetic h5 fixtures must import from here so the next minor bump
is a one-file edit.  Per ADR 0023's two-version reader window,
``*_PRIOR_MINOR`` is the oldest version the current reader accepts.
"""
OPENSEES_CURRENT     = "2.18.0"  # ADR 0055 Phase 2 (/opensees/stages staged archival)
OPENSEES_PRIOR_MINOR = "2.17.0"  # ADR 0049 (node-pair zeroLength: inline_connectivity)
NEUTRAL_CURRENT      = "2.13.0"  # coupling host auto-scalers (cpl_k_auto/cpl_k_alpha/cpl_host/cpl_wcap columns)
NEUTRAL_PRIOR_MINOR  = "2.12.0"  # fork coupling knobs (CouplingControl: cpl_* columns on node_group + interpolation payloads)
