"""The Compose builder: a static page that writes SparkRing deployments for one checkout.

See docs/operations/compose-builder.md. export.py renders the page's data with
runtime/common/compose.py, engine.js turns that data and a site into files in
the browser, and verify.py compares the engine with compose.build.
"""
