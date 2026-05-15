# Docs CLAUDE.md

## Building Documentation

```bash
cd docs && source ~/.venvs/deep/bin/activate && python buildme.py
```

The build output goes to `~/Workspace/sites/blog/public/deepdiff/<version>/` (configured via `docs/.env`).
To verify changes, check the generated HTML, e.g. `~/Workspace/sites/blog/public/deepdiff/8.7.0/faq.html`.

The Sphinx doctree cache is stored in `/tmp/sphinx_doctree`. The build script clears it each run. If you get permission errors on that directory, ask the user to `rm -rf /tmp/sphinx_doctree` first.

## Theme

Uses the **Furo** Sphinx theme. Key customizations:

- **Font**: Open Sans, loaded via `_static/custom.css` and set in `conf.py` via `light_css_variables` / `dark_css_variables`
- **Footer**: Custom `_templates/page.html` overrides the default footer to remove Sphinx/Furo credit while keeping the copyright notice
- **GA4**: Google Analytics tag (`G-KVVHD37BKD`) is injected via `_templates/page.html` in the `extrahead` block
- **Pygments**: Uses Furo's default syntax highlighting (no explicit `pygments_style` set)

## Symlinked Docstrings

Some RST files in `docs/` (e.g., `diff_doc.rst`, `deephash_doc.rst`, `search_doc.rst`) are symlinks to `deepdiff/docstrings/`. The files need to exist in both places:

- **`deepdiff/docstrings/`** — So they're included in the generated wheel. `flit_core` (our build system) only packages files under the `deepdiff/` directory, and these are loaded at runtime by `get_doc()` in `helper.py` to serve as Python docstrings.
- **`docs/`** — So Sphinx can find and build them as documentation pages.

These files have a `:orphan:` directive on line 1 (needed by Sphinx to suppress toctree warnings). `get_doc()` strips it at runtime so it doesn't appear in the Python docstrings.

## File Structure

- `conf.py` — Sphinx configuration
- `buildme.py` — Build script (reads `.env` for `BUILD_PATH` and `DOC_VERSION`)
- `_templates/page.html` — Extends `furo/page.html` for GA4 and custom footer
- `_static/custom.css` — Loads Open Sans font from Google Fonts
