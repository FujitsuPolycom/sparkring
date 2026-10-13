"""Exercise a generated site locally and capture all pages in both themes and widths.

Requires the optional Playwright Python package and its Chromium browser.
Usage: python scripts/compose_builder/check_browser.py --site site --output DIRECTORY
The temporary server binds loopback only and shuts down when the check finishes.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
from threading import Thread
import zipfile

from playwright.sync_api import sync_playwright


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(args.site.resolve())))
    Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_port}'
    errors, external, shots, checks = [], [], [], []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context()
            context.on('page', lambda page: page.on('pageerror', lambda error: errors.append(str(error))))

            def route(request):
                if not request.request.url.startswith(base):
                    external.append(request.request.url)
                    request.abort()
                else:
                    request.continue_()

            context.route('**/*', route)
            page = context.new_page()
            for width in (1400, 375):
                page.set_viewport_size({'width': width, 'height': 1000 if width == 1400 else 812})
                for theme in ('dark', 'light'):
                    for name in ('index', 'builder', 'commands'):
                        response = page.goto(f'{base}/{name}.html', wait_until='networkidle')
                        assert response.status == 200
                        page.evaluate('(t) => {localStorage.setItem("sparkring-theme", t); document.documentElement.dataset.theme=t}', theme)
                        page.reload(wait_until='networkidle')
                        assert not errors, errors
                        if name == 'builder':
                            assert page.locator('#plan-command-0').count() == 1
                        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'Overflow: {name} {width} {theme}'
                        assert page.locator('#theme').inner_text() == ('Light theme' if theme == 'dark' else 'Dark theme')
                        page.locator('#theme').focus()
                        assert page.locator('#theme').evaluate('(e) => getComputedStyle(e).outlineStyle') == 'solid'
                        page.locator('#theme').evaluate('(e) => e.blur()')
                        path = args.output / f'{name}-{width}-{theme}.png'
                        page.screenshot(path=str(args.output / f'{name}-{width}-{theme}-viewport.png'))
                        page.screenshot(path=str(path), full_page=True)
                        shots.append(path.name)
            checks.append('12 page/theme/width combinations: no horizontal page overflow')
            page.set_viewport_size({'width': 1400, 'height': 1000})
            page.goto(base + '/index.html')
            page.get_by_role('button', name='Ring of eight', exact=True).click()
            page.get_by_role('button', name='Two halves', exact=True).click()
            page.locator('#load-example').click()
            page.wait_for_load_state('networkidle')
            assert page.locator('#spark-count').input_value() == '8'
            assert page.locator('[data-model]').count() == 2
            assert '--on 4,5,6,7' in page.locator('#plan-commands').inner_text()
            page.locator('#where-1').select_option('0,1,2,3')
            assert 'overlaps' in page.locator('#plan-problems').inner_text()
            assert page.locator('#plan-commands [data-copy]').count() == 0
            page.locator('#where-1').select_option('auto')
            assert '--on 4,5,6,7' in page.locator('#plan-commands').inner_text()
            checks.append('Landing example, explicit overlap blocking and Automatic resolution')
            page.locator('#setup-only').check()
            assert '--package-only' in page.locator('#plan-command-0').inner_text()
            assert page.locator('#plan-command-1').inner_text() == 'sudo sparkring setup'
            page.locator('#plan-form').select_option('installed')
            assert page.locator('#plan-command-0').inner_text() == 'sudo sparkring setup'
            checks.append('New and existing setup-only installations')
            page.locator('#setup-only').uncheck()
            page.locator('#plan-limit').fill('invalid')
            assert page.locator('#plan-commands [data-copy]').count() == 0
            page.locator('#plan-limit').fill('850Mbit')
            assert '--download-limit 850Mbit' in page.locator('#plan-command-0').inner_text()
            details = page.locator('[data-slot="0"] details')
            details.locator('summary').click()
            page.locator('#model-0-max_concurrency').fill('4')
            assert '--max-concurrency 4' in page.locator('#plan-command-0').inner_text()
            page.locator('#model-0-api_port').fill('9100')
            assert '--api-port 9100' in page.locator('#plan-command-0').inner_text()
            page.locator('#model-0-api_port').fill('bad')
            assert page.locator('#plan-commands [data-copy]').count() == 0
            page.locator('#model-0-api_port').fill('9100')
            checks.append('Download, serving and API settings validate and reach commands')
            page.locator('#advanced > summary').click()
            assert page.locator('#config').is_visible()
            assert page.evaluate('Boolean(window.jsyaml && window.jsyaml.load("name: test").name === "test")')
            page.locator('#mode-compose').click()
            page.locator('[data-step="sparks"] .toggle').click()
            page.locator('details.import > summary').click()
            yaml = page.evaluate('window.jsyaml.dump(JSON.parse(document.getElementById("data").textContent).profiles.find(p => p.nodes === 2).example_site)')
            page.locator('#site-text').fill(yaml)
            page.locator('#site-read').click()
            assert 'Read 2 Sparks' in page.locator('#site-read-result').inner_text()
            page.locator('#mode-compose').click()
            page.wait_for_function('!document.getElementById("save-zip").disabled')
            with page.expect_download() as download:
                page.locator('#save-zip').click()
            with tempfile.TemporaryDirectory() as work:
                path = Path(work) / 'deployment.zip'
                download.value.save_as(path)
                with zipfile.ZipFile(path) as archive:
                    assert any(name.endswith('rank0/compose.yaml') for name in archive.namelist())
                    assert any(name.endswith('README.txt') for name in archive.namelist())
            checks.append('Detailed editor imports YAML and downloads a valid Compose zip')
            page.set_viewport_size({'width': 375, 'height': 812})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Expanded editor overflow'
            page.screenshot(path=str(args.output / 'builder-375-light-expanded.png'), full_page=True)
            page.goto(base + '/?profile=qwen38-flash-next-tp2&mode=compose', wait_until='networkidle')
            assert '/builder.html?' in page.url
            assert page.locator('#advanced').get_attribute('open') is not None
            assert page.locator('#compose-panel').is_visible()
            checks.append('Previously published builder links still open the detailed editor')
            assert not errors, errors
            assert not external, external
            (args.output / 'browser-checks.json').write_text(json.dumps({'checks': checks, 'screenshots': shots,
                'browser': browser.version, 'page_errors': errors, 'external_requests': external}, indent=2), encoding='utf-8')
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    print(json.dumps({'checks': len(checks), 'screenshots': len(shots), 'errors': errors, 'external_requests': external}))


if __name__ == '__main__':
    main()
