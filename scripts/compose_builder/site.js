(() => {
  const button = document.getElementById('theme');
  const label = () => { button.textContent = document.documentElement.dataset.theme === 'light' ? 'Dark theme' : 'Light theme'; };
  label();
  button.addEventListener('click', () => {
    const theme = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('sparkring-theme', theme); } catch (_) { /* Optional. */ }
    label();
  });
  document.addEventListener('click', async event => {
    const button = event.target.closest('[data-copy]');
    if (!button) return;
    const source = document.getElementById(button.dataset.copy);
    try { await navigator.clipboard.writeText(source.textContent); button.textContent = 'Copied'; }
    catch (_) {
      const range = document.createRange(); range.selectNodeContents(source);
      const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
      button.textContent = 'Selected — copy with your keyboard';
    }
  });
})();
