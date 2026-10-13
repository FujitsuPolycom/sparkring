try {
  const theme = localStorage.getItem('sparkring-theme');
  if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme;
} catch (_) { /* Storage is optional; dark is the reference theme. */ }
