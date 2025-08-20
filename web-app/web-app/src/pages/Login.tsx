import type React from 'react';
import { useState, type FormEvent, useEffect } from 'react';
import { login } from '../api/auth';
import { useAuth } from '../auth/useAuth';
import { Link, useNavigate } from 'react-router-dom';
import { notify } from '../lib/notify';

export default function Login() {
  const { setUser } = useAuth();
  const nav = useNavigate();

  const [username, setUsername]   = useState('');
  const [password, setPassword]   = useState('');
  const [remember, setRemember]   = useState(true);
  const [showPwd, setShowPwd]     = useState(false);
  const [peek, setPeek]           = useState(false);
  const [capsLock, setCapsLock]   = useState(false);
  const [busy, setBusy]           = useState(false);
  const [fieldErrs, setFieldErrs] = useState<{ username?: string; password?: string }>({});

  const disabled = busy || !username.trim() || !password;

  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('theme');
    if (saved === 'light' || saved === 'dark') return saved;
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    localStorage.setItem('theme', theme);
    window.dispatchEvent(new CustomEvent('themechange'));
  }, [theme]);
  const toggleTheme = () => setTheme(t => (t === 'dark' ? 'light' : 'dark'));

  useEffect(() => {
    document.querySelector<HTMLInputElement>('#username')?.focus();
  }, []);

  function validate() {
    const fe: typeof fieldErrs = {};
    if (username.trim().length < 3) fe.username = 'At least 3 characters';
    if (password.length < 6)        fe.password = 'At least 6 characters';
    setFieldErrs(fe);
    return Object.keys(fe).length === 0;
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (disabled) return;

    if (!validate()) {
      notify.error('Please fix the form', 'Check the highlighted fields.');
      return;
    }

    setBusy(true);
    try {
      const task = login(username.trim(), password).then(u => {
        setUser(u);
        if (!remember) localStorage.removeItem('refreshToken');
        setTimeout(() => {
          nav(u.role === 'admin' ? '/analytics' : '/map', { replace: true });
        }, 50);
        return u;
      });

      notify.promise(task, {
        loading: 'Signing in…',
        success: (u) => `Welcome back, ${u.username}!`,
        error: 'Invalid credentials. Please try again.',
      });
    } finally {
      setBusy(false);
    }
  }

  function handlePeekDown() {
    if (!showPwd) { setPeek(true); setShowPwd(true); }
  }
  function handlePeekUp() {
    if (peek) { setShowPwd(false); setPeek(false); }
  }

  return (
    <div className="auth-layout">
      {/* Left hero */}
      <aside className="auth-hero">
        <div className="auth-hero-inner">
          <div className="badge">Urban-AI</div>
          <h1>Keep Cluj clean & safe</h1>
          <p>
            Admin & authority console for monitoring uploads, mapping issues and
            chatting with the assistant.
          </p>
          <ul className="feature-list">
            <li>
              <svg viewBox="0 0 24 24" fill="none" aria-hidden>
                <path d="M20 6 9 17l-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              Real-time issue tracking
            </li>
            <li>
              <svg viewBox="0 0 24 24" fill="none" aria-hidden>
                <path d="M20 6 9 17l-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              Smart analytics & heatmaps
            </li>
            <li>
              <svg viewBox="0 0 24 24" fill="none" aria-hidden>
                <path d="M20 6 9 17l-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              Assistant-powered insights
            </li>
          </ul>
        </div>
        <div className="auth-hero-blobs" aria-hidden />
      </aside>

      {/* Right panel */}
      <main className="auth-main">
        <div className="auth-card" data-busy={busy}>
          {/* Theme toggle */}
          <button
            type="button"
            className="theme-toggle"
            onClick={toggleTheme}
            title={theme === 'dark' ? 'Switch to light' : 'Switch to dark'}
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? (
              <svg viewBox="0 0 24 24" aria-hidden>
                <circle cx="12" cy="12" r="4" stroke="currentColor" strokeWidth="2" />
                <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"
                      stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" aria-hidden>
                <path d="M21 12.79A9 9 0 1 1 11.21 3a7 7 0 0 0 9.79 9.79Z"
                      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            )}
          </button>

          <header className="auth-header">
            <div className="auth-logo">Urban AI</div>
            <p className="auth-sub">Sign in to your account</p>
            <span className="sr-only" aria-live="polite">{busy ? 'Signing in…' : ''}</span>
          </header>

          {/* Form */}
          <form onSubmit={handleSubmit} className="auth-form" aria-busy={busy} noValidate>
            {/* Username */}
            <div className="field">
              <label htmlFor="username">Username</label>
              <div className="with-icon">
                <span className="icon-left" aria-hidden>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                    <path d="M20 21a8 8 0 0 0-16 0" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    <circle cx="12" cy="7" r="4" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                </span>
                <input
                  id="username"
                  className={`input ${fieldErrs.username ? 'error' : ''}`}
                  placeholder="Enter your username"
                  autoCapitalize="none"
                  autoCorrect="off"
                  spellCheck={false}
                  value={username}
                  onChange={(e) => { setUsername(e.target.value); setFieldErrs(f => ({ ...f, username: '' })); }}
                  onBlur={() =>
                    setFieldErrs(f => ({ ...f, username: username.trim().length < 3 ? 'At least 3 characters' : '' }))
                  }
                  autoComplete="username"
                  aria-invalid={!!fieldErrs.username}
                />
              </div>
              {fieldErrs.username && <div className="field-error">{fieldErrs.username}</div>}
            </div>

            {/* Password */}
            <div className="field">
              <label htmlFor="password">Password</label>
              <div className="with-icon">
                <span className="icon-left" aria-hidden>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                    <rect x="3" y="11" width="18" height="10" rx="2" stroke="currentColor" strokeWidth="2"/>
                    <path d="M7 11V8a5 5 0 0 1 10 0v3" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                </span>
                <input
                  id="password"
                  className={`input ${fieldErrs.password ? 'error' : ''}`}
                  placeholder="Enter your password"
                  type={showPwd ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => { setPassword(e.target.value); setFieldErrs(f => ({ ...f, password: '' })); }}
                  onBlur={() =>
                    setFieldErrs(f => ({ ...f, password: password.length < 6 ? 'At least 6 characters' : '' }))
                  }
                  onKeyUp={(e: React.KeyboardEvent<HTMLInputElement>) =>
                      setCapsLock(e.getModifierState('CapsLock'))
                }
                  autoComplete="current-password"
                  aria-invalid={!!fieldErrs.password}
                />
                <button
                  type="button"
                  className="icon-right-btn"
                  onMouseDown={handlePeekDown}
                  onMouseUp={handlePeekUp}
                  onMouseLeave={handlePeekUp}
                  onClick={(e) => { if (peek) { e.preventDefault(); return; } setShowPwd(s => !s); }}
                  aria-label={showPwd ? 'Hide password' : 'Show password'}
                  title={showPwd ? 'Hide password' : 'Show password'}
                >
                  {showPwd ? (
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                      <path d="M17.94 17.94A10.94 10.94 0 0 1 12 20C7 20 2.73 16.11 1 12c.62-1.45 1.52-2.79 2.63-3.96M9.9 4.24A10.94 10.94 0 0 1 12 4c5 0 9.27 3.89 11 8a11.66 11.66 0 0 1-2.1 3.2M3 3l18 18"
                            stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    </svg>
                  ) : (
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8S1 12 1 12Z" stroke="currentColor" strokeWidth="2"/>
                      <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
                    </svg>
                  )}
                </button>
              </div>

              {capsLock && <div className="hint" id="caps-hint">Caps Lock is ON</div>}
              {fieldErrs.password && <div className="field-error">{fieldErrs.password}</div>}
            </div>

            <div className="row-between" style={{ marginTop: 2 }}>
              <label className="checkbox" style={{ gap: 12 }} role="switch" aria-checked={remember}>
                <input
                  className="switch-input"
                  type="checkbox"
                  checked={remember}
                  onChange={(e) => setRemember(e.target.checked)}
                />
                <span className="switch" aria-hidden />
                <span className="muted">Remember me</span>
              </label>
              <div style={{ display:'flex', gap:12 }}>
                <a className="muted" href="#" onClick={(e) => e.preventDefault()}>Forgot password?</a>
                <Link className="muted" to="/register">Create account</Link>
              </div>
            </div>

            <button className="btn btn-primary btn-block" disabled={disabled} aria-disabled={disabled}>
              {busy ? (<><span className="spinner" />Signing in…</>) : 'Sign in'}
            </button>
          </form>

          <footer className="auth-footer">
            <small className="muted">By signing in you agree to our terms & privacy.</small>
          </footer>
        </div>
      </main>
    </div>
  );
}
