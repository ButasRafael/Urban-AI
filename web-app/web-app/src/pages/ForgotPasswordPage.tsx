import { useEffect, useState, type FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { notify } from '../lib/notify';
import { forgotPassword } from '../api/auth';
import { isAxiosError } from 'axios';

export default function ForgotPasswordPage() {
  const nav = useNavigate();
  const [email, setEmail] = useState('');
  const [busy, setBusy] = useState(false);
  const [fieldErr, setFieldErr] = useState('');

  const disabled = busy || !email.trim();

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
    document.querySelector<HTMLInputElement>('#email')?.focus();
  }, []);

  function validate() {
    if (!email.includes('@') || email.length < 5) {
      setFieldErr('Invalid email address');
      return false;
    }
    setFieldErr('');
    return true;
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (disabled) return;

    if (!validate()) {
      notify.error('Please fix the form', 'Check the highlighted field.');
      return;
    }

    setBusy(true);
    try {
      await forgotPassword(email.trim().toLowerCase());
      notify.success('Email sent!', 'Check your inbox for password reset instructions.');
      setTimeout(() => nav('/login', { replace: true }), 2000);
    } catch (err) {
      notify.error('Failed to send email', isAxiosError(err) ? err.response?.data?.detail : 'Please try again.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-layout">
      {/* Left hero */}
      <aside className="auth-hero">
        <div className="auth-hero-inner">
          <div className="badge">Urban-AI</div>
          <h1>Reset your password</h1>
          <p>
            Enter your email address and we'll send you instructions to reset your password.
          </p>
          <ul className="feature-list">
            <li>
              <svg viewBox="0 0 24 24" fill="none" aria-hidden>
                <path d="M20 6 9 17l-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              Secure reset link
            </li>
            <li>
              <svg viewBox="0 0 24 24" fill="none" aria-hidden>
                <path d="M20 6 9 17l-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              1-hour validity
            </li>
            <li>
              <svg viewBox="0 0 24 24" fill="none" aria-hidden>
                <path d="M20 6 9 17l-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              Single-use token
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
            <p className="auth-sub">Forgot your password?</p>
          </header>

          <form onSubmit={handleSubmit} className="auth-form" aria-busy={busy} noValidate>
            {/* Password reset message */}
            <div style={{ textAlign: 'center', marginBottom: 24 }}>
              <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>Password Reset</h3>
              <p className="muted" style={{ fontSize: 14 }}>
                We'll send you reset instructions
              </p>
            </div>

            {/* Email field */}
            <div className="field">
              <label htmlFor="email">Email address</label>
              <div className="with-icon">
                <span className="icon-left" aria-hidden>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                    <rect x="2" y="4" width="20" height="16" rx="2" stroke="currentColor" strokeWidth="2"/>
                    <path d="m2 7 10 7 10-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </span>
                <input
                  id="email"
                  type="email"
                  className={`input ${fieldErr ? 'error' : ''}`}
                  placeholder="your.email@example.com"
                  autoCapitalize="none"
                  autoCorrect="off"
                  spellCheck={false}
                  value={email}
                  onChange={(e) => { setEmail(e.target.value); setFieldErr(''); }}
                  onBlur={() => {
                    if (email && (!email.includes('@') || email.length < 5)) {
                      setFieldErr('Invalid email address');
                    }
                  }}
                  autoComplete="email"
                  aria-invalid={!!fieldErr}
                />
              </div>
              {fieldErr && <div className="field-error">{fieldErr}</div>}
            </div>

            {/* Info box */}
            <div style={{
              padding: 12,
              backgroundColor: 'var(--surface-200, #f5f5f5)',
              borderRadius: 8,
              marginBottom: 16,
              fontSize: 13
            }}>
              💡 The reset link expires in 1 hour and can only be used once.
            </div>

            {/* Submit button */}
            <button className="btn btn-primary btn-block" disabled={disabled} aria-disabled={disabled}>
              {busy ? (
                <>
                  <span className="spinner" />
                  Sending…
                </>
              ) : (
                'Send reset instructions'
              )}
            </button>

            {/* Divider */}
            <div className="row-between" style={{ marginTop: 20, marginBottom: 16 }}>
              <hr style={{ flex: 1, border: 'none', borderTop: '1px solid var(--border)' }} />
              <span className="muted" style={{ padding: '0 12px', fontSize: 13 }}>or</span>
              <hr style={{ flex: 1, border: 'none', borderTop: '1px solid var(--border)' }} />
            </div>

            {/* Back to login */}
            <div style={{ textAlign: 'center' }}>
              <span className="muted" style={{ fontSize: 14 }}>Remembered your password? </span>
              <Link className="muted" to="/login" style={{ fontSize: 14 }}>
                Sign in
              </Link>
            </div>
          </form>

          <footer className="auth-footer">
            <small className="muted">Only verified accounts can reset passwords.</small>
          </footer>
        </div>
      </main>
    </div>
  );
}