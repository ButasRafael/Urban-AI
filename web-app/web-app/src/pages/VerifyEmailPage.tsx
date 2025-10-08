import { useEffect, useState, useRef } from 'react';
import { Link, useSearchParams, useNavigate } from 'react-router-dom';
import { notify } from '../lib/notify';
import { verifyEmail, resendVerificationEmail } from '../api/auth';
import { isAxiosError } from 'axios';

export default function VerifyEmailPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const token = searchParams.get('token') || '';
  const email = searchParams.get('email') || '';
  const [busy, setBusy] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationSuccess, setVerificationSuccess] = useState(false);
  const [verificationError, setVerificationError] = useState('');
  const hasVerified = useRef(false);

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

  // Auto-verify if token is present in URL
  useEffect(() => {
    if (token && !hasVerified.current) {
      hasVerified.current = true;
      handleTokenVerification();
    }
  }, [token]);

  async function handleTokenVerification() {
    setIsVerifying(true);
    setVerificationError('');
    try {
      const result = await verifyEmail(token);
      setVerificationSuccess(true);
      notify.success('Email verified!', result.message || 'You can now sign in.');
      setTimeout(() => {
        navigate('/login');
      }, 2000);
    } catch (err) {
      const msg = isAxiosError(err) ? err.response?.data?.detail : 'Invalid or expired token';
      setVerificationError(msg);
      notify.error('Verification failed', msg);
    } finally {
      setIsVerifying(false);
    }
  }

  async function handleResend() {
    if (!email) {
      notify.error('No email provided', 'Please return to the registration page.');
      return;
    }

    setBusy(true);
    try {
      await resendVerificationEmail(email);
      notify.success('Email sent!', 'Check your inbox for the verification link.');
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
          <h1>Check your email</h1>
          <p>We've sent a verification link to your email address.</p>
          <ul className="feature-list">
            <li>
              <svg viewBox="0 0 24 24" fill="none" aria-hidden>
                <path d="M20 6 9 17l-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              Secure verification
            </li>
            <li>
              <svg viewBox="0 0 24 24" fill="none" aria-hidden>
                <path d="M20 6 9 17l-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              Quick activation
            </li>
            <li>
              <svg viewBox="0 0 24 24" fill="none" aria-hidden>
                <path d="M20 6 9 17l-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              24/7 support
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
            <p className="auth-sub">
              {isVerifying ? 'Verifying...' : verificationSuccess ? 'Email verified!' : verificationError ? 'Verification failed' : 'Email verification pending'}
            </p>
          </header>

          <div className="auth-form">
            {/* Show verification status if token is present */}
            {token ? (
              <div style={{ textAlign: 'center', marginBottom: 24 }}>
                <div style={{
                  width: 80,
                  height: 80,
                  margin: '0 auto 16px',
                  borderRadius: 12,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  backgroundColor: verificationSuccess ? '#d4edda' : verificationError ? '#f8d7da' : '#d1ecf1',
                  border: `2px solid ${verificationSuccess ? '#28a745' : verificationError ? '#dc3545' : '#17a2b8'}`
                }}>
                  {isVerifying ? (
                    <span className="spinner" style={{ width: 40, height: 40 }} />
                  ) : verificationSuccess ? (
                    <svg viewBox="0 0 24 24" width="40" height="40" fill="none">
                      <path d="M20 6 9 17l-5-5" stroke="#28a745" strokeWidth="3" strokeLinecap="round" />
                    </svg>
                  ) : (
                    <svg viewBox="0 0 24 24" width="40" height="40" fill="none">
                      <path d="M6 6l12 12M18 6L6 18" stroke="#dc3545" strokeWidth="3" strokeLinecap="round" />
                    </svg>
                  )}
                </div>

                <h3 style={{ fontSize: 20, fontWeight: 600, marginBottom: 8 }}>
                  {isVerifying ? 'Verifying your email...' : verificationSuccess ? 'Email verified!' : 'Verification failed'}
                </h3>

                <p className="muted" style={{ fontSize: 14 }}>
                  {isVerifying ? 'Please wait while we verify your email address.' : verificationSuccess ? 'Redirecting you to login...' : verificationError || 'Invalid or expired verification token.'}
                </p>

                {verificationError && (
                  <div style={{ marginTop: 24 }}>
                    <Link to="/login" className="btn btn-primary btn-block">
                      Go to sign in
                    </Link>
                  </div>
                )}
              </div>
            ) : (
              <>
                {/* Email verification message */}
                <div style={{ textAlign: 'center', marginBottom: 24 }}>
                  <h3 style={{ fontSize: 20, fontWeight: 600, marginBottom: 8 }}>Verify your email</h3>
                  <p className="muted" style={{ fontSize: 14, marginBottom: 16 }}>
                    We've sent a verification link to
                  </p>
                  {email && (
                    <div style={{
                      padding: '8px 16px',
                      fontSize: 14,
                      fontWeight: 500,
                      marginBottom: 16
                    }}>
                      {email}
                    </div>
                  )}
                  <p className="muted" style={{ fontSize: 14 }}>
                    Click the link in the email to activate your account.
                  </p>
                </div>

                {/* Info box */}
                <div style={{
                  padding: 16,
                  backgroundColor: 'var(--surface-200, #f5f5f5)',
                  borderRadius: 8,
                  marginBottom: 24
                }}>
                  <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 4 }}>
                    📧 Check your spam folder
                  </div>
                  <div className="muted" style={{ fontSize: 13 }}>
                    If you don't see the email in your inbox, check your spam or promotional folders.
                  </div>
                </div>

                {/* Resend button */}
                <div style={{ marginBottom: 16 }}>
                  <p className="muted" style={{ fontSize: 14, textAlign: 'center', marginBottom: 12 }}>
                    Didn't receive the email?
                  </p>
                  <button
                    className="btn btn-primary btn-block"
                    onClick={handleResend}
                    disabled={busy || !email}
                  >
                    {busy ? (
                      <>
                        <span className="spinner" />
                        Sending…
                      </>
                    ) : (
                      'Resend verification email'
                    )}
                  </button>
                </div>

                {/* Divider */}
                <div className="row-between" style={{ marginTop: 24, marginBottom: 16 }}>
                  <hr style={{ flex: 1, border: 'none', borderTop: '1px solid var(--border)' }} />
                  <span className="muted" style={{ padding: '0 12px', fontSize: 13 }}>or</span>
                  <hr style={{ flex: 1, border: 'none', borderTop: '1px solid var(--border)' }} />
                </div>

                {/* Back to login */}
                <div style={{ textAlign: 'center' }}>
                  <Link className="muted" to="/login" style={{ fontSize: 14 }}>
                    ← Back to sign in
                  </Link>
                </div>
              </>
            )}
          </div>

          <footer className="auth-footer">
            <small className="muted">The verification link expires in 24 hours.</small>
          </footer>
        </div>
      </main>
    </div>
  );
}
