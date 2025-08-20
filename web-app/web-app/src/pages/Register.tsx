import type React from 'react';
import { useEffect, useState, type FormEvent } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { notify } from '../lib/notify'
import { register as apiRegister } from '../api/auth'
import { isAxiosError } from 'axios';
type Role = 'authority' | 'admin'

const PW_REGEX = /^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@$!%*#?&]{8,}$/

// helpers
function scorePassword(pw: string) {
  let s = 0;
  if (pw.length >= 8) s++;
  if (/[A-Z]/.test(pw)) s++;
  if (/[a-z]/.test(pw)) s++;
  if (/\d/.test(pw)) s++;
  if (/[^A-Za-z0-9]/.test(pw)) s++;
  return Math.min(s, 4);
}
const STRENGTH_LABELS = ['Too short', 'Weak', 'Fair', 'Good', 'Strong'];


export default function Register() {
  const nav = useNavigate()

  const [username, setUsername] = useState('')
  const [role, setRole] = useState<Role>('authority')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')

  const [showPwd, setShowPwd] = useState(false)
  const [peek, setPeek] = useState(false)
  const [capsLock, setCapsLock] = useState(false)

  const [busy, setBusy] = useState(false)
  const [fieldErrs, setFieldErrs] = useState<{ username?: string; password?: string; confirm?: string }>({})

  const disabled = busy || !username.trim() || !password || !confirm
  const score = scorePassword(password)
  const strengthClass = ['strength--weak','strength--weak','strength--fair','strength--good','strength--strong'][score]
  const strengthLabel = STRENGTH_LABELS[score]
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('theme')
    if (saved === 'light' || saved === 'dark') return saved
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    localStorage.setItem('theme', theme)
    window.dispatchEvent(new CustomEvent('themechange'))
  }, [theme])
  const toggleTheme = () => setTheme(t => (t === 'dark' ? 'light' : 'dark'))

  useEffect(() => {
    document.querySelector<HTMLInputElement>('#username')?.focus()
  }, [])

  function validate() {
    const fe: typeof fieldErrs = {}
    if (username.trim().length < 3) fe.username = 'At least 3 characters'
    if (!PW_REGEX.test(password)) fe.password = 'Min 8 chars, include letters & digits'
    if (confirm !== password) fe.confirm = 'Passwords do not match'
    setFieldErrs(fe)
    return Object.keys(fe).length === 0
  }

  function handlePeekDown() {
    if (!showPwd) { setPeek(true); setShowPwd(true) }
  }
  function handlePeekUp() {
    if (peek) { setShowPwd(false); setPeek(false) }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (disabled) return

    if (!validate()) {
      notify.error('Please fix the form', 'Check the highlighted fields.')
      return
    }

    setBusy(true)
    try {
      const task = apiRegister(username.trim(), password, role).then(() => {
        setTimeout(() => nav('/login', { replace: true }), 200)
      })

      notify.promise(task, {
        loading: 'Creating your account…',
        success: 'Account created. You can sign in now.',
        error: (err: unknown) => isAxiosError(err)
            ? err.response?.data?.detail
            : err instanceof Error
                ? err.message
                : 'Registration failed',
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth-layout">
      {/* Left hero */}
      <aside className="auth-hero">
        <div className="auth-hero-inner">
          <div className="badge">Urban-AI</div>
          <h1>Create your account</h1>
          <p>Only authorities and admins may register.</p>
          <ul className="feature-list">
            <li>
              <svg viewBox="0 0 24 24" fill="none" aria-hidden>
                <path d="M20 6 9 17l-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              Role-based access
            </li>
            <li>
              <svg viewBox="0 0 24 24" fill="none" aria-hidden>
                <path d="M20 6 9 17l-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              Strong password required
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
            <p className="auth-sub">Create your authority/admin account</p>
          </header>

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
                  placeholder="jane.doe"
                  autoCapitalize="none"
                  autoCorrect="off"
                  spellCheck={false}
                  value={username}
                  onChange={(e) => { setUsername(e.target.value); setFieldErrs(f => ({ ...f, username: '' })) }}
                  onBlur={() => setFieldErrs(f => ({ ...f, username: username.trim().length < 3 ? 'At least 3 characters' : '' }))}
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
                  placeholder="Create a strong password"
                  type={showPwd ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => { setPassword(e.target.value); setFieldErrs(f => ({ ...f, password: '' })) }}
                  onBlur={() => setFieldErrs(f => ({ ...f, password: PW_REGEX.test(password) ? '' : 'Min 8 chars, include letters & digits' }))}
                  onKeyUp={(e: React.KeyboardEvent<HTMLInputElement>) =>
                      setCapsLock(e.getModifierState('CapsLock'))
                }
                  autoComplete="new-password"
                  aria-invalid={!!fieldErrs.password}
                />
                <button
                  type="button"
                  className="icon-right-btn"
                  onMouseDown={handlePeekDown}
                  onMouseUp={handlePeekUp}
                  onMouseLeave={handlePeekUp}
                  onClick={(e) => { if (peek) { e.preventDefault(); return } setShowPwd(s => !s) }}
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
              {capsLock && <div className="hint">Caps Lock is ON</div>}
              <div className={`strength ${strengthClass}`} aria-label={`Password strength: ${strengthLabel}`}>
                {Array.from({ length: 4 }).map((_, i) => (
                    <div key={i} className={`bar ${i < score ? 'active' : ''}`} />
                    ))}
                    <span className="strength-label">{strengthLabel}</span>
                    </div>
              {fieldErrs.password && <div className="field-error">{fieldErrs.password}</div>}
            </div>

            {/* Confirm password */}
            <div className="field">
              <label htmlFor="confirm">Confirm password</label>
              <div className="with-icon">
                <span className="icon-left" aria-hidden>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                    <path d="M12 3l8 4v5c0 5-3.4 8-8 9-4.6-1-8-4-8-9V7l8-4Z" stroke="currentColor" strokeWidth="2"/>
                    <path d="m9 12 2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </span>
                <input
                  id="confirm"
                  className={`input ${fieldErrs.confirm ? 'error' : ''}`}
                  placeholder="Repeat your password"
                  type="password"
                  value={confirm}
                  onChange={(e) => { setConfirm(e.target.value); setFieldErrs(f => ({ ...f, confirm: '' })) }}
                  onBlur={() => setFieldErrs(f => ({ ...f, confirm: confirm !== password ? 'Passwords do not match' : '' }))}
                  autoComplete="new-password"
                  aria-invalid={!!fieldErrs.confirm}
                />
              </div>
              {fieldErrs.confirm && <div className="field-error">{fieldErrs.confirm}</div>}
            </div>

            {/* Role */}
            <div className="field">
              <label>Role</label>
              <div className="radio-group">
                {(['authority','admin'] as Role[]).map((r) => (
                  <label key={r} className="radio-pill" data-checked={role === r}>
                    <input
                      type="radio"
                      name="role"
                      value={r}
                      checked={role === r}
                      onChange={() => setRole(r)}
                    />
                    {r === 'authority' ? 'Authority' : 'Admin'}
                  </label>
                ))}
              </div>
            </div>

            <div className="row-between" style={{ marginTop: 2 }}>
              <span className="muted">Already registered?</span>
              <Link className="muted" to="/login">Sign in</Link>
            </div>

            <button className="btn btn-primary btn-block" disabled={disabled} aria-disabled={disabled} aria-busy={busy}>
              {busy ? (<><span className="spinner" />Creating…</>) : 'Create account'}
            </button>
          </form>

          <footer className="auth-footer">
            <small className="muted">Only authorized personnel may register.</small>
          </footer>
        </div>
      </main>
    </div>
  )
}
