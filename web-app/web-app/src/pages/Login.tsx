import { useState, type FormEvent } from 'react'
import { login } from '../api/auth'
import { useAuth } from '../auth/useAuth'
import { useNavigate } from 'react-router-dom'

export default function Login() {
  const { setUser } = useAuth()
  const nav = useNavigate()

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [err, setErr]       = useState('')
  const [busy, setBusy]     = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setErr('')
    setBusy(true)
    try {
      const u = await login(username.trim(), password)
      setUser(u)

      const target = u.role === 'admin' ? '/analytics' : '/map'
      nav(target, { replace: true })

    } catch (e: unknown) {
      console.error(e)
      setErr('Invalid credentials')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{
      display: 'flex',
      minHeight: '100vh',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'var(--surface-100)',
      padding: 'var(--xl)',
    }}>
      <div style={{
        background: 'var(--surface-0)',
        padding: 'var(--xl)',
        borderRadius: 'var(--radius-md)',
        width: '100%',
        maxWidth: '480px',
        boxShadow: 'var(--shadow-md)',
      }}>
        <div style={{ textAlign: 'center', marginBottom: 'var(--xl)' }}>
          <h1 style={{ 
            color: 'var(--primary-500)',
            fontSize: '2rem',
            fontWeight: 700,
            marginBottom: 'var(--s)'
          }}>
            Urban AI
          </h1>
          <p style={{ color: 'var(--secondary-700)' }}>
            Sign in to your account
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: 'var(--m)' }}>
            <label style={{
              display: 'block',
              marginBottom: 'var(--s)',
              fontWeight: 500,
              color: 'var(--secondary-700)'
            }}>
              Username
            </label>
            <input
              className="input"
              placeholder="Enter your username"
              autoFocus
              value={username}
              onChange={e => setUsername(e.target.value)}
              style={{ width: '100%' }}
            />
          </div>

          <div style={{ marginBottom: 'var(--l)' }}>
            <label style={{
              display: 'block',
              marginBottom: 'var(--s)',
              fontWeight: 500,
              color: 'var(--secondary-700)'
            }}>
              Password
            </label>
            <input
              className="input"
              placeholder="Enter your password"
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              style={{ width: '100%' }}
            />
          </div>

          {err && (
            <div style={{
              color: 'var(--error)',
              textAlign: 'center',
              marginBottom: 'var(--m)',
              padding: 'var(--s)',
              background: 'rgba(192, 57, 43, 0.1)',
              borderRadius: 'var(--radius-sm)'
            }}>
              {err}
            </div>
          )}

          <button
            className="btn btn-primary"
            style={{ width: '100%', padding: 'var(--m)' }}
            disabled={busy}
          >
            {busy ? 'Signing in...' : 'Sign in'}
          </button>
        </form>
      </div>
    </div>
  )
}
