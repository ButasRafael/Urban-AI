import { Navigate } from 'react-router-dom'
import { useAuth } from '../auth/useAuth'
import type { ReactNode } from 'react'

export default function ProtectedRoute({
  role,
  children,
}: {
  role?: 'admin'|'authority'
  children: ReactNode
}) {
  const { user, loading } = useAuth()

  if (loading) {
    return <p style={{ textAlign:'center', marginTop:50 }}>Loading…</p>
  }
  if (!user) {
    return <Navigate to="/login" replace />
  }
  if (role && user.role !== role) {
    return <Navigate to="/" replace />
  }
  return <>{children}</>
}
