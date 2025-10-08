import client from './client'

export interface User {
  username: string
  email: string
  email_verified: boolean
  role: 'user' | 'authority' | 'admin'
}

export async function login(username: string, password: string): Promise<User> {
  const form = new URLSearchParams({ username, password }).toString()
  const { data: tok } = await client.post<{
    access_token: string
    refresh_token: string
  }>(
    '/auth/login',
    form,
    { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
  )

  localStorage.setItem('accessToken', tok.access_token)
  localStorage.setItem('refreshToken', tok.refresh_token)

  const { data: me } = await client.get<User>('/auth/me')
  return me
}

export async function register(
  username: string,
  email: string,
  password: string,
  role: 'authority' | 'admin'
) {
  await client.post('/auth/register', { username, email, password, role }, {
    params: { platform: 'web' }
  })
}

export async function logout() {
  await client.post('/auth/logout')
  localStorage.removeItem('accessToken')
  localStorage.removeItem('refreshToken')
}

export async function verifyEmail(token: string) {
  const { data } = await client.get(`/auth/verify-email?token=${token}`)
  return data
}

export async function resendVerificationEmail(email: string) {
  const { data } = await client.post('/auth/resend-verification', null, {
    params: { email, platform: 'web' }
  })
  return data
}

export async function forgotPassword(email: string) {
  const { data } = await client.post('/auth/forgot-password', null, {
    params: { email, platform: 'web' }
  })
  return data
}

export async function resetPassword(token: string, new_password: string) {
  const { data } = await client.post('/auth/reset-password', {
    token,
    new_password
  })
  return data
}
