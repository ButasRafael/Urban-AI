import client from './client'
export interface User {
  username: string
  role: 'user'|'authority'|'admin'
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


export async function register(username: string, password: string) {
  await client.post("/auth/register", { username, password });
}

export async function logout() {
  await client.post("/auth/logout");
  localStorage.removeItem("accessToken");
  localStorage.removeItem("refreshToken");
}