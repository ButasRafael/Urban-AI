// src/api/client.ts
import axios, { AxiosError } from 'axios'
import AsyncStorage from '@react-native-async-storage/async-storage'
import { API_BASE } from '../config'
import { resetToLogin } from '../navigation/RootNavigation'

const client = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
})

// Attach access token
client.interceptors.request.use(async config => {
  const token = await AsyncStorage.getItem('accessToken')
  if (token) config.headers!['Authorization'] = `Bearer ${token}`
  return config
})

// Refresh on 401
client.interceptors.response.use(
  r => r,
  async (error: AxiosError) => {
    const original = error.config as any
    // if we got a 401 and we haven't already tried to refresh on this request
    if (error.response?.status === 401 && !original._retry) {
      original._retry = true
      try {
        const refreshToken = await AsyncStorage.getItem('refreshToken')
        const { data } = await axios.post(
          `${API_BASE}/auth/refresh`,
          { refresh_token: refreshToken },
          { headers: { 'Content-Type': 'application/json' } }
        )
        // store the brand-new tokens
        await AsyncStorage.setItem('accessToken', data.access_token)
        await AsyncStorage.setItem('refreshToken', data.refresh_token)
        // update the header and retry
        original.headers!['Authorization'] = `Bearer ${data.access_token}`
        return client(original)
      } catch (e) {
        // refresh also failed → force logout
        await AsyncStorage.multiRemove(['accessToken', 'refreshToken'])
        resetToLogin()
      
      }
    }
    return Promise.reject(error)
  }
)

export default client
