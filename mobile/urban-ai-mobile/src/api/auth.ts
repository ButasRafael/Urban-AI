// src/api/auth.ts
import client from './client';
import AsyncStorage from '@react-native-async-storage/async-storage'

export async function register(username: string, password: string) {
  return client.post('/auth/register', { username, password });
}

export async function login(username: string, password: string) {
  const { data } = await client.post('/auth/login',
    new URLSearchParams({ username, password }).toString(),
    { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
  );
  await AsyncStorage.setItem('accessToken', data.access_token);
  await AsyncStorage.setItem('refreshToken', data.refresh_token);
  return data;
}

export async function logout() {
  await client.post('/auth/logout');
  await AsyncStorage.multiRemove(['accessToken','refreshToken']);
}
