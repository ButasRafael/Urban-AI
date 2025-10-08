import client from './client';
import AsyncStorage from '@react-native-async-storage/async-storage'

export async function register(username: string, email: string, password: string, role: string = 'user') {
  return client.post('/auth/register', { username, email, password, role }, {
    params: { platform: 'mobile' }
  });
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

export async function verifyEmail(token: string) {
  const { data } = await client.get(`/auth/verify-email?token=${token}`);
  return data;
}

export async function resendVerificationEmail(email: string) {
  const { data } = await client.post('/auth/resend-verification', null, {
    params: { email, platform: 'mobile' }
  });
  return data;
}

export async function forgotPassword(email: string) {
  const { data } = await client.post('/auth/forgot-password', null, {
    params: { email, platform: 'mobile' }
  });
  return data;
}

export async function resetPassword(token: string, new_password: string) {
  const { data } = await client.post('/auth/reset-password', {
    token,
    new_password
  });
  return data;
}

export const authService = {
  getToken: async () => {
    return await AsyncStorage.getItem('accessToken');
  }
};
