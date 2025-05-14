// src/screens/RegisterScreen.tsx
//------------------------------------------------------------
import React, { useState } from 'react';
import { SafeAreaView } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useTheme } from '@shopify/restyle';

import { RootStackParamList } from '../navigation/types';
import { Box, Text } from '../components/restylePrimitives';
import StyledInput  from '../components/StyledInput';
import StyledButton from '../components/StyledButton';
import { register } from '../api/auth';
import type { Theme } from '../theme';

//------------------------------------------------------------
type Props = NativeStackScreenProps<RootStackParamList, 'Register'>;
//------------------------------------------------------------

export default function RegisterScreen({ navigation }: Props) {
  const theme = useTheme<Theme>();

  /* form state */
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);

  /* sign-up handler */
  const handleSignUp = async () => {
    setError(null);
    setLoading(true);
    try {
      await register(username, password);
      navigation.replace('Login');
    } catch (err: any) {
      // unwrap FastAPI / Pydantic detail
      const detail = err.response?.data?.detail;
      let message: string;
      if (Array.isArray(detail))    message = detail.map((e: any) => e.msg).join('\n');
      else if (typeof detail === 'string') message = detail;
      else                                message = err.message ?? 'Registration failed';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  /* ------------------- UI ------------------- */
  return (
    <SafeAreaView style={{ flex: 1 }}>
      <Box flex={1} bg="background" p="m" justifyContent="center">
        <Text variant="title" color="primary500" textAlign="center" mb="l">
          Create&nbsp;Account
        </Text>

        <Box width="100%">
          <StyledInput
            value={username}
            onChangeText={setUsername}
            placeholder="Username"
          />
          <StyledInput
            value={password}
            onChangeText={setPassword}
            placeholder="Password"
            secureTextEntry
          />

          {error && (
            <Text color="error" textAlign="center" mb="s">
              {error}
            </Text>
          )}

          <StyledButton
            title={loading ? 'Signing Up…' : 'Sign Up'}
            onPress={handleSignUp}
            disabled={loading}
          />
          <StyledButton
            title="Already have an account? Login"
            variant="secondary"
            onPress={() => navigation.navigate('Login')}
          />
        </Box>
      </Box>
    </SafeAreaView>
  );
}
