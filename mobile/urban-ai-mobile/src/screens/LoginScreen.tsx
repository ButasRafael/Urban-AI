// src/screens/LoginScreen.tsx
//-----------------------------------------------------------
import React, { useState } from 'react';
import { SafeAreaView, View } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useTheme } from '@shopify/restyle';

import { RootStackParamList } from '../navigation/types';
import { Box, Text } from '../components/restylePrimitives';
import StyledButton   from '../components/StyledButton';
import StyledInput    from '../components/StyledInput';
import { login }      from '../api/auth';
import type { Theme } from '../theme';

//-----------------------------------------------------------
type Props = NativeStackScreenProps<RootStackParamList, 'Login'>;
//-----------------------------------------------------------

export default function LoginScreen({ navigation }: Props) {
  const theme = useTheme<Theme>();

  /* form state */
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);

  /* sign-in handler */
  const handleSignIn = async () => {
    setError(null);
    setLoading(true);
    try {
      await login(username, password);
      navigation.replace('Home');
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  /* --------------- UI --------------- */
  return (
    <SafeAreaView style={{ flex: 1 }}>
      <Box flex={1} bg="background" p="m" justifyContent="center">
        <Text variant="title" color="primary500" textAlign="center" mb="l">
          Welcome&nbsp;Back
        </Text>

        <View style={{ width: '100%' }}>
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
            title={loading ? 'Signing In…' : 'Sign In'}
            onPress={handleSignIn}
            disabled={loading}
          />
          <StyledButton
            title="No account? Register"
            variant="secondary"
            onPress={() => navigation.navigate('Register')}
          />
        </View>
      </Box>
    </SafeAreaView>
  );
}
