import React, { useEffect, useRef, useState } from 'react';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useTheme } from '@shopify/restyle';
import type { Theme } from '../theme';
import {
  Animated,
  Easing,
  View,
  TouchableWithoutFeedback,
  Keyboard,
} from 'react-native';
import { Feather } from '@expo/vector-icons';

import { RootStackParamList } from '../navigation/types';
import { Box, Text } from '../components/restylePrimitives';
import StyledButton from '../components/StyledButton';
import HeroHeader from '../components/ui/HeroHeader';
import InlineNotice from '../components/ui/InlineNotice';
import Badge from '../components/ui/Badge';
import { notify } from '../components/ui/Toast';
import { resendVerificationEmail, verifyEmail } from '../api/auth';

import Screen from '../components/ui/Screen';
import Card from '../components/ui/Card';

type Props = NativeStackScreenProps<RootStackParamList, 'VerifyEmail'>;

export default function VerifyEmailScreen({ navigation, route }: Props) {
  const theme = useTheme<Theme>();
  const { email, token } = route.params;
  const [isResending, setIsResending] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationSuccess, setVerificationSuccess] = useState(false);
  const hasVerified = useRef(false);

  const cardY = useRef(new Animated.Value(16)).current;
  const cardOpacity = useRef(new Animated.Value(0)).current;

  // Handle deep link verification
  useEffect(() => {
    if (token && !hasVerified.current) {
      hasVerified.current = true;
      handleTokenVerification(token);
    }
  }, [token]);

  const handleTokenVerification = async (verificationToken: string) => {
    setIsVerifying(true);
    try {
      await verifyEmail(verificationToken);
      setVerificationSuccess(true);
      notify.success('Email verificat!', 'Poți acum să te autentifici.');
      setTimeout(() => {
        navigation.navigate('Login');
      }, 2000);
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Token invalid sau expirat';
      notify.error('Eroare verificare', msg);
    } finally {
      setIsVerifying(false);
    }
  };

  useEffect(() => {
    Animated.parallel([
      Animated.timing(cardY, {
        toValue: 0,
        duration: 320,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
      Animated.timing(cardOpacity, {
        toValue: 1,
        duration: 260,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
    ]).start();
  }, [cardOpacity, cardY]);

  const handleResendVerification = async () => {
    if (!email) return;
    setIsResending(true);
    try {
      await resendVerificationEmail(email);
      notify.success('Email trimis', 'Verifică-ți inbox-ul pentru link-ul de verificare.');
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Eroare la trimiterea emailului';
      notify.error('Eroare', msg);
    } finally {
      setIsResending(false);
    }
  };

  // If verifying via deep link, show loading/success state
  if (token) {
    return (
      <Screen center>
        <Box alignItems="center" padding="xl">
          <Box
            width={80}
            height={80}
            borderRadius="lg"
            backgroundColor="primary100"
            borderColor="primary300"
            borderWidth={1}
            alignItems="center"
            justifyContent="center"
            marginBottom="l"
            style={theme.shadows.md}
          >
            <Feather
              name={isVerifying ? "mail" : verificationSuccess ? "check-circle" : "x-circle"}
              size={40}
              color={verificationSuccess ? theme.colors.success : isVerifying ? theme.colors.primary700 : theme.colors.error}
            />
          </Box>
          <Text variant="hero" color="text" textAlign="center" marginBottom="m">
            {isVerifying ? 'Se verifică...' : verificationSuccess ? 'Email verificat!' : 'Eroare verificare'}
          </Text>
          {verificationSuccess && (
            <Text variant="body" color="muted" textAlign="center">
              Vei fi redirecționat către pagina de autentificare...
            </Text>
          )}
        </Box>
      </Screen>
    );
  }

  return (
    <Screen
      scroll
      padded={false}
      edges={['left','right']}
      dismissKeyboardOnTap
      center
      maxWidth={640}
    >
      <TouchableWithoutFeedback onPress={Keyboard.dismiss} accessible={false}>
        <View>
          {/* Enhanced Hero Section */}
          <HeroHeader
            title="Urban AI"
            subtitle="Verificare Email"
            height={260}
            blobs
            rightSlot={
              <Box alignItems="center">
                <Box
                  width={64}
                  height={64}
                  borderRadius="lg"
                  style={{
                    backgroundColor: 'rgba(255,255,255,0.15)',
                    borderWidth: 1,
                    borderColor: 'rgba(255,255,255,0.25)',
                    ...theme.shadows.md,
                  }}
                  alignItems="center"
                  justifyContent="center"
                  marginBottom="s"
                >
                  <Feather name="mail" size={28} color="#fff" />
                </Box>
                <Badge variant="primary" size="sm" style={{ backgroundColor: 'rgba(255,255,255,0.2)' }}>
                  <Text style={{ color: 'white', fontSize: 11, fontWeight: '700' }}>Verificare</Text>
                </Badge>
              </Box>
            }
          />

          {/* Floating card */}
          <Animated.View
            style={{
              transform: [{ translateY: cardY }],
              opacity: cardOpacity,
              marginTop: -48,
              zIndex: 2,
            }}
          >
            <Box paddingHorizontal="l">
              <Card variant="elevated" gradientBorder padding="xl">
                {/* Enhanced Header */}
                <Box alignItems="center" marginBottom="l">
                  <Box
                    width={64}
                    height={64}
                    borderRadius="lg"
                    backgroundColor="primary100"
                    borderColor="primary300"
                    borderWidth={1}
                    alignItems="center"
                    justifyContent="center"
                    marginBottom="m"
                    style={theme.shadows.sm}
                  >
                    <Feather name="mail" size={32} color={theme.colors.primary700} />
                  </Box>

                  <Text variant="hero" color="text" textAlign="center" marginBottom="xs">
                    Verifică-ți emailul
                  </Text>

                  <Text variant="body" color="muted" textAlign="center" marginBottom="m">
                    Am trimis un email de verificare la
                  </Text>

                  <Box
                    backgroundColor="primary100"
                    paddingHorizontal="l"
                    paddingVertical="s"
                    borderRadius="m"
                    marginBottom="s"
                  >
                    <Text variant="body" color="primary700" style={{ fontWeight: '600' }}>
                      {email}
                    </Text>
                  </Box>

                  <Box flexDirection="row" alignItems="center" marginTop="s" style={{ gap: 4 }}>
                    <Badge variant="warning" size="sm" dot>
                      În așteptare
                    </Badge>
                  </Box>
                </Box>

                {/* Instructions */}
                <Box marginBottom="l">
                  <InlineNotice
                    variant="info"
                    title="Pași de urmat"
                    message="Deschide emailul și dă click pe link-ul de verificare pentru a-ți activa contul."
                    compact={false}
                  />
                </Box>

                {/* Resend Button */}
                <Box marginBottom="m">
                  <Text variant="caption" color="muted" textAlign="center" marginBottom="m">
                    Nu ai primit emailul?
                  </Text>
                  <StyledButton
                    title="Retrimite email de verificare"
                    onPress={handleResendVerification}
                    loading={isResending}
                    variant="tonal"
                    size="lg"
                    leftIconName="mail"
                    fullWidth
                    radius="l"
                    loadingText="Se trimite…"
                    showSpinnerOnly={false}
                  />
                </Box>

                {/* Divider */}
                <Box
                  flexDirection="row"
                  alignItems="center"
                  justifyContent="center"
                  marginBottom="m"
                >
                  <Box
                    height={1}
                    flex={1}
                    backgroundColor="border"
                  />
                  <Box
                    backgroundColor="card"
                    paddingHorizontal="m"
                    paddingVertical="xs"
                    borderRadius="lg"
                    borderWidth={1}
                    borderColor="border"
                  >
                    <Text variant="caption" color="muted" style={{ fontWeight: '600' }}>
                      sau
                    </Text>
                  </Box>
                  <Box
                    height={1}
                    flex={1}
                    backgroundColor="border"
                  />
                </Box>

                {/* Back to Login */}
                <Box alignItems="center">
                  <StyledButton
                    title="Înapoi la autentificare"
                    variant="tonal"
                    onPress={() => navigation.navigate('Login')}
                    leftIconName="log-in"
                    size="sm"
                  />
                </Box>
              </Card>
            </Box>
          </Animated.View>

          {/* Help Section */}
          <Box paddingHorizontal="l" marginTop="l" marginBottom="xl">
            <Box
              backgroundColor="card"
              borderRadius="lg"
              padding="l"
              borderWidth={1}
              borderColor="border"
            >
              <Box flexDirection="row" alignItems="center" marginBottom="s">
                <Feather name="info" size={20} color={theme.colors.primary500} />
                <Text variant="subtitle" color="text" marginLeft="s" style={{ fontWeight: '600' }}>
                  Verifică folderul spam
                </Text>
              </Box>
              <Text variant="caption" color="muted">
                Dacă nu găsești emailul în inbox, verifică și folderul de spam sau promotional.
              </Text>
            </Box>
          </Box>

          {/* bottom space */}
          <Box height={theme.spacing.xl} />
        </View>
      </TouchableWithoutFeedback>
    </Screen>
  );
}