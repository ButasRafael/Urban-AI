import React, { useRef, useEffect, useMemo, useState } from 'react';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useTheme } from '@shopify/restyle';
import type { Theme } from '../theme';
import {
  TextInput as RNTextInput,
  Animated,
  Easing,
  Switch,
  View,
  TouchableWithoutFeedback,
  Keyboard,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Feather } from '@expo/vector-icons';

import { useForm, Controller, FieldErrors } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';

import { RootStackParamList } from '../navigation/types';
import { Box, Text } from '../components/restylePrimitives';
import StyledInput from '../components/StyledInput';
import StyledButton from '../components/StyledButton';
import HeroHeader from '../components/ui/HeroHeader';
import InlineNotice from '../components/ui/InlineNotice';
import Badge from '../components/ui/Badge';
import StatCard from '../components/ui/StatCard';
import FeatureCard from '../components/ui/FeatureCard';
import { login, resendVerificationEmail } from '../api/auth';
import { notify } from '../components/ui/Toast';

import Screen from '../components/ui/Screen';
import Card from '../components/ui/Card';

type Props = NativeStackScreenProps<RootStackParamList, 'Login'>;

const REMEMBER_KEY = 'auth:rememberUsername';

const schema = z.object({
  username: z.string().min(3, 'Minim 3 caractere'),
  password: z.string().min(8, 'Minim 8 caractere'),
  remember: z.boolean(),
});
type FormValues = z.infer<typeof schema>;

export default function LoginScreen({ navigation }: Props) {
  const theme = useTheme<Theme>();
  const [unverifiedEmail, setUnverifiedEmail] = useState<string | null>(null);
  const [isResending, setIsResending] = useState(false);

  const {
    control,
    handleSubmit,
    setValue,
    getValues,
    formState: { errors, isValid, isSubmitting, isSubmitted },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    mode: 'onChange',
    defaultValues: { username: '', password: '', remember: true },
  });

  const passwordRef = useRef<RNTextInput>(null);
  const cardY = useRef(new Animated.Value(16)).current;
  const cardOpacity = useRef(new Animated.Value(0)).current;
  const shakeX = useRef(new Animated.Value(0)).current;

  const lastErrorRef = useRef<string | null>(null);

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

  useEffect(() => {
    (async () => {
      const saved = await AsyncStorage.getItem(REMEMBER_KEY);
      if (saved) setValue('username', saved, { shouldValidate: true });
    })();
  }, [setValue]);

  const triggerShake = () => {
    shakeX.setValue(0);
    Animated.sequence([
      Animated.timing(shakeX, { toValue: 1, duration: 60, useNativeDriver: true }),
      Animated.timing(shakeX, { toValue: -1, duration: 60, useNativeDriver: true }),
      Animated.timing(shakeX, { toValue: 1, duration: 50, useNativeDriver: true }),
      Animated.timing(shakeX, { toValue: 0, duration: 40, useNativeDriver: true }),
    ]).start();
  };

  const canSubmit = useMemo(() => isValid && !isSubmitting, [isValid, isSubmitting]);

  const handleResendVerification = async () => {
    if (!unverifiedEmail) return;

    setIsResending(true);
    try {
      await resendVerificationEmail(unverifiedEmail);
      notify.success('Email trimis', 'Verifică-ți inbox-ul pentru link-ul de verificare.');
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Eroare la trimiterea emailului';
      notify.error('Eroare', msg);
    } finally {
      setIsResending(false);
    }
  };

  const onSubmit = async (values: FormValues) => {
    try {
      setUnverifiedEmail(null);
      await login(values.username.trim(), values.password);
      if (values.remember) {
        await AsyncStorage.setItem(REMEMBER_KEY, values.username.trim());
      } else {
        await AsyncStorage.removeItem(REMEMBER_KEY);
      }
      notify.success('Autentificare reușită');
      navigation.replace('Home');
    } catch (err: any) {
      const status = err?.response?.status;
      const msg = err?.response?.data?.detail || err?.message || 'Autentificare eșuată';

      // Check if error is 403 (email not verified)
      if (status === 403 && msg.toLowerCase().includes('email')) {
        setUnverifiedEmail(values.username.trim());
        if (lastErrorRef.current !== msg) {
          lastErrorRef.current = msg;
          triggerShake();
        }
        notify.error('Email neverificat', msg);
      } else {
        if (lastErrorRef.current !== msg) {
          lastErrorRef.current = msg;
          triggerShake();
        }
        notify.error(msg);
      }
    }
  };

  const onInvalid = (errs: FieldErrors<FormValues>) => {
    const first =
      (errs.username?.message as string | undefined) ??
      (errs.password?.message as string | undefined) ??
      'Te rugăm să verifici datele.';
    triggerShake();
    notify.info(first);
  };

  const topError =
    isSubmitted ? (errors.username?.message ?? errors.password?.message) : undefined;

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
            subtitle="Platforma inteligentă pentru sesizări urbane"
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
                  <Feather name="zap" size={28} color="#fff" />
                </Box>
                <Badge variant="primary" size="sm" style={{ backgroundColor: 'rgba(255,255,255,0.2)' }}>
                  <Text style={{ color: 'white', fontSize: 11, fontWeight: '700' }}>AI Powered</Text>
                </Badge>
              </Box>
            }
          />

          {/* Quick Stats Row (lift a bit less since header is shorter) */}
          <Box
            paddingHorizontal="l"
            style={{ marginTop: -48, zIndex: 2 }} // ⬅️ was -60
            marginBottom="l"
          >
            <Box flexDirection="row" style={{ gap: 8 }}>
              <StatCard value="500+" label="Sesizări" style={{ flex: 1 }} />
              <StatCard value="98%" label="Acuratețe AI" style={{ flex: 1 }} />
              <StatCard value="24h" label="Răspuns" style={{ flex: 1 }} />
            </Box>
          </Box>

          {/* Floating auth card */}
          <Animated.View
            style={{
              transform: [
                { translateY: cardY },
                {
                  translateX: shakeX.interpolate({
                    inputRange: [-1, 0, 1],
                    outputRange: [-6, 0, 6],
                  }),
                },
              ],
              opacity: cardOpacity,
            }}
          >
            <Box paddingHorizontal="l">
              <Card variant="elevated" gradientBorder padding="xl">
                {/* Enhanced Header */}
                <Box alignItems="center" marginBottom="l">
                  <Box
                    width={48}
                    height={48}
                    borderRadius="lg"
                    backgroundColor="primary100"
                    borderColor="primary300"
                    borderWidth={1}
                    alignItems="center"
                    justifyContent="center"
                    marginBottom="m"
                    style={theme.shadows.sm}
                  >
                    <Feather name="user" size={24} color={theme.colors.primary700} />
                  </Box>
                  
                  <Text variant="hero" color="text" textAlign="center" marginBottom="xs">
                    Bun venit înapoi
                  </Text>
                  
                  <Text variant="body" color="muted" textAlign="center">
                    Conectează-te la contul tău Urban AI
                  </Text>
                  
                  <Box flexDirection="row" alignItems="center" marginTop="s" style={{ gap: 4 }}>
                    <Badge variant="success" size="sm" dot>
                      Securizat
                    </Badge>
                    <Badge variant="primary" size="sm">
                      Rapid
                    </Badge>
                  </Box>
                </Box>

                {topError ? (
                  <Box marginBottom="m">
                    <InlineNotice
                      variant="error"
                      title="Verifică datele"
                      message={topError}
                      compact
                    />
                  </Box>
                ) : null}

                {/* Email Verification Notice */}
                {unverifiedEmail ? (
                  <Box marginBottom="m">
                    <InlineNotice
                      variant="warning"
                      title="Email neverificat"
                      message="Verifică-ți emailul pentru a activa contul."
                      compact
                    />
                    <Box marginTop="s" alignItems="center">
                      <StyledButton
                        title="Retrimite email de verificare"
                        variant="tonal"
                        onPress={handleResendVerification}
                        loading={isResending}
                        size="sm"
                        leftIconName="mail"
                      />
                    </Box>
                  </Box>
                ) : null}

                {/* Enhanced Username/Email Input */}
                <Box marginBottom="s">
                  <Text variant="label" color="text" marginBottom="xs" style={{ fontWeight: '600' }}>
                    Email sau nume utilizator
                  </Text>
                  <Controller
                    name="username"
                    control={control}
                    render={({ field: { value, onChange, onBlur } }) => (
                      <StyledInput
                        leftIcon="mail"
                        value={value}
                        onChangeText={onChange}
                        placeholder="Email sau nume de utilizator"
                        returnKeyType="next"
                        onBlur={onBlur}
                        onSubmitEditing={() => passwordRef.current?.focus()}
                        autoComplete="username"
                        textContentType="username"
                        autoCapitalize="none"
                        allowClear
                        errorText={errors.username?.message}
                        helperText={!errors.username?.message ? "Email sau nume utilizator" : undefined}
                      />
                    )}
                  />
                </Box>

                {/* Enhanced Password Input */}
                <Box marginBottom="l">
                  <Text variant="label" color="text" marginBottom="xs" style={{ fontWeight: '600' }}>
                    Parolă
                  </Text>
                  <Controller
                    name="password"
                    control={control}
                    render={({ field: { value, onChange, onBlur } }) => (
                      <StyledInput
                        ref={passwordRef}
                        leftIcon="lock"
                        value={value}
                        onChangeText={onChange}
                        placeholder="Introdu parola"
                        secureTextEntry
                        passwordToggle
                        returnKeyType="go"
                        onBlur={onBlur}
                        onSubmitEditing={handleSubmit(onSubmit, onInvalid)}
                        autoComplete="password"
                        textContentType="password"
                        errorText={errors.password?.message}
                        helperText={!errors.password?.message ? "Minim 8 caractere" : undefined}
                      />
                    )}
                  />
                </Box>

                {/* Enhanced Remember Toggle */}
                <Box 
                  flexDirection="row" 
                  alignItems="center" 
                  justifyContent="space-between" 
                  marginBottom="m"
                  padding="m"
                  backgroundColor="surface50"
                  borderRadius="m"
                  borderWidth={1}
                  borderColor="border"
                >
                  <Box flex={1}>
                    <Text variant="label" color="text" style={{ fontWeight: '600' }} marginBottom="xs">
                      Ține-mă minte
                    </Text>
                    <Text variant="caption" color="muted">
                      Nu vei mai fi nevoit să te autentifici din nou
                    </Text>
                  </Box>
                  <Controller
                    name="remember"
                    control={control}
                    render={({ field: { value, onChange } }) => (
                      <Switch
                        value={value}
                        onValueChange={onChange}
                        trackColor={{ false: theme.colors.muted, true: theme.colors.primary300 }}
                        thumbColor={value ? theme.colors.primary500 : '#f4f3f4'}
                        ios_backgroundColor={theme.colors.muted}
                      />
                    )}
                  />
                </Box>
                
                {/* Forgot Password Link */}
                <Box alignItems="center" marginBottom="xl">
                  <Text
                    variant="caption"
                    color="primary500"
                    style={{ textDecorationLine: 'underline' }}
                    onPress={() => navigation.navigate('ForgotPassword')}
                  >
                    Ai uitat parola?
                  </Text>
                </Box>

                {/* Enhanced Login Button */}
                <StyledButton
                  title="Conectează-te"
                  onPress={handleSubmit(onSubmit, onInvalid)}
                  disabled={!canSubmit}
                  loading={isSubmitting}
                  gradient
                  size="lg"
                  leftIconName="zap"
                  fullWidth
                  radius="l" // softer, pill-like
                  loadingText="Se conectează…"
                  showSpinnerOnly={false}
                  style={{
                    // lighter, softer shadow
                    shadowColor: '#000',
                    shadowOpacity: 0.04,
                    shadowRadius: 3,
                    shadowOffset: { width: 0, height: 2 },
                    elevation: 0, // no harsh Android drop
                    // subtle outline to define edge on both themes
                    borderWidth: 1,
                    borderColor: 'rgba(255,255,255,0.18)',
                  }}
                />


                {/* Enhanced Divider */}
                <Box 
                  flexDirection="row" 
                  alignItems="center" 
                  justifyContent="center" 
                  marginBottom="l"
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
                    borderRadius="pill"
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

                {/* Enhanced Register Link */}
                <Box alignItems="center">
                  <Text variant="caption" color="muted" marginBottom="s" textAlign="center">
                    Nu ai încă un cont?
                  </Text>
                  <StyledButton
                    title="Creează cont nou"
                    variant="tonal"
                    onPress={() => navigation.navigate('Register')}
                    leftIconName="user-plus"
                    size="sm"
                  />
                </Box>
              </Card>
            </Box>
          </Animated.View>
          
          {/* Feature Highlights */}
          <Box paddingHorizontal="l" marginTop="l" marginBottom="xl">
            <Text 
              variant="subtitle" 
              color="text" 
              textAlign="center" 
              marginBottom="l"
              style={{ fontWeight: '700' }}
            >
              De ce Urban AI?
            </Text>
            
            <Box flexDirection="row" style={{ gap: 8 }} marginBottom="m">
              <FeatureCard 
                icon="zap"
                title="AI Avansat"
                description="Detectează probleme urban cu acuratețe de 98%"
                style={{ flex: 1 }}
              />
              <FeatureCard 
                icon="shield"
                title="Securitate"
                description="Date protejate cu criptare end-to-end"
                style={{ flex: 1 }}
              />
            </Box>
            
            <Box flexDirection="row" style={{ gap: 8 }}>
              <FeatureCard 
                icon="clock"
                title="24/7 Disponibil"
                description="Raportează probleme oricând"
                style={{ flex: 1 }}
              />
              <FeatureCard 
                icon="users"
                title="Comunitate"
                description="Îmbunătățim împreună orașul"
                style={{ flex: 1 }}
              />
            </Box>
          </Box>

          {/* bottom space */}
          <Box height={theme.spacing.xl} />
        </View>
      </TouchableWithoutFeedback>
    </Screen>
  );
}
