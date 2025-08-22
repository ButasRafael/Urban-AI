import React, { useEffect, useMemo, useRef } from 'react';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useTheme } from '@shopify/restyle';
import type { Theme } from '../theme';
import {
  Animated,
  Easing,
  TextInput as RNTextInput,
} from 'react-native';
import { Feather } from '@expo/vector-icons';

import { useForm, Controller, FieldErrors } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';

import { RootStackParamList } from '../navigation/types';
import { Box, Text } from '../components/restylePrimitives';
import Screen from '../components/ui/Screen';
import HeroHeader from '../components/ui/HeroHeader';
import Card from '../components/ui/Card';
import StyledInput from '../components/StyledInput';
import StyledButton from '../components/StyledButton';
import InlineNotice from '../components/ui/InlineNotice';
import Badge from '../components/ui/Badge';
import StatCard from '../components/ui/StatCard';
import FeatureCard from '../components/ui/FeatureCard';
import { notify } from '../components/ui/Toast';
import { register as apiRegister } from '../api/auth';

type Props = NativeStackScreenProps<RootStackParamList, 'Register'>;

const schema = z
  .object({
    username: z.string().min(3, 'Minim 3 caractere'),
    password: z.string().min(6, 'Minim 6 caractere'),
    confirm:  z.string().min(6, 'Repetă parola'),
  })
  .refine((d) => d.password === d.confirm, {
    path: ['confirm'],
    message: 'Parolele nu coincid',
  });

type FormValues = z.infer<typeof schema>;

export default function RegisterScreen({ navigation }: Props) {
  const theme = useTheme<Theme>();

  const {
    control,
    handleSubmit,
    formState: { errors, isValid, isSubmitting, isSubmitted },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    mode: 'onChange',
    defaultValues: { username: '', password: '', confirm: '' },
  });

  const passwordRef = useRef<RNTextInput>(null);
  const confirmRef  = useRef<RNTextInput>(null);
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

  const triggerShake = () => {
    shakeX.setValue(0);
    Animated.sequence([
      Animated.timing(shakeX, { toValue: 1,  duration: 60, useNativeDriver: true }),
      Animated.timing(shakeX, { toValue: -1, duration: 60, useNativeDriver: true }),
      Animated.timing(shakeX, { toValue: 1,  duration: 50, useNativeDriver: true }),
      Animated.timing(shakeX, { toValue: 0,  duration: 40, useNativeDriver: true }),
    ]).start();
  };

  const canSubmit = useMemo(() => isValid && !isSubmitting, [isValid, isSubmitting]);

  const onSubmit = async (values: FormValues) => {
    try {
      await apiRegister(values.username.trim(), values.password);
      notify.success('Cont creat', 'Te poți autentifica acum.');
      navigation.replace('Login');
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      const msg = Array.isArray(detail)
        ? detail.map((e: any) => e.msg).join('\n')
        : typeof detail === 'string'
          ? detail
          : err?.message || 'Înregistrare eșuată';
      if (lastErrorRef.current !== msg) {
        lastErrorRef.current = msg;
        triggerShake();
      }
      notify.error('Eroare', msg);
    }
  };

  const onInvalid = (errs: FieldErrors<FormValues>) => {
    const first =
      (errs.username?.message as string | undefined) ??
      (errs.password?.message as string | undefined) ??
      (errs.confirm?.message as string | undefined) ??
      'Te rugăm să verifici câmpurile.';
    triggerShake();
    notify.info(first);
  };

  const topError =
    isSubmitted
      ? (errors.username?.message ?? errors.password?.message ?? errors.confirm?.message)
      : undefined;

  return (
    <Screen
      scroll
      padded={false}
      edges={['left','right']}
      dismissKeyboardOnTap
      center
      maxWidth={640}
      statusBarStyle="light-content"
      translucentStatusBar
    >
      {/* Enhanced Hero Section */}
      <HeroHeader
        title="Urban AI"
        subtitle="Alătură-te comunității pentru un oraș mai bun"
        height={260} // ⬅️ was 320
        blobs
        rightSlot={
          <Box alignItems="center">
            <Box
              width={64}
              height={64}
              borderRadius="l"
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
              <Feather name="users" size={28} color="#fff" />
            </Box>
            <Badge variant="primary" size="sm" style={{ backgroundColor: 'rgba(255,255,255,0.2)' }}>
              <Text style={{ color: 'white', fontSize: 11, fontWeight: '700' }}>Comunitate</Text>
            </Badge>
          </Box>
        }
      />

      {/* Quick Benefits Row (lift a bit less since header is shorter) */}
      <Box
        paddingHorizontal="l"
        style={{ marginTop: -48, zIndex: 2 }} // ⬅️ was -60
        marginBottom="l"
      >
        <Box flexDirection="row" style={{ gap: 8 }}>
          <StatCard value="Gratuit" label="Cont" style={{ flex: 1 }} />
          <StatCard value="2 min" label="Înregistrare" style={{ flex: 1 }} />
          <StatCard value="Instant" label="Activare" style={{ flex: 1 }} />
        </Box>
      </Box>

      {/* Floating card */}
      <Animated.View
        style={{
          transform: [
            { translateY: cardY },
            {
              translateX: shakeX.interpolate({
                inputRange: [-1, 0, 1],
                outputRange:  [-6, 0, 6],
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
                borderRadius="l"
                backgroundColor="primary100"
                borderColor="primary300"
                borderWidth={1}
                alignItems="center"
                justifyContent="center"
                marginBottom="m"
                style={theme.shadows.sm}
              >
                <Feather name="user-plus" size={24} color={theme.colors.primary700} />
              </Box>
              
              <Text variant="hero" color="text" textAlign="center" marginBottom="xs">
                Creează cont
              </Text>
              
              <Text variant="body" color="muted" textAlign="center">
                Alătură-te comunității Urban AI
              </Text>
              
              <Box flexDirection="row" alignItems="center" marginTop="s" style={{ gap: 4 }}>
                <Badge variant="success" size="sm" dot>
                  Gratuit
                </Badge>
                <Badge variant="primary" size="sm">
                  2 minute
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

            {/* Enhanced Username Input */}
            <Box marginBottom="s">
              <Text variant="label" color="text" marginBottom="xs" style={{ fontWeight: '600' }}>
                Nume utilizator
              </Text>
              <Controller
                name="username"
                control={control}
                render={({ field: { value, onChange, onBlur } }) => (
                  <StyledInput
                    leftIcon="user"
                    value={value}
                    onChangeText={onChange}
                    placeholder="Alege un nume de utilizator"
                    returnKeyType="next"
                    onBlur={onBlur}
                    onSubmitEditing={() => passwordRef.current?.focus()}
                    autoComplete="username"
                    textContentType="username"
                    allowClear
                    errorText={errors.username?.message}
                    helperText={!errors.username?.message ? "Minim 3 caractere, unic" : undefined}
                  />
                )}
              />
            </Box>

            {/* Enhanced Password Input */}
            <Box marginBottom="s">
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
                    placeholder="Creează o parolă puternică"
                    secureTextEntry
                    passwordToggle
                    returnKeyType="next"
                    onBlur={onBlur}
                    onSubmitEditing={() => confirmRef.current?.focus()}
                    autoComplete="password-new"
                    textContentType="newPassword"
                    errorText={errors.password?.message}
                    helperText={!errors.password?.message ? "Minim 6 caractere" : undefined}
                  />
                )}
              />
            </Box>

            {/* Enhanced Confirm Password Input */}
            <Box marginBottom="l">
              <Text variant="label" color="text" marginBottom="xs" style={{ fontWeight: '600' }}>
                Confirmă parola
              </Text>
              <Controller
                name="confirm"
                control={control}
                render={({ field: { value, onChange, onBlur } }) => (
                  <StyledInput
                    ref={confirmRef}
                    leftIcon="lock"
                    value={value}
                    onChangeText={onChange}
                    placeholder="Repetă parola"
                    secureTextEntry
                    passwordToggle
                    returnKeyType="go"
                    onBlur={onBlur}
                    onSubmitEditing={handleSubmit(onSubmit, onInvalid)}
                    autoComplete="password-new"
                    textContentType="newPassword"
                    errorText={errors.confirm?.message}
                    helperText={!errors.confirm?.message ? "Trebuie să fie identică" : undefined}
                  />
                )}
              />
            </Box>

            {/* Enhanced Register Button */}
            <StyledButton
              title="Creează contul"
              onPress={handleSubmit(onSubmit, onInvalid)}
              disabled={!canSubmit}
              loading={isSubmitting}
              gradient
              size="lg"
              leftIconName="user-plus"
              fullWidth
              radius="l" // softer, pill-like
              loadingText="Se creează contul…"
              showSpinnerOnly={false}
              style={{
                shadowColor: '#000',
                shadowOpacity: 0.04,
                shadowRadius: 3,
                shadowOffset: { width: 0, height: 2 },
                elevation: 0,
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
                borderRadius="l"
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

            {/* Enhanced Login Link */}
            <Box alignItems="center">
              <Text variant="caption" color="muted" marginBottom="s" textAlign="center">
                Ai deja un cont?
              </Text>
              <StyledButton
                title="Conectează-te aici"
                variant="tonal"
                onPress={() => navigation.navigate('Login')}
                leftIconName="log-in"
                size="sm"
              />
            </Box>
          </Card>
        </Box>
      </Animated.View>
      
      {/* Registration Benefits */}
      <Box paddingHorizontal="l" marginTop="l" marginBottom="xl">
        <Text 
          variant="subtitle" 
          color="text" 
          textAlign="center" 
          marginBottom="l"
          style={{ fontWeight: '700' }}
        >
          Beneficiile contului Urban AI
        </Text>
        
        <Box flexDirection="row" style={{ gap: 8 }} marginBottom="m">
          <FeatureCard 
            icon="camera"
            title="Reportează ușor"
            description="Fă poze și lasă AI-ul să identifice problema"
            style={{ flex: 1 }}
          />
          <FeatureCard 
            icon="trending-up"
            title="Urmărește progresul"
            description="Vezi statusul sesizărilor tale în timp real"
            style={{ flex: 1 }}
          />
        </Box>
        
        <Box flexDirection="row" style={{ gap: 8 }}>
          <FeatureCard 
            icon="award"
            title="Impact vizibil"
            description="Contribția ta contează pentru comunitate"
            style={{ flex: 1 }}
          />
          <FeatureCard 
            icon="bell"
            title="Notificări smart"
            description="Fi anunțat când problemele sunt rezolvate"
            style={{ flex: 1 }}
          />
        </Box>
      </Box>

      {/* bottom space */}
      <Box height={theme.spacing.xl} />
    </Screen>
  );
}
