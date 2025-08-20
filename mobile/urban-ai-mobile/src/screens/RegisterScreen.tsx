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
      maxWidth={560}
      statusBarStyle="light-content"
      translucentStatusBar
    >
      {/* Hero header with curved bottom */}
      <HeroHeader
        title="Creează cont"
        subtitle="Înscrie-te pentru a începe"
        rightSlot={
          <Box
            width={56}
            height={56}
            borderRadius="l"
            bg="primary100"
            alignItems="center"
            justifyContent="center"
            style={{
              shadowColor: '#000',
              shadowOpacity: 0.08,
              shadowRadius: 6,
              shadowOffset: { width: 0, height: 3 },
              elevation: 2,
            }}
          >
            <Feather name="user-plus" size={22} color={theme.colors.primary500} />
          </Box>
        }
      />

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
        <Box paddingHorizontal="l" style={{ marginTop: -40, zIndex: 1 }}>
          <Card variant="elevated" gradientBorder>
            {/* card header */}
            <Box mb="m" alignItems="center">
              <Text variant="title" color="text">Înregistrare</Text>
              <Text variant="label" color="muted">Completează datele de mai jos</Text>
            </Box>

            {topError ? (
              <InlineNotice
                variant="error"
                title="Verifică datele"
                message={topError}
                compact
                style={{ marginBottom: 12 }}
              />
            ) : null}

            {/* username */}
            <Controller
              name="username"
              control={control}
              render={({ field: { value, onChange, onBlur } }) => (
                <StyledInput
                  leftIcon="user"
                  value={value}
                  onChangeText={onChange}
                  placeholder="Utilizator"
                  returnKeyType="next"
                  onBlur={onBlur}
                  onSubmitEditing={() => passwordRef.current?.focus()}
                  autoComplete="username"
                  textContentType="username"
                  allowClear
                  errorText={errors.username?.message}
                  style={{ marginBottom: theme.spacing.m }}
                />
              )}
            />

            {/* password */}
            <Controller
              name="password"
              control={control}
              render={({ field: { value, onChange, onBlur } }) => (
                <StyledInput
                  ref={passwordRef}
                  leftIcon="lock"
                  value={value}
                  onChangeText={onChange}
                  placeholder="Parolă (minim 6 caractere)"
                  secureTextEntry
                  passwordToggle
                  returnKeyType="next"
                  onBlur={onBlur}
                  onSubmitEditing={() => confirmRef.current?.focus()}
                  autoComplete="password-new"
                  textContentType="newPassword"
                  errorText={errors.password?.message}
                  style={{ marginBottom: theme.spacing.s }}
                />
              )}
            />

            {/* confirm password */}
            <Controller
              name="confirm"
              control={control}
              render={({ field: { value, onChange, onBlur } }) => (
                <StyledInput
                  ref={confirmRef}
                  leftIcon="lock"
                  value={value}
                  onChangeText={onChange}
                  placeholder="Confirmă parola"
                  secureTextEntry
                  passwordToggle
                  returnKeyType="go"
                  onBlur={onBlur}
                  onSubmitEditing={handleSubmit(onSubmit, onInvalid)}
                  autoComplete="password-new"
                  textContentType="newPassword"
                  errorText={errors.confirm?.message}
                  style={{ marginBottom: theme.spacing.s }}
                />
              )}
            />

            {/* actions */}
            <StyledButton
              title="Creează cont"
              onPress={handleSubmit(onSubmit, onInvalid)}
              disabled={!canSubmit}
              loading={isSubmitting}
              size="lg"
              gradient
              leftIconName="user-plus"
              fullWidth
              loadingText="Se creează…"
              showSpinnerOnly={false}
              style={{ marginTop: 4 }}
            />

            {/* divider */}
            <Box flexDirection="row" alignItems="center" justifyContent="center" mt="m" mb="s">
              <Box style={{ height: 1, flex: 1, backgroundColor: '#E6E9EF' }} />
              <Text variant="label" color="muted" style={{ marginHorizontal: 8 }}>sau</Text>
              <Box style={{ height: 1, flex: 1, backgroundColor: '#E6E9EF' }} />
            </Box>

            {/* secondary */}
            <StyledButton
              title="Ai deja cont? Autentifică-te"
              variant="ghost"
              onPress={() => navigation.navigate('Login')}
              size="sm"
              style={{ alignSelf: 'center' }}
            />
          </Card>
        </Box>
      </Animated.View>

      {/* bottom space */}
      <Box height={theme.spacing.l} />
    </Screen>
  );
}
