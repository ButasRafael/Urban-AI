import React, { useRef, useEffect, useMemo } from 'react';
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
import { login } from '../api/auth';
import { notify } from '../components/ui/Toast';

import Screen from '../components/ui/Screen';
import Card from '../components/ui/Card';

type Props = NativeStackScreenProps<RootStackParamList, 'Login'>;

const REMEMBER_KEY = 'auth:rememberUsername';

const schema = z.object({
  username: z.string().min(3, 'Minim 3 caractere'),
  password: z.string().min(6, 'Minim 6 caractere'),
  remember: z.boolean(),
});
type FormValues = z.infer<typeof schema>;

export default function LoginScreen({ navigation }: Props) {
  const theme = useTheme<Theme>();

  const {
    control,
    handleSubmit,
    setValue,
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

  const onSubmit = async (values: FormValues) => {
    try {
      await login(values.username.trim(), values.password);
      if (values.remember) {
        await AsyncStorage.setItem(REMEMBER_KEY, values.username.trim());
      } else {
        await AsyncStorage.removeItem(REMEMBER_KEY);
      }
      notify.success('Autentificare reușită');
      navigation.replace('Home');
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Autentificare eșuată';
      if (lastErrorRef.current !== msg) {
        lastErrorRef.current = msg;
        triggerShake();
      }
      notify.error(msg);
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
      maxWidth={560}
      statusBarStyle="light-content"
      translucentStatusBar
>
      <TouchableWithoutFeedback onPress={Keyboard.dismiss} accessible={false}>
        <View>
          {/* Hero header with curved bottom */}
          <HeroHeader
            title="Bine ai revenit"
            subtitle="Autentifică-te pentru a continua"
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
                <Feather name="map-pin" size={22} color={theme.colors.primary500} />
              </Box>
            }
          />

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
            <Box paddingHorizontal="l" style={{ marginTop: -40, zIndex: 1 }}>
              <Card variant="elevated" gradientBorder>
                {/* card header */}
                <Box mb="m" alignItems="center">
                  <Text variant="title" color="text">
                    Autentificare
                  </Text>
                  <Text variant="label" color="muted">
                    Introdu datele contului tău
                  </Text>
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
                      errorText={errors.username?.message}
                      allowClear
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
                      placeholder="Parolă"
                      secureTextEntry
                      passwordToggle
                      returnKeyType="go"
                      onBlur={onBlur}
                      onSubmitEditing={handleSubmit(onSubmit, onInvalid)}
                      autoComplete="password"
                      textContentType="password"
                      errorText={errors.password?.message}
                      style={{ marginBottom: theme.spacing.s }}
                    />
                  )}
                />

                {/* remember / forgot */}
                <Box flexDirection="row" alignItems="center" justifyContent="space-between" mb="m">
                  <Controller
                    name="remember"
                    control={control}
                    render={({ field: { value, onChange } }) => (
                      <Box flexDirection="row" alignItems="center">
                        <Switch
                          value={value}
                          onValueChange={onChange}
                          thumbColor={value ? theme.colors.primary500 : '#fff'}
                          trackColor={{ false: '#CDD6E1', true: theme.colors.primary300 }}
                        />
                        <Text variant="label" color="text" style={{ marginLeft: 8 }}>
                          Ține-mă minte
                        </Text>
                      </Box>
                    )}
                  />

                  <Text
                    variant="label"
                    color="primary500"
                    onPress={() => notify.info('Contactează administratorul pentru resetare parolă.')}
                  >
                    Ai uitat parola?
                  </Text>
                </Box>

                {/* submit */}
                <StyledButton
                  title="Autentificare"
                  onPress={handleSubmit(onSubmit, onInvalid)}
                  disabled={!canSubmit}
                  loading={isSubmitting}
                  size="lg"
                  gradient
                  leftIconName="log-in"
                  fullWidth
                  loadingText="Se autentifică…"
                  showSpinnerOnly={false}
                  style={{ marginTop: 4 }}
                />

                {/* divider */}
                <Box flexDirection="row" alignItems="center" justifyContent="center" mt="m" mb="s">
                  <View style={{ height: 1, flex: 1, backgroundColor: '#E6E9EF' }} />
                  <Text variant="label" color="muted" style={{ marginHorizontal: 8 }}>
                    sau
                  </Text>
                  <View style={{ height: 1, flex: 1, backgroundColor: '#E6E9EF' }} />
                </Box>

                {/* secondary CTA */}
                <StyledButton
                  title="Nu ai cont? Creează unul"
                  variant="ghost"
                  onPress={() => navigation.navigate('Register')}
                  size="sm"
                  style={{ alignSelf: 'center' }}
                />
              </Card>
            </Box>
          </Animated.View>

          {/* bottom space */}
          <Box height={theme.spacing.l} />
        </View>
      </TouchableWithoutFeedback>
    </Screen>
  );
}
