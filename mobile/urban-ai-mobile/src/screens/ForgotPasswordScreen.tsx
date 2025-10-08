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
import { notify } from '../components/ui/Toast';
import { forgotPassword } from '../api/auth';

type Props = NativeStackScreenProps<RootStackParamList, 'ForgotPassword'>;

const schema = z.object({
  email: z.string().email('Email invalid'),
});
type FormValues = z.infer<typeof schema>;

export default function ForgotPasswordScreen({ navigation }: Props) {
  const theme = useTheme<Theme>();

  const {
    control,
    handleSubmit,
    formState: { errors, isValid, isSubmitting, isSubmitted },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    mode: 'onChange',
    defaultValues: { email: '' },
  });

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
      await forgotPassword(values.email.trim().toLowerCase());
      notify.success('Email trimis', 'Verifică-ți inbox-ul pentru instrucțiuni.');
      // Navigate back to login after short delay
      setTimeout(() => navigation.navigate('Login'), 2000);
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Eroare la trimiterea emailului';
      if (lastErrorRef.current !== msg) {
        lastErrorRef.current = msg;
        triggerShake();
      }
      notify.error('Eroare', msg);
    }
  };

  const onInvalid = (errs: FieldErrors<FormValues>) => {
    const first = (errs.email?.message as string | undefined) ?? 'Te rugăm să verifici emailul.';
    triggerShake();
    notify.info(first);
  };

  const topError = isSubmitted ? errors.email?.message : undefined;

  return (
    <Screen
      scroll
      padded={false}
      edges={['left','right']}
      dismissKeyboardOnTap
      center
      maxWidth={640}
    >
      {/* Enhanced Hero Section */}
      <HeroHeader
        title="Urban AI"
        subtitle="Resetează-ți parola"
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
              <Feather name="key" size={28} color="#fff" />
            </Box>
            <Badge variant="primary" size="sm" style={{ backgroundColor: 'rgba(255,255,255,0.2)' }}>
              <Text style={{ color: 'white', fontSize: 11, fontWeight: '700' }}>Recuperare</Text>
            </Badge>
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
          marginTop: -48,
          zIndex: 2,
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
                <Feather name="key" size={24} color={theme.colors.primary700} />
              </Box>

              <Text variant="hero" color="text" textAlign="center" marginBottom="xs">
                Ai uitat parola?
              </Text>

              <Text variant="body" color="muted" textAlign="center">
                Introdu emailul și îți vom trimite instrucțiuni
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

            {/* Info Notice */}
            <Box marginBottom="l">
              <InlineNotice
                variant="info"
                title="Resetare parolă"
                message="Vei primi un email cu instrucțiuni pentru resetarea parolei."
                compact={false}
              />
            </Box>

            {/* Enhanced Email Input */}
            <Box marginBottom="l">
              <Text variant="label" color="text" marginBottom="xs" style={{ fontWeight: '600' }}>
                Adresă email
              </Text>
              <Controller
                name="email"
                control={control}
                render={({ field: { value, onChange, onBlur } }) => (
                  <StyledInput
                    leftIcon="mail"
                    value={value}
                    onChangeText={onChange}
                    placeholder="email@exemplu.ro"
                    returnKeyType="go"
                    onBlur={onBlur}
                    onSubmitEditing={handleSubmit(onSubmit, onInvalid)}
                    autoComplete="email"
                    textContentType="emailAddress"
                    keyboardType="email-address"
                    autoCapitalize="none"
                    allowClear
                    errorText={errors.email?.message}
                    helperText={!errors.email?.message ? "Emailul asociat contului tău" : undefined}
                  />
                )}
              />
            </Box>

            {/* Submit Button */}
            <StyledButton
              title="Trimite instrucțiuni"
              onPress={handleSubmit(onSubmit, onInvalid)}
              disabled={!canSubmit}
              loading={isSubmitting}
              gradient
              size="lg"
              leftIconName="send"
              fullWidth
              radius="l"
              loadingText="Se trimite…"
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

            {/* Divider */}
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
              <Text variant="caption" color="muted" marginBottom="s" textAlign="center">
                Ți-ai amintit parola?
              </Text>
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
            <Feather name="shield" size={20} color={theme.colors.primary500} />
            <Text variant="subtitle" color="text" marginLeft="s" style={{ fontWeight: '600' }}>
              Securitate
            </Text>
          </Box>
          <Text variant="caption" color="muted">
            Link-ul de resetare este valabil doar 1 oră și poate fi folosit o singură dată.
          </Text>
        </Box>
      </Box>

      {/* bottom space */}
      <Box height={theme.spacing.xl} />
    </Screen>
  );
}