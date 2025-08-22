import React, { forwardRef, PropsWithChildren, useRef } from 'react';
import { Platform, Pressable, ViewStyle, Animated } from 'react-native';
import { useTheme } from '@shopify/restyle';
import { LinearGradient } from 'expo-linear-gradient';
import { Box } from '../restylePrimitives';
import type { Theme } from '../../theme';
import { alpha, opacity } from '../../theme/utils';

type CardVariant = 'elevated' | 'outlined' | 'tonal' | 'flat';

type Props = PropsWithChildren<{
  style?: ViewStyle;
  variant?: CardVariant;
  gradientBorder?: boolean;
  header?: React.ReactNode;
  footer?: React.ReactNode;
  onPress?: () => void;
  padding?: keyof Theme['spacing'];
  radius?: keyof Theme['borderRadii'];
}>;

function getCardShadow(variant: CardVariant, theme: Theme): ViewStyle {
  switch (variant) {
    case 'elevated':
      return theme.shadows.md;
    case 'outlined':
      return theme.shadows.sm;
    case 'tonal':
      return theme.shadows.sm;
    default:
      return {};
  }
}

const Card = forwardRef<any, Props>(function Card(
  {
    children,
    style,
    variant = 'elevated',
    gradientBorder = false,
    header,
    footer,
    onPress,
    padding = 'l',
    radius = 'l',
  },
  ref
) {
  const theme = useTheme<Theme>();

  const bgToken: keyof Theme['colors'] =
    variant === 'tonal'
      ? 'primary100'
      : 'card';

  const base = (
    <Box
      bg={bgToken}
      p={padding}
      borderRadius={radius}
      borderWidth={variant === 'outlined' ? 1 : 0}
      borderColor={variant === 'outlined' ? 'border' : 'transparent'}
      style={[
        getCardShadow(variant, theme),
        // Keep rounded corners respected for inner content
        { overflow: variant === 'flat' ? 'visible' : 'hidden' },
          variant === 'tonal' ? { backgroundColor: alpha(theme.colors.primary100, opacity.overlayWeak) } : null,
          style,
      ]}
    >
      {header ? <Box mb="m">{header}</Box> : null}
      {children}
      {footer ? <Box mt="m">{footer}</Box> : null}
    </Box>
  );

  const withGradientBorder = (node: React.ReactNode) => (
    <LinearGradient
      colors={[theme.colors.primary300, theme.colors.primary500]}
      start={{ x: 0, y: 0.5 }}
      end={{ x: 1, y: 0.5 }}
      style={{ padding: 1.2, borderRadius: theme.borderRadii[radius] }}
    >
      <Box bg="card" borderRadius={radius}>
        {node}
      </Box>
    </LinearGradient>
  );

  const body = gradientBorder ? withGradientBorder(base) : base;

  if (!onPress) return <>{body}</>;

  const scale = useRef(new Animated.Value(1)).current;
  const pressIn = () =>
    Animated.spring(scale, { toValue: 0.98, useNativeDriver: true }).start();
  const pressOut = () =>
    Animated.spring(scale, { toValue: 1, useNativeDriver: true }).start();

  return (
    <Animated.View ref={ref} style={{ transform: [{ scale }] }}>
      <Pressable
        onPress={onPress}
        onPressIn={pressIn}
        onPressOut={pressOut}
        android_ripple={{ color: alpha(theme.colors.text, opacity.overlayWeak) }}
        style={{ borderRadius: theme.borderRadii[radius] }}
      >
        {body}
      </Pressable>
    </Animated.View>
  );
});

export default Card;
