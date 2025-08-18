// src/components/ui/Card.tsx
import React, { forwardRef, PropsWithChildren, useRef } from 'react';
import { Platform, Pressable, ViewStyle, Animated } from 'react-native';
import { useTheme } from '@shopify/restyle';
import { LinearGradient } from 'expo-linear-gradient';
import { Box } from '../restylePrimitives';
import type { Theme } from '../../theme';

type CardVariant = 'elevated' | 'outlined' | 'tonal' | 'flat';

type Props = PropsWithChildren<{
  style?: ViewStyle;
  /** Visual style of the card */
  variant?: CardVariant;
  /** Optional gradient border wrapper */
  gradientBorder?: boolean;
  /** Header/footer convenience slots */
  header?: React.ReactNode;
  footer?: React.ReactNode;
  /** Make the whole card tappable */
  onPress?: () => void;
  /** Override paddings/radius using theme tokens */
  padding?: keyof Theme['spacing'];
  radius?: keyof Theme['borderRadii'];
}>;

function platformShadow(intensity = 6): ViewStyle {
  if (Platform.OS === 'ios') {
    return {
      shadowColor: '#000',
      shadowOpacity: 0.08,
      shadowRadius: intensity,
      shadowOffset: { width: 0, height: Math.round(intensity / 2) },
    };
  }
  return { elevation: Math.max(2, Math.round(intensity / 2)) };
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
      borderColor={variant === 'outlined' ? 'muted' : 'transparent'}
      style={[
        variant === 'elevated' ? platformShadow(6) : null,
        // Keep rounded corners respected for inner content
        { overflow: variant === 'flat' ? 'visible' : 'hidden' },
        style,
      ]}
    >
      {header ? <Box mb="m">{header}</Box> : null}
      {children}
      {footer ? <Box mt="m">{footer}</Box> : null}
    </Box>
  );

  // Optional gradient border wrapper
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

  // Clickable: gentle scale + ripple
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
        android_ripple={{ color: '#00000014' }}
        style={{ borderRadius: theme.borderRadii[radius] }}
      >
        {body}
      </Pressable>
    </Animated.View>
  );
});

export default Card;
