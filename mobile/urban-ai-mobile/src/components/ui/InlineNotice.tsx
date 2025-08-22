import React, { useEffect, useRef } from 'react';
import { Animated, ViewStyle, Pressable } from 'react-native';
import { Feather } from '@expo/vector-icons';
import { useTheme } from '@shopify/restyle';
import { Box, Text } from '../restylePrimitives';
import type { Theme } from '../../theme';
import { alpha, opacity } from '../../theme/utils';
import {motion} from "../../theme/motion";

type Variant = 'error' | 'success' | 'info' | 'warning';

type Props = {
  variant?: Variant;
  title?: string;
  message?: string | React.ReactNode;
  onClose?: () => void;
  style?: ViewStyle;
  compact?: boolean;
};

export default function InlineNotice({
  variant = 'info',
  title,
  message,
  onClose,
  style,
  compact = false,
}: Props) {
  const theme = useTheme<Theme>();
  const fade = useRef(new Animated.Value(0)).current;
  const slide = useRef(new Animated.Value(6)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fade, { toValue: 1, duration: motion.dur.xs, easing: motion.curve.standard, useNativeDriver: true }),
      Animated.spring(slide, { toValue: 0, useNativeDriver: true }),
    ]).start();
  }, [fade, slide]);

  const colorMap: Record<Variant, string> = {
    error: theme.colors.error,
    success: theme.colors.primary500,
    info: theme.colors.primary500,
    warning: theme.colors.secondary500 ?? theme.colors.primary500,
  };

  const accent = colorMap[variant];
  const bgTint = alpha(accent, opacity.overlayWeak);
  const borderTint = alpha(accent, 0.35);

  const iconMap: Record<Variant, React.ComponentProps<typeof Feather>['name']> = {
    error: 'alert-circle',
    success: 'check-circle',
    info: 'info',
    warning: 'alert-triangle',
  };

  return (
    <Animated.View style={{ opacity: fade, transform: [{ translateY: slide }] }}>
      <Box
        style={style}
        borderRadius="m"
        borderWidth={1}
        borderColor="transparent"
        backgroundColor="transparent"
      >
        <Box
          p={compact ? 's' : 'm'}
          borderRadius="m"
          style={{ backgroundColor: bgTint, borderWidth: 1, borderColor: borderTint }}
          accessibilityRole="alert"
        >
          <Box flexDirection="row" alignItems="flex-start">
            <Feather name={iconMap[variant]} size={18} color={accent} style={{ marginTop: 2 }} />
            <Box flex={1} ml="s">
              {title ? (
                <Text variant="label" style={{ color: accent, marginBottom: 2 }}>
                  {title}
                </Text>
              ) : null}
              {typeof message === 'string' ? (
                <Text variant={compact ? 'label' : 'body'} color="text">
                  {message}
                </Text>
              ) : (
                message
              )}
            </Box>

            {onClose ? (
              <Pressable
                onPress={onClose}
                hitSlop={10}
                style={{ marginLeft: 8, padding: 2 }}
                accessibilityRole="button"
                accessibilityLabel="Închide mesajul"
              >
                <Feather name="x" size={16} color={accent} />
              </Pressable>
            ) : null}
          </Box>
        </Box>
      </Box>
    </Animated.View>
  );
}
