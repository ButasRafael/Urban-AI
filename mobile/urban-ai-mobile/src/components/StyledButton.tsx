import React, { ReactNode, useRef } from 'react';
import {
  Pressable,
  Animated,
  ActivityIndicator,
  Platform,
  ViewStyle,
  ColorValue
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import { useTheme } from '@shopify/restyle';
import { Feather } from '@expo/vector-icons';
import { Box, Text } from './restylePrimitives';
import type { Theme } from '../theme';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'tonal' | 'danger';
type BoxColor = keyof Theme['colors'] | 'transparent';

type TextVariantKey = 'display' | 'title' | 'body' | 'label';
type Size = 'sm' | 'md' | 'lg';

interface Props {
  title: ReactNode;
  onPress: () => void;
  variant?: ButtonVariant;
  loading?: boolean;
  disabled?: boolean;
  style?: ViewStyle;
  flex?: number;

  /** ——— Niceties ——— */
  size?: Size;
  fullWidth?: boolean;
  radius?: keyof Theme['borderRadii'];
  haptic?: boolean;
  gradient?: boolean;
  leftIconName?: React.ComponentProps<typeof Feather>['name'];
  rightIconName?: React.ComponentProps<typeof Feather>['name'];
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  iconGap?: number;
  loadingText?: string;
  showSpinnerOnly?: boolean;
  testID?: string;
  accessibilityLabel?: string;
}

export default function StyledButton({
  title,
  onPress,
  variant = 'primary',
  loading = false,
  disabled = false,
  style,
  flex,

  size = 'md',
  fullWidth = false,
  radius = 'm',
  haptic = true,
  gradient = false,
  leftIconName,
  rightIconName,
  leftIcon,
  rightIcon,
  iconGap = 8,
  loadingText,
  showSpinnerOnly = true,
  testID,
  accessibilityLabel,
}: Props) {
  const theme = useTheme<Theme>();
  const scale = useRef(new Animated.Value(1)).current;

  const pressIn  = () =>
    Animated.spring(scale, { toValue: 0.97, useNativeDriver: true, speed: 30, bounciness: 0 }).start();
  const pressOut = () =>
    Animated.spring(scale, { toValue: 1,    useNativeDriver: true, speed: 20, bounciness: 6 }).start();

  const handlePress = () => {
    if (disabled || loading) return;
    if (haptic) Haptics.selectionAsync();
    onPress();
  };

  const sizeMap: Record<
    Size,
    { py: keyof Theme['spacing']; px: keyof Theme['spacing']; font: TextVariantKey; icon: number }
  > = {
    sm: { py: 'xs', px: 'm', font: 'label',  icon: 16 },
    md: { py: 's',  px: 'm', font: 'label',  icon: 18 },
    lg: { py: 'm',  px: 'l', font: 'title',  icon: 20 },
  };
  const S = sizeMap[size];

  let bgColor: BoxColor;
  let borderColor: BoxColor;
  let textColor = '#fff';

  switch (variant) {
    case 'secondary':
      bgColor = borderColor = 'secondary500';
      break;
    case 'ghost':
      bgColor = 'transparent';
      borderColor = 'primary500';
      textColor = theme.colors.primary500;
      break;
    case 'tonal':
      bgColor = 'primary100';
      borderColor = 'transparent';
      textColor = theme.colors.primary500;
      break;
    case 'danger':
      bgColor = borderColor = 'error';
      break;
    default:
      bgColor = borderColor = 'primary500';
  }

  const tactileShadow: ViewStyle =
    Platform.OS === 'ios'
      ? {
          shadowColor: '#000',
          shadowOpacity: 0.12,
          shadowRadius: 6,
          shadowOffset: { width: 0, height: 3 },
        }
      : { elevation: 2 };

  const computedRadius = theme.borderRadii[radius];

  const containerStyle: ViewStyle = {
    borderRadius: computedRadius,
    overflow: 'hidden',
  };

  const content = (
    <Box
      backgroundColor={variant === 'ghost' ? 'transparent' : bgColor}
      borderColor={variant === 'ghost' ? borderColor : 'transparent'}
      borderWidth={variant === 'ghost' ? 1 : 0}
      borderRadius={radius}
      paddingVertical={S.py}
      paddingHorizontal={S.px}
      alignItems="center"
      justifyContent="center"
      flexDirection="row"
      style={[
        tactileShadow,
        fullWidth ? { alignSelf: 'stretch' } : undefined,
        disabled || loading ? { opacity: 0.7 } : null,
        style,
      ]}
    >
      {/* Left icon / spinner */}
      {loading && !showSpinnerOnly ? (
        <ActivityIndicator size="small" color={textColor} style={{ marginRight: iconGap }} />
      ) : leftIcon ? (
        <Box style={{ marginRight: iconGap }}>{leftIcon}</Box>
      ) : leftIconName ? (
        <Box style={{ marginRight: iconGap }}>
          <Feather name={leftIconName} size={S.icon} color={textColor} />
        </Box>
      ) : null}

      {/* Title */}
      {!loading || !showSpinnerOnly ? (
        <Text variant={S.font} style={{ color: textColor }} numberOfLines={1}>
          {loading && loadingText ? loadingText : title}
        </Text>
      ) : (
        <ActivityIndicator size="small" color={textColor} />
      )}

      {/* Right icon */}
      {!loading || !showSpinnerOnly ? (
        rightIcon ? (
          <Box style={{ marginLeft: iconGap }}>{rightIcon}</Box>
        ) : rightIconName ? (
          <Box style={{ marginLeft: iconGap }}>
            <Feather name={rightIconName} size={S.icon} color={textColor} />
          </Box>
        ) : null
      ) : null}
    </Box>
  );

  const isSolidVariant = variant === 'primary' || variant === 'secondary' || variant === 'danger';
  const useGradient = gradient && isSolidVariant;

  const gradientColors: readonly [ColorValue, ColorValue] =
  variant === 'danger'
    ? [theme.colors.error, theme.colors.error]
    : variant === 'secondary'
    ? [
        theme.colors.secondary700 ?? theme.colors.primary700,
        theme.colors.secondary500 ?? theme.colors.primary500,
      ]
    : [theme.colors.primary700, theme.colors.primary500];

  return (
    <Animated.View style={{ transform: [{ scale }], flex, width: fullWidth ? '100%' : undefined }}>
      <Pressable
        onPress={handlePress}
        onPressIn={pressIn}
        onPressOut={pressOut}
        disabled={disabled || loading}
        android_ripple={{ color: '#00000014', borderless: false }}
        style={containerStyle}
        accessibilityRole="button"
        accessibilityLabel={accessibilityLabel ?? (typeof title === 'string' ? title : undefined)}
        accessibilityState={{ disabled: disabled || loading, busy: loading }}
        testID={testID}
      >
        {useGradient ? (
          <LinearGradient colors={gradientColors} start={{ x: 0, y: 0.5 }} end={{ x: 1, y: 0.5 }}>
            {content}
            </LinearGradient>

        ) : (
          content
        )}
      </Pressable>
    </Animated.View>
  );
}
