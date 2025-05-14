// tactile + shadows + platform elevation ------------------------------------
import React, { ReactNode, useRef } from 'react';
import {
  TouchableWithoutFeedback,
  Animated,
  ActivityIndicator,
  Platform,
  ViewStyle,
} from 'react-native';
import { useTheme } from '@shopify/restyle';
import { Box, Text } from './restylePrimitives';
import type { Theme } from '../theme';

export type ButtonVariant =
  | 'primary'
  | 'secondary'
  | 'ghost'
  | 'tonal'
  | 'danger';

type BoxColor = keyof Theme['colors'] | 'transparent';

interface Props {
  title: ReactNode;
  onPress: () => void;
  variant?: ButtonVariant;
  loading?: boolean;
  disabled?: boolean;
  style?: ViewStyle;
  /** when used inside flex-row layouts just pass `flex={1}` ↓ */
  flex?: number;
}

export default function StyledButton({
  title,
  onPress,
  variant = 'primary',
  loading = false,
  disabled = false,
  style,
  flex,
}: Props) {
  /* scale on press ------------------------------------------------ */
  const scale = useRef(new Animated.Value(1)).current;
  const handlePressIn  = () => Animated.spring(scale, { toValue: 0.96, useNativeDriver: true }).start();
  const handlePressOut = () => Animated.spring(scale, { toValue: 1,    useNativeDriver: true }).start();

  /* colours ------------------------------------------------------- */
  const { colors } = useTheme<Theme>();

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
      textColor = colors.primary500;
      break;
    case 'tonal':
      bgColor = 'primary100';
      borderColor = 'transparent';
      textColor = colors.primary500;
      break;
    case 'danger':
      bgColor = borderColor = 'error';
      break;
    default:
      bgColor = borderColor = 'primary500';
  }

  /* platform shadow ---------------------------------------------- */
  const tactileShadow: ViewStyle =
    Platform.OS === 'ios'
      ? { shadowColor: '#000', shadowOpacity: 0.1, shadowRadius: 4, shadowOffset: { width: 0, height: 2 } }
      : { elevation: 1 };

  /* render -------------------------------------------------------- */
  return (
    <TouchableWithoutFeedback
      onPress={onPress}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      disabled={disabled || loading}
    >
      <Animated.View style={{ transform: [{ scale }], flex }}>
        <Box
          backgroundColor={bgColor}
          borderColor={borderColor}
          borderWidth={variant === 'ghost' ? 1 : 0}
          borderRadius="m"
          paddingVertical="s"
          paddingHorizontal="m"
          alignItems="center"
          justifyContent="center"
          marginVertical="xs"
          style={[tactileShadow, style]}
        >
          {loading ? (
            <ActivityIndicator color={textColor} />
          ) : (
            <Text variant="label" style={{ color: textColor }}>
              {title}
            </Text>
          )}
        </Box>
      </Animated.View>
    </TouchableWithoutFeedback>
  );
}
