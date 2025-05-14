// src/components/StyledInput.tsx
import React, { useState } from 'react';
import {
  TextInput,
  Platform,
  TextStyle,
  ViewStyle,
} from 'react-native';
import { Box } from './restylePrimitives';
import { useTheme } from '@shopify/restyle';
import type { Theme } from '../theme';

type InputProps = {
  value: string;
  onChangeText: (text: string) => void;
  placeholder?: string;
  secureTextEntry?: boolean;
  style?: ViewStyle;
};

export default function StyledInput({
  value,
  onChangeText,
  placeholder,
  secureTextEntry = false,
  style,
}: InputProps) {
  const [focused, setFocused] = useState(false);
  const theme = useTheme<Theme>();

  const shadowStyle = Platform.OS === 'ios'
    ? styles.iosShadow
    : styles.androidShadow;

  return (
    <Box
      borderWidth={1}
      /* pass token names, not hex */
      borderColor={focused ? 'primary500' : 'muted'}
      borderRadius="m"
      backgroundColor="surface0"
      marginBottom="m"
      style={[shadowStyle, style]}
    >
      <TextInput
        value={value}
        onChangeText={onChangeText}
        placeholder={placeholder}
        /* TextInput prop, hex from theme is fine */
        placeholderTextColor={theme.colors.muted}
        secureTextEntry={secureTextEntry}
        style={{
          padding: theme.spacing.m,
          ...(theme.textVariants.body as TextStyle),
          color: theme.colors.text,
        }}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
      />
    </Box>
  );
}

const styles = {
  iosShadow: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.075,
    shadowRadius: 2,
  } as ViewStyle,
  androidShadow: {
    elevation: 1,
  } as ViewStyle,
};
