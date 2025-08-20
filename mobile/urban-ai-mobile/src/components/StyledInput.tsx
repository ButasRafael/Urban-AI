import React, { useState, forwardRef, useImperativeHandle, useRef } from 'react';
import {
  TextInput as RNTextInput,
  TextInputProps,
  Platform,
  TextStyle,
  ViewStyle,
  Pressable,
} from 'react-native';
import { Box, Text } from './restylePrimitives';
import { useTheme } from '@shopify/restyle';
import type { Theme } from '../theme';
import { Feather } from '@expo/vector-icons';

type Props = {
  value: string;
  onChangeText: (text: string) => void;
  placeholder?: string;
  secureTextEntry?: boolean;
  passwordToggle?: boolean;
  leftIcon?: React.ComponentProps<typeof Feather>['name'];
  style?: ViewStyle;
  inputStyle?: TextStyle;
  errorText?: string;
  disabled?: boolean;
  helperText?: string;
  allowClear?: boolean;
  rightAdornment?: React.ReactNode;
  onRightPress?: () => void;
} & Omit<
  TextInputProps,
  'style' | 'onChangeText' | 'value' | 'placeholder' | 'secureTextEntry'
>;

const StyledInput = forwardRef<RNTextInput, Props>(function StyledInput(
  {
    value,
    onChangeText,
    placeholder,
    secureTextEntry = false,
    passwordToggle = false,
    leftIcon,
    style,
    inputStyle,
    errorText,
    autoCapitalize = 'none',
    autoCorrect = false,

    disabled = false,
    helperText,
    allowClear = true,
    rightAdornment,
    onRightPress,

    ...rest
  },
  ref
) {
  const innerRef = useRef<RNTextInput>(null);
  useImperativeHandle(ref, () => innerRef.current as RNTextInput);

  const [focused, setFocused] = useState(false);
  const [show, setShow] = useState(false);
  const theme = useTheme<Theme>();

  const shadowStyle = Platform.OS === 'ios' ? styles.iosShadow : styles.androidShadow;
  const isSecure = secureTextEntry && !show;
  const hasError = Boolean(errorText);
  const hasValue = value?.length > 0;

  const borderToken = hasError ? 'error' : focused ? 'primary500' : 'muted';
  const containerOpacity = disabled ? 0.5 : 1;

  const focusInput = () => {
    if (!disabled) innerRef.current?.focus();
  };

  const canShowClear = allowClear && !secureTextEntry && hasValue && !passwordToggle && !rightAdornment;

  return (
    <>
      <Pressable onPress={focusInput} disabled={disabled} style={{ opacity: containerOpacity }}>
        <Box
          borderWidth={1}
          borderColor={borderToken}
          borderRadius="m"
          backgroundColor="surface0"
          marginBottom="m"
          style={[shadowStyle, style, { position: 'relative' }]}
          accessibilityRole="text"
          accessibilityLabel={placeholder}
        >
          {leftIcon && (
            <Pressable
              onPress={focusInput}
              hitSlop={10}
              style={{ position: 'absolute', left: 12, top: 12 }}
            >
              <Feather
                name={leftIcon}
                size={18}
                color={focused ? theme.colors.primary500 : theme.colors.muted}
              />
            </Pressable>
          )}

          <RNTextInput
            ref={innerRef}
            value={value}
            onChangeText={onChangeText}
            placeholder={placeholder}
            placeholderTextColor={theme.colors.muted}
            secureTextEntry={isSecure}
            autoCapitalize={autoCapitalize}
            autoCorrect={autoCorrect}
            editable={!disabled}
            selectionColor={theme.colors.primary300}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            style={{
              padding: theme.spacing.m,
              paddingLeft: leftIcon ? theme.spacing.m + 24 : theme.spacing.m,
              paddingRight:
                (passwordToggle && secureTextEntry) || canShowClear || rightAdornment
                  ? theme.spacing.m + 28
                  : theme.spacing.m,
              ...(theme.textVariants.body as TextStyle),
              color: theme.colors.text,
              ...inputStyle,
            }}
            {...rest}
          />

          {/* Right controls: password toggle > custom right > clear */}
          {passwordToggle && secureTextEntry ? (
            <Pressable
              onPress={() => setShow((s) => !s)}
              hitSlop={10}
              style={{ position: 'absolute', right: 12, top: 12 }}
              accessibilityRole="button"
              accessibilityLabel={show ? 'Ascunde parola' : 'Arată parola'}
            >
              <Feather
                name={show ? 'eye-off' : 'eye'}
                size={18}
                color={focused ? theme.colors.primary500 : theme.colors.muted}
              />
            </Pressable>
          ) : rightAdornment ? (
            <Pressable
              onPress={onRightPress}
              hitSlop={10}
              disabled={!onRightPress}
              style={{ position: 'absolute', right: 10, top: 10 }}
            >
              {rightAdornment}
            </Pressable>
          ) : canShowClear ? (
            <Pressable
              onPress={() => onChangeText('')}
              hitSlop={10}
              style={{ position: 'absolute', right: 12, top: 12 }}
              accessibilityRole="button"
              accessibilityLabel="Șterge textul"
            >
              <Feather
                name="x-circle"
                size={18}
                color={focused ? theme.colors.primary500 : theme.colors.muted}
              />
            </Pressable>
          ) : null}
        </Box>
      </Pressable>

      {!!errorText && (
        <Text variant="label" color="error" mb="s" accessibilityLiveRegion="polite">
          {errorText}
        </Text>
      )}

      {!errorText && !!helperText && (
        <Text variant="label" color="muted" mb="s">
          {helperText}
        </Text>
      )}
    </>
  );
});

export default StyledInput;

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
