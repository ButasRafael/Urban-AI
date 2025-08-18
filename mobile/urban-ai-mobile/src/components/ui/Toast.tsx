// src/components/ui/Toast.tsx
import React from 'react';
import Toast, { BaseToastProps } from 'react-native-toast-message';
import * as Haptics from 'expo-haptics';
import { Feather } from '@expo/vector-icons';
import { useTheme } from '@shopify/restyle';
import { Box, Text } from '../restylePrimitives';
import type { Theme } from '../../theme';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Platform } from 'react-native';
import { getDefaultHeaderHeight } from '@react-navigation/elements';
import { Dimensions } from 'react-native';

type Kind = 'success' | 'error' | 'info';

const ToastView = ({ text1, text2, props }: BaseToastProps & { props: { kind: Kind } }) => {
  const theme = useTheme<Theme>();
  const kind = props?.kind ?? 'info';
  const map = {
    success: { color: theme.colors.primary500, icon: 'check-circle' as const },
    error:   { color: theme.colors.error,       icon: 'alert-circle' as const },
    info:    { color: theme.colors.primary500,  icon: 'info' as const },
  }[kind];

  return (
    <Box
      bg="card"
      borderRadius="m"
      p="m"
      style={{
        alignSelf: 'center',
        maxWidth: 720,
        width: '94%',
        shadowColor: '#000',
        shadowOpacity: 0.08,
        shadowRadius: 6,
        shadowOffset: { width: 0, height: 3 },
        elevation: 4,                // ensure it draws above headers on Android
      }}
    >
      <Box flexDirection="row" alignItems="flex-start">
        <Feather name={map.icon} size={18} color={map.color} style={{ marginTop: 2 }} />
        <Box ml="s" flex={1}>
          {!!text1 && (
            <Text variant="label" style={{ color: map.color }} numberOfLines={1}>
              {text1}
            </Text>
          )}
          {!!text2 && (
            <Text variant="body" color="text" numberOfLines={3} style={{ marginTop: 2 }}>
              {text2}
            </Text>
          )}
        </Box>
      </Box>
    </Box>
  );
};

export const RootToaster = () => {
  const insets = useSafeAreaInsets();

  // Compute a sane header height for the current platform
  const { width, height } = Dimensions.get('window');
  // `getDefaultHeaderHeight` matches React Navigation's header sizing
  const headerHeight = getDefaultHeaderHeight(
    { width, height },
    Platform.OS === 'ios' ? false : false, // stackPresentation modal? false is fine for normal stacks
    insets.top
  );

  // Final offset: status bar inset + header height + a small gap
  const topOffset = Math.round(insets.top + headerHeight - 24);

  return (
    <Toast
      position="top"
      topOffset={topOffset}
      visibilityTime={2800}
      config={{
        success: (p) => <ToastView {...p} props={{ kind: 'success' }} />,
        error:   (p) => <ToastView {...p} props={{ kind: 'error' }} />,
        info:    (p) => <ToastView {...p} props={{ kind: 'info' }} />,
      }}
    />
  );
};

export const notify = {
  success: (title: string, message?: string) => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    Toast.show({ type: 'success', text1: title, text2: message, position: 'top' });
  },
  error: (title: string, message?: string) => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    Toast.show({ type: 'error', text1: title, text2: message, position: 'top' });
  },
  info: (title: string, message?: string) => {
    Haptics.selectionAsync();
    Toast.show({ type: 'info', text1: title, text2: message, position: 'top' });
  },
};
