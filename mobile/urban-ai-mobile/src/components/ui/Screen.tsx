// src/components/ui/Screen.tsx
import React from 'react';
import {
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StatusBar,
  TouchableWithoutFeedback,
  Keyboard,
  ViewStyle,
} from 'react-native';
import { SafeAreaView, Edge, useSafeAreaInsets } from 'react-native-safe-area-context';
import { useTheme } from '@shopify/restyle';
import type { Theme } from '../../theme';
import { Box } from '../restylePrimitives';

type Props = React.PropsWithChildren<{
  /** Wrap children in a ScrollView */
  scroll?: boolean;
  /** Apply standard page padding to the content box */
  padded?: boolean;
  /** Extra style for the content box */
  contentStyle?: ViewStyle;
  /** Extra style for the outer container */
  style?: ViewStyle;

  /** Safe edges to respect. Default: left/right only so headers can “bleed” top/bottom */
  edges?: ReadonlyArray<Edge>;

  /** Keyboard behavior; 'none' disables KAV */
  keyboard?: 'padding' | 'position' | 'height' | 'none';
  /** Extra vertical offset if you have a custom header */
  keyboardOffset?: number;

  /** Tap anywhere to dismiss the keyboard */
  dismissKeyboardOnTap?: boolean;

  /** Center content and constrain max width (nice on tablets/landscape) */
  center?: boolean;
  maxWidth?: number;

  /** Status bar controls for edge-to-edge layouts */
  statusBarStyle?: 'light-content' | 'dark-content';
  translucentStatusBar?: boolean;
}>;

export default function Screen({
  children,
  scroll,
  padded = true,
  contentStyle,
  style,
  edges = ['left', 'right'],
  keyboard,
  keyboardOffset = 0,
  dismissKeyboardOnTap = true,
  center = false,
  maxWidth = 640,
  statusBarStyle = 'dark-content',
  translucentStatusBar = true,
}: Props) {
  const insets = useSafeAreaInsets();
  const theme = useTheme<Theme>();

  const content = (
    <Box
      flex={1}
      bg="background"
      // when centered, the padding feels nicer on large screens
      padding={padded ? 'l' : undefined}
      style={style}
    >
      <Box
        // constrain inner width on big screens
        style={[
          center ? { alignSelf: 'center', width: '100%', maxWidth } : null,
          contentStyle,
        ]}
      >
        {children}
      </Box>
    </Box>
  );

  const scroller = scroll ? (
    <ScrollView
      contentContainerStyle={{ flexGrow: 1 }}
      contentInsetAdjustmentBehavior="never"
      keyboardShouldPersistTaps="handled"
      keyboardDismissMode={Platform.OS === 'ios' ? 'on-drag' : 'none'}
      // removes Android blue glow; optional
      overScrollMode="never"
      // keep indicators away from rounded corners
      scrollIndicatorInsets={{ top: 0, bottom: Math.max(0, insets.bottom - 2) }}
    >
      {content}
    </ScrollView>
  ) : (
    content
  );

  const body = dismissKeyboardOnTap ? (
    <TouchableWithoutFeedback onPress={Keyboard.dismiss} accessible={false}>
      {scroller}
    </TouchableWithoutFeedback>
  ) : (
    scroller
  );

  const kavBehavior =
    keyboard === 'none' ? undefined : keyboard ?? (Platform.OS === 'ios' ? 'padding' : undefined);

  return (
    <SafeAreaView edges={edges} style={{ flex: 1 }}>
      <StatusBar translucent={translucentStatusBar} backgroundColor="transparent" barStyle={statusBarStyle} />
      {keyboard === 'none' ? (
        body
      ) : (
        <KeyboardAvoidingView style={{ flex: 1 }} behavior={kavBehavior} keyboardVerticalOffset={keyboardOffset}>
          {body}
        </KeyboardAvoidingView>
      )}
    </SafeAreaView>
  );
}
