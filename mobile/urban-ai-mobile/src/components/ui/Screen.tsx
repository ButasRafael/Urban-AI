import React from 'react';
import {
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  TouchableWithoutFeedback,
  Keyboard,
  ViewStyle,
} from 'react-native';
import { SafeAreaView, Edge, useSafeAreaInsets } from 'react-native-safe-area-context';
import { useTheme } from '@shopify/restyle';
import type { Theme } from '../../theme';
import { Box } from '../restylePrimitives';

type Props = React.PropsWithChildren<{
  scroll?: boolean;
  padded?: boolean;
  contentStyle?: ViewStyle;
  style?: ViewStyle;
  edges?: ReadonlyArray<Edge>;
  keyboard?: 'padding' | 'position' | 'height' | 'none';
  keyboardOffset?: number;
  dismissKeyboardOnTap?: boolean;
  center?: boolean;
  maxWidth?: number;
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
}: Props) {
  const insets = useSafeAreaInsets();
  const theme = useTheme<Theme>();

  const content = (
    <Box
      flex={1}
      bg="background"
      padding={padded ? 'l' : undefined}
      style={style}
    >
      <Box
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
      overScrollMode="never"
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
