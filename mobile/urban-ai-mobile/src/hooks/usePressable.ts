import { useCallback } from 'react';
import { PressableProps, StyleProp, ViewStyle } from 'react-native';
import Animated, {
  interpolate,
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
} from 'react-native-reanimated';

type Options = {
  scaleFrom?: number;
  scaleTo?: number;
  pressedOpacity?: number;
  durationIn?: number;
  durationOut?: number;
  spring?: boolean;
  disabled?: boolean;
};

type HookReturn = {
  animatedStyle: StyleProp<ViewStyle>;
  onPressIn: NonNullable<PressableProps['onPressIn']>;
  onPressOut: NonNullable<PressableProps['onPressOut']>;
};

export function usePressable({
  scaleFrom = 1,
  scaleTo = 0.98,
  pressedOpacity = 0.9,
  durationIn = 80,
  durationOut = 140,
  spring = true,
  disabled = false,
}: Options = {}): HookReturn {
  const pressed = useSharedValue(0);

  const onPressIn: NonNullable<PressableProps['onPressIn']> = useCallback(() => {
    if (disabled) return;
    pressed.value = withTiming(1, { duration: durationIn });
  }, [disabled, durationIn, pressed]);

  const onPressOut: NonNullable<PressableProps['onPressOut']> = useCallback(() => {
    if (disabled) return;
    pressed.value = spring
      ? withSpring(0, { damping: 14, stiffness: 160 })
      : withTiming(0, { duration: durationOut });
  }, [disabled, durationOut, pressed, spring]);

  const animatedStyle = useAnimatedStyle(() => {
    const s = interpolate(pressed.value, [0, 1], [scaleFrom, scaleTo]);
    const o = interpolate(pressed.value, [0, 1], [1, pressedOpacity]);
    return { transform: [{ scale: s }], opacity: o };
  }, [scaleFrom, scaleTo, pressedOpacity]);

  return { animatedStyle, onPressIn, onPressOut };
}

export const AnimatedPressable = Animated.createAnimatedComponent<any>('Pressable' as any);

