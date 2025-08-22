import React from 'react';
import { View, ViewStyle } from 'react-native';
import { useTheme } from '@shopify/restyle';
import type { Theme } from '../../theme';

import { Skeleton as MotiSkeleton } from 'moti/skeleton';

type Radius = keyof Theme['borderRadii'] | number;

type BaseProps = {
  width?: number | string;
  height?: number | string;
  radius?: Radius;
  show?: boolean;
  containerStyle?: ViewStyle | ViewStyle[];
};

export function Skeleton({ width = '100%', height = 16, radius = 's', show = true, containerStyle }: BaseProps) {
  const { borderRadii } = useTheme<Theme>();
  const mode = (useTheme<Theme>() as any).mode as 'light' | 'dark';
  const r = typeof radius === 'number' ? radius : borderRadii[radius];
  return (
    <View style={containerStyle as any}>
      <MotiSkeleton
        show={show}
        colorMode={mode === 'dark' ? 'dark' : 'light'}
        width={width as any}
        height={height as any}
        radius={r}
      />
    </View>
  );
}

type TextSkeletonProps = {
  lines?: number;
  lineHeight?: number;
  gap?: number;
  widths?: Array<number | string>; // optional per-line widths
  containerStyle?: ViewStyle;
};

export function SkeletonText({
  lines = 3,
  lineHeight = 14,
  gap = 8,
  widths,
  containerStyle,
}: TextSkeletonProps) {
  const items = Array.from({ length: lines });
  return (
    <>
      {items.map((_, i) => (
        <Skeleton
          key={i}
          height={lineHeight}
          radius={4}
          width={widths?.[i] ?? (i === lines - 1 ? '70%' : '100%')}
          containerStyle={[{ marginBottom: i === lines - 1 ? 0 : gap }, containerStyle] as any}
        />
      ))}
    </>
  );
}

export default Skeleton;
