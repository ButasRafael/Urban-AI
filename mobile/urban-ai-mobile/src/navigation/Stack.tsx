// src/navigation/Stack.tsx
import React from 'react';
import { createSharedElementStackNavigator } from 'react-navigation-shared-element';
import { FadeIn, FadeOut } from 'react-native-reanimated';
import type { StackCardInterpolationProps } from '@react-navigation/stack';
import type { RootStackParamList } from './types';

export const Stack = createSharedElementStackNavigator<RootStackParamList>();

export const screenOptions = {
  // ⬇️ fade-through for every push/pop except shared-element screens
  presentation: 'card' as const,
  animation: 'fade' as const,
  cardStyleInterpolator: ({ current, closing }: StackCardInterpolationProps) => ({
    cardStyle: {
      opacity: current.progress,
      transform: [{ scale: closing ? current.progress : 1 }],
    },
  }),
};
