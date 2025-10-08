import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import { FadeIn, FadeOut } from 'react-native-reanimated';
import type { StackCardInterpolationProps } from '@react-navigation/stack';
import type { RootStackParamList } from './types';

export const Stack = createStackNavigator<RootStackParamList>();

export const screenOptions = {
  presentation: 'card' as const,
  animation: 'fade' as const,
  cardStyleInterpolator: ({ current, closing }: StackCardInterpolationProps) => ({
    cardStyle: {
      opacity: current.progress,
      transform: [{ scale: closing ? current.progress : 1 }],
    },
  }),
};
