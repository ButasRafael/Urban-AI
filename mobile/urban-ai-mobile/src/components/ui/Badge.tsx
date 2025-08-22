import React from 'react';
import { ViewStyle } from 'react-native';
import { useTheme } from '@shopify/restyle';
import { Box, Text } from '../restylePrimitives';
import type { Theme } from '../../theme';

type BadgeVariant = 'primary' | 'secondary' | 'success' | 'warning' | 'error';

type Props = {
  children: React.ReactNode;
  variant?: BadgeVariant;
  size?: 'sm' | 'md';
  style?: ViewStyle;
  dot?: boolean;
};

export default function Badge({ 
  children, 
  variant = 'primary',
  size = 'md',
  style,
  dot = false 
}: Props) {
  const theme = useTheme<Theme>();

  const getColors = () => {
    switch (variant) {
      case 'secondary':
        return {
          bg: 'secondary100' as keyof Theme['colors'],
          text: 'secondary700' as keyof Theme['colors'],
          border: 'secondary300' as keyof Theme['colors'],
        };
      case 'success':
        return {
          bg: 'surface50' as keyof Theme['colors'],
          text: 'success' as keyof Theme['colors'],
          border: 'success' as keyof Theme['colors'],
        };
      case 'warning':
        return {
          bg: 'surface50' as keyof Theme['colors'],
          text: 'warning' as keyof Theme['colors'],
          border: 'warning' as keyof Theme['colors'],
        };
      case 'error':
        return {
          bg: 'surface50' as keyof Theme['colors'],
          text: 'error' as keyof Theme['colors'],
          border: 'error' as keyof Theme['colors'],
        };
      default:
        return {
          bg: 'primary100' as keyof Theme['colors'],
          text: 'primary700' as keyof Theme['colors'],
          border: 'primary300' as keyof Theme['colors'],
        };
    }
  };

  const colors = getColors();
  const padding = size === 'sm' ? 'xs' : 's';

  return (
    <Box
      flexDirection="row"
      alignItems="center"
      alignSelf="flex-start"
      backgroundColor={colors.bg}
      borderColor={colors.border}
      borderWidth={1}
      borderRadius="l"
      paddingHorizontal={padding === 'xs' ? 's' : 'm'}
      paddingVertical={padding}
      style={style}
    >
      {dot && (
        <Box
          backgroundColor={colors.text}
          marginRight="xs"
          style={{
            width: 8,
            height: 8,
            borderRadius: 4,
            shadowColor: theme.colors[colors.text],
            shadowOpacity: 0.4,
            shadowRadius: 3,
            shadowOffset: { width: 0, height: 0 },
          }}
        />
      )}
      
      <Text 
        variant={size === 'sm' ? 'caption' : 'label'}
        color={colors.text}
        style={{ fontWeight: '700' }}
      >
        {children}
      </Text>
    </Box>
  );
}