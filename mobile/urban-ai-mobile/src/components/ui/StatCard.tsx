import React from 'react';
import { ViewStyle } from 'react-native';
import { useTheme } from '@shopify/restyle';
import { Box, Text } from '../restylePrimitives';
import type { Theme } from '../../theme';

type Props = {
  value: string;
  label: string;
  style?: ViewStyle;
};

export default function StatCard({ value, label, style }: Props) {
  const theme = useTheme<Theme>();

  return (
    <Box
      backgroundColor="card"
      borderRadius="l"
      padding="m"
      alignItems="center"
      justifyContent="center"
      style={[theme.shadows.sm, { minHeight: 70 }, style]}
      borderWidth={1}
      borderColor="border"
    >
      <Text 
        variant="title" 
        color="primary500" 
        style={{ fontWeight: '700', fontSize: 20 }}
        marginBottom="xs"
      >
        {value}
      </Text>
      
      <Text 
        variant="caption" 
        color="muted" 
        textAlign="center"
        numberOfLines={1}
      >
        {label}
      </Text>
    </Box>
  );
}