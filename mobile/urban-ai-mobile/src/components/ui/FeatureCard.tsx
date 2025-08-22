import React from 'react';
import { ViewStyle } from 'react-native';
import { useTheme } from '@shopify/restyle';
import { Feather } from '@expo/vector-icons';
import { Box, Text } from '../restylePrimitives';
import type { Theme } from '../../theme';

type Props = {
  icon: React.ComponentProps<typeof Feather>['name'];
  title: string;
  description: string;
  style?: ViewStyle;
};

export default function FeatureCard({ icon, title, description, style }: Props) {
  const theme = useTheme<Theme>();

  return (
    <Box
      backgroundColor="card"
      borderRadius="l"
      padding="m"
      style={[theme.shadows.sm, style]}
      borderWidth={1}
      borderColor="border"
    >
      <Box
        width={36}
        height={36}
        borderRadius="m"
        backgroundColor="primary100"
        borderColor="primary300"
        borderWidth={1}
        alignItems="center"
        justifyContent="center"
        marginBottom="s"
      >
        <Feather name={icon} size={18} color={theme.colors.primary700} />
      </Box>
      
      <Text variant="label" color="text" style={{ fontWeight: '600' }} marginBottom="xs">
        {title}
      </Text>
      
      <Text variant="caption" color="muted" numberOfLines={2}>
        {description}
      </Text>
    </Box>
  );
}