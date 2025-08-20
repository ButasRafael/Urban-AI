import React, { useRef, useEffect } from 'react';
import { View, Animated, Easing, StyleSheet } from 'react-native';
import { useTheme } from '@shopify/restyle';
import type { Theme } from '../../theme';
import { LinearGradient } from 'expo-linear-gradient';
import Svg, { Path } from 'react-native-svg';
import { Box, Text } from '../restylePrimitives';

type Props = {
  height?: number;
  title: string;
  subtitle?: string;
  rightSlot?: React.ReactNode;
  blobs?: boolean;
};

export default function HeroHeader({
  height = 240,
  title,
  subtitle,
  rightSlot,
  blobs = true,
}: Props) {
  const theme = useTheme<Theme>();
  const logoScale = useRef(new Animated.Value(0.92)).current;

  useEffect(() => {
    Animated.sequence([
      Animated.timing(logoScale, {
        toValue: 1.04,
        duration: 300,
        easing: Easing.out(Easing.quad),
        useNativeDriver: true,
      }),
      Animated.spring(logoScale, { toValue: 1, useNativeDriver: true }),
    ]).start();
  }, [logoScale]);

  const bg = theme.colors.background;

  return (
    <View style={{ height, position: 'relative' }}>
      {/* gradient background */}
      <LinearGradient
        colors={[theme.colors.primary700, theme.colors.primary500]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={StyleSheet.absoluteFill}
      />

      {/* soft blobs */}
      {blobs && (
        <>
          <View
            pointerEvents="none"
            style={{
              position: 'absolute',
              top: -30,
              right: -30,
              width: 140,
              height: 140,
              borderRadius: 999,
              backgroundColor: 'rgba(255,255,255,0.12)',
            }}
          />
          <View
            pointerEvents="none"
            style={{
              position: 'absolute',
              top: 20,
              left: -20,
              width: 90,
              height: 90,
              borderRadius: 999,
              backgroundColor: 'rgba(255,255,255,0.08)',
            }}
          />
        </>
      )}

      {/* content row */}
      <Box
        flexDirection="row"
        alignItems="center"
        justifyContent="space-between"
        style={{
          paddingHorizontal: 24,
          paddingBottom: 20,
          position: 'absolute',
          bottom: 42,
          left: 0,
          right: 0,
        }}
      >
        <Box>
          <Text variant="title" style={{ color: '#fff' }}>
            {title}
          </Text>
        </Box>

        {subtitle ? (
          <Box position="absolute" left={24} bottom={4}>
            <Text variant="label" style={{ color: '#ffffffcc' }}>
              {subtitle}
            </Text>
          </Box>
        ) : null}

        {rightSlot ? (
          <Animated.View style={{ transform: [{ scale: logoScale }] }}>
            {rightSlot}
          </Animated.View>
        ) : null}
      </Box>

      {/* bottom wave cutout (fills with screen background) */}
      <Svg
        width="100%"
        height={64}
        viewBox="0 0 100 64"
        preserveAspectRatio="none"
        style={{ position: 'absolute', bottom: -1, left: 0 }}
      >
        <Path
          d="
            M0,0
            H100
            V32
            C70,48 30,48 0,32
            Z
          "
          fill={bg}
        />
      </Svg>
    </View>
  );
}
