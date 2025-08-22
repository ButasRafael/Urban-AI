import React, { useRef, useEffect } from 'react';
import { View, Animated, StyleSheet } from 'react-native';
import { useTheme } from '@shopify/restyle';
import type { Theme } from '../../theme';
import { LinearGradient } from 'expo-linear-gradient';
import Svg, { Path } from 'react-native-svg';
import { Box, Text } from '../restylePrimitives';
import { alpha, opacity } from '../../theme/utils';
import { motion } from '../../theme/motion';

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
        duration: motion.dur.sm,
        easing: motion.curve.standard,
        useNativeDriver: true,
      }),
      Animated.spring(logoScale, { toValue: 1, useNativeDriver: true }),
    ]).start();
  }, [logoScale]);

  const bg = theme.colors.background;

  return (
    <View style={{ height, position: 'relative' }}>
      {/* gradient background - matching web design */}
      <LinearGradient
        colors={[
          theme.colors.primary300,
          theme.colors.primary500,
          theme.colors.secondary500,
        ]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={StyleSheet.absoluteFill}
      />

      {/* Subtle overlay for better text contrast */}
      <View
        style={[
          StyleSheet.absoluteFill,
          { backgroundColor: alpha('#000', opacity.overlayWeak) },
        ]}
      />

      {/* Web-inspired gradient blobs */}
      {blobs && (
        <>
          <View
            pointerEvents="none"
            style={{
              position: 'absolute',
              top: -50,
              right: -50,
              width: 180,
              height: 180,
              borderRadius: 999,
              backgroundColor: alpha(theme.colors.primary100, opacity.overlayStrong),
            }}
          />
          <View
            pointerEvents="none"
            style={{
              position: 'absolute',
              top: 40,
              left: -40,
              width: 120,
              height: 120,
              borderRadius: 999,
              backgroundColor: alpha(theme.colors.secondary300, opacity.overlayWeak),
            }}
          />
          <View
            pointerEvents="none"
            style={{
              position: 'absolute',
              bottom: 20,
              right: 30,
              width: 80,
              height: 80,
              borderRadius: 999,
              backgroundColor: alpha(theme.colors.surface0, opacity.overlayWeak),
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
        <Box flex={1}>
          <Text
            variant="display"
            style={{
              color: '#fff',
              fontSize: 28,
              fontWeight: '800',
              letterSpacing: 0.3,
              textShadowColor: 'rgba(0,0,0,0.3)',
              textShadowOffset: { width: 0, height: 1 },
              textShadowRadius: 3,
            }}
          >
            {title}
          </Text>
        </Box>

        {subtitle ? (
          <Box position="absolute" left={24} bottom={-6}>
            <Text
              variant="body"
              style={{
                color: '#ffffffdd',
                fontSize: 14,
                lineHeight: 18,
                textShadowColor: 'rgba(0,0,0,0.2)',
                textShadowOffset: { width: 0, height: 1 },
                textShadowRadius: 2,
              }}
            >
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
