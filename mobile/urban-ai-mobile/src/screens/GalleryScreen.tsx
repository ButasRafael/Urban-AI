import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import {
  SafeAreaView,
  FlatList,
  Animated as RNAnimated,
  ActivityIndicator,
  Pressable,
  useWindowDimensions,
  StyleSheet,
  View,
  Platform,
} from 'react-native';
import Animated from 'react-native-reanimated';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useFocusEffect } from '@react-navigation/native';
import { useTheme } from '@shopify/restyle';
import * as VideoThumbnails from 'expo-video-thumbnails';
import { LinearGradient } from 'expo-linear-gradient';
import { Feather } from '@expo/vector-icons';

import { Box, Text } from '../components/restylePrimitives';
import type { Theme } from '../theme';
import { spacing } from '../theme';
import { RootStackParamList } from '../navigation/types';
import client from '../api/client';
import { API_BASE } from '../config';
import { alpha } from '../theme/utils';
import { Skeleton } from '../components/ui/Skeleton';
import { usePressable } from '../hooks/usePressable';

type MediaItem = {
  media_id: number;
  media_type: 'image' | 'video';
  annotated_image_url?: string;
  annotated_video_url?: string;
  created_at?: string;
  address: string;
  predicted_classes: string[];
  descriptions?: string[];
  summary_description?: string;
  summary_solution?: string;
  processing_status: 'pending' | 'processing' | 'completed' | 'failed'; // Made non-optional
  task_id?: string;
};

type Props = NativeStackScreenProps<RootStackParamList, 'Gallery'>;

type GalleryItemProps = {
  item: MediaItem;
  thumbUri?: string;
  size: number;
  marginRight: number;
  navigation: Props['navigation'];
  isLastInRow: boolean;
};

const formatWhen = (iso?: string) => {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleDateString('ro-RO', { day: '2-digit', month: 'short' });
};

const firstNonEmpty = (arr?: string[]) => (arr && arr.length ? arr[0] : '');

const GalleryItem: React.FC<GalleryItemProps> = React.memo(
  ({ item, thumbUri, size, marginRight, navigation, isLastInRow }) => {
    const theme = useTheme<Theme>();
    const fadeIn = useRef(new RNAnimated.Value(0)).current;
    const pressable = usePressable({ scaleTo: 0.97, pressedOpacity: 0.96, spring: true });

    const onPressIn = pressable.onPressIn;
    const onPressOut = pressable.onPressOut;

    // Normalize status robustly - treat undefined/null as pending
    const isProcessing = !item.processing_status ||
                        item.processing_status === 'pending' ||
                        item.processing_status === 'processing';
    const isFailed = item.processing_status === 'failed';
    const isCompleted = item.processing_status === 'completed';
    const hasArtifact = !!(item.annotated_image_url || item.annotated_video_url);

    // Only show image if EXPLICITLY completed with artifact
    const showImage = isCompleted && hasArtifact;

    // For videos use server thumbnail first, then generated, never the mp4
    const base = item.media_type === 'image'
      ? item.annotated_image_url
      : (item.annotated_image_url || thumbUri || undefined);
    const resolved = base?.startsWith('/') ? API_BASE + base : base;

    const onLoad = () =>
      RNAnimated.timing(fadeIn, { toValue: 1, duration: 250, useNativeDriver: true }).start();

    const goDetail = () => {
      // Navigation based ONLY on completion status
      if (!isCompleted) {
        // Navigate to Processing for ANY non-completed state
        navigation.navigate('Processing', {
          taskId: item.task_id || '',
          mediaId: item.media_id,
        });
      } else {
        // Only go to Detail for explicitly completed items
        navigation.navigate('Detail', {
          media: {
            ...item,
            predicted_classes: item.predicted_classes || [],
            descriptions: item.descriptions || [],
            summary_description: item.summary_description,
            summary_solution: item.summary_solution,
          },
          showInfo: true,
        });
      }
    };

    return (
      <Pressable
        onPress={goDetail}
        onPressIn={onPressIn}
        onPressOut={onPressOut}
        android_ripple={{ color: theme.colors.primary100 }}
        accessibilityRole="imagebutton"
        accessibilityLabel={item.media_type === 'video' ? 'Video încărcat' : 'Imagine încărcată'}
        style={[
          styles.cardContainer,
          {
            width: size,
            height: size,
            marginRight: isLastInRow ? 0 : marginRight,
            marginBottom: marginRight,
          },
        ]}
      >
        <Animated.View
          style={[
            styles.card,
            {
              borderRadius: theme.borderRadii.m,
              backgroundColor: isProcessing || isFailed
                ? theme.colors.surface0
                : theme.colors.card,
              borderColor: isProcessing
                ? theme.colors.primary500
                : isFailed
                ? theme.colors.error
                : theme.colors.muted,
              borderWidth: isProcessing || isFailed ? 2 : 1,
              ...(pressable.animatedStyle as any),
              ...(Platform.OS === 'ios'
                ? {
                    shadowColor: '#000',
                    shadowOpacity: 0.08,
                    shadowRadius: 8,
                    shadowOffset: { width: 0, height: 4 },
                  }
                : { elevation: 2 }),
            },
          ]}
        >
          {/* Main content */}
          {showImage && resolved ? (
            <RNAnimated.Image
              source={{ uri: resolved }}
              style={[styles.media, { opacity: fadeIn }]}
              resizeMode="cover"
              onLoad={onLoad}
            />
          ) : (
            <Box
              flex={1}
              alignItems="center"
              justifyContent="center"
              backgroundColor={isProcessing ? "primary100" : isFailed ? "surface50" : "surface0"}
            >
              {isProcessing && (
                <>
                  <ActivityIndicator size="large" color={theme.colors.primary500} />
                  <Text variant="label" color="primary500" mt="s" fontWeight="600">
                    {(!item.processing_status || item.processing_status === 'pending') ? 'În așteptare...' : 'Se procesează...'}
                  </Text>
                </>
              )}
              {isFailed && (
                <>
                  <Feather name="alert-circle" size={32} color={theme.colors.error} />
                  <Text variant="label" color="error" mt="s" fontWeight="600">
                    Procesare eșuată
                  </Text>
                </>
              )}
              {!isProcessing && !isFailed && !hasArtifact && (
                <>
                  <Feather name="image" size={32} color={theme.colors.muted} />
                  <Text variant="label" color="muted" mt="s">
                    Fără previzualizare
                  </Text>
                </>
              )}
            </Box>
          )}

          {/* Processing badge overlay */}
          {isProcessing && (
            <Box
              position="absolute"
              top={8}
              right={8}
              px="s"
              py="xs"
              borderRadius="s"
              backgroundColor="primary500"
              style={{
                borderWidth: 1,
                borderColor: theme.colors.primary700,
              }}
            >
              <Box flexDirection="row" alignItems="center">
                <ActivityIndicator size="small" color="#fff" />
                <Text variant="label" style={{ color: '#fff', marginLeft: 4 }}>
                  {(!item.processing_status || item.processing_status === 'pending') ? 'Waiting' : 'Processing'}
                </Text>
              </Box>
            </Box>
          )}

          {/* Failed badge overlay */}
          {isFailed && (
            <Box
              position="absolute"
              top={8}
              right={8}
              px="s"
              py="xs"
              borderRadius="s"
              backgroundColor="error"
              style={{
                borderWidth: 1,
                borderColor: theme.colors.error,
              }}
            >
              <Text variant="label" style={{ color: '#fff' }}>
                Failed
              </Text>
            </Box>
          )}

          {/* Play button overlay for videos */}
          {item.media_type === 'video' && !isProcessing && !isFailed && showImage && (
            <View style={styles.centerOverlay} pointerEvents="none">
              <View style={styles.playBadge}>
                <Feather name="play" size={22} color="#fff" />
              </View>
            </View>
          )}

          {/* Video badge (top-left corner) */}
          {item.media_type === 'video' && !isProcessing && !isFailed && (
            <Box
              position="absolute"
              top={8}
              left={8}
              px="s"
              py="xs"
              borderRadius="s"
              bg="overlay"
              style={{
                borderWidth: 1,
                borderColor: 'rgba(255,255,255,0.2)',
              }}
            >
              <Text variant="label" style={{ color: '#fff' }}>
                Video
              </Text>
            </Box>
          )}

          {/* Bottom gradient + meta (only for completed items) */}
          {showImage && (
            <>
              <LinearGradient
                colors={['transparent', alpha('#000', 0.55)]}
                style={styles.gradient}
              />
              <View style={styles.metaRow}>
                <Text numberOfLines={1} style={styles.metaText}>
                  {item.summary_description ?
                    item.summary_description.slice(0, 50) + (item.summary_description.length > 50 ? '...' : '') :
                    (firstNonEmpty(item.predicted_classes) || item.address || '—')}
                </Text>
                <Text style={styles.metaDot}>•</Text>
                <Text style={styles.metaText}>{formatWhen(item.created_at) || ''}</Text>
              </View>
            </>
          )}
        </Animated.View>
      </Pressable>
    );
  }
);

export default function GalleryScreen({ navigation }: Props) {
  const theme = useTheme<Theme>();
  const { width } = useWindowDimensions();

  const [data, setData] = useState<MediaItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [thumbs, setThumbs] = useState<Record<number, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'images' | 'videos'>('all');

  const horizontalPad = theme.spacing.m;
  const gutter = theme.spacing.s;

  const columns = useMemo(() => {
    if (width >= 1000) return 5;
    if (width >= 820) return 4;
    if (width >= 560) return 3;
    return 2;
  }, [width]);

  const itemSize = useMemo(() => {
    const totalGutter = gutter * (columns - 1);
    const available = width - (horizontalPad * 2) - totalGutter;
    return Math.floor(available / columns);
  }, [width, horizontalPad, gutter, columns]);

  const fetchData = useCallback(async () => {
    try {
      setError(null);
      const r = await client.get<MediaItem[]>('/infer/list?include_pending=true');
      const items = r.data ?? [];
      setData(items);

      items.forEach((item) => {
        if (item.media_type === 'video' && item.annotated_video_url) {
          const uri = item.annotated_video_url.startsWith('/')
            ? API_BASE + item.annotated_video_url
            : item.annotated_video_url;
          VideoThumbnails.getThumbnailAsync(uri, { time: 1000 })
            .then(({ uri: thumbUri }) =>
              setThumbs((t) => ({ ...t, [item.media_id]: thumbUri }))
            )
            .catch(() => {});
        }
      });
    } catch (e: any) {
      setError(e?.message || 'Eroare la încărcarea încărcărilor tale.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Refresh data when screen gains focus
  useFocusEffect(
    useCallback(() => {
      fetchData();
    }, [fetchData])
  );

  const onRefresh = () => {
    setRefreshing(true);
    fetchData();
  };

  const filtered = useMemo(() => {
    if (filter === 'images') return data.filter((i) => i.media_type === 'image');
    if (filter === 'videos') return data.filter((i) => i.media_type === 'video');
    return data;
  }, [data, filter]);

  if (loading && !refreshing) {
    const placeholders = Array.from({ length: columns * 4 }, (_, i) => i);
    return (
      <SafeAreaView style={{ flex: 1 }}>
        <Box flex={1} bg="background" pt="m">
          <FlatList
            data={placeholders}
            keyExtractor={(i) => `ph-${i}`}
            numColumns={columns}
            contentContainerStyle={{
              paddingHorizontal: horizontalPad,
            }}
            renderItem={({ index }) => {
              const isLastInRow = (index + 1) % columns === 0;
              return (
                <Box
                  style={{
                    width: itemSize,
                    height: itemSize,
                    marginRight: isLastInRow ? 0 : gutter,
                    marginBottom: gutter,
                  }}
                >
                  <Skeleton width={itemSize} height={itemSize} radius="m" />
                </Box>
              );
            }}
          />
        </Box>
      </SafeAreaView>
    );
  }

  if (error) {
    return (
      <SafeAreaView style={{ flex: 1 }}>
        <Box flex={1} bg="background" p="l" alignItems="center" justifyContent="center">
          <Text color="error" mb="m">
            {error}
          </Text>
          <Pressable
            onPress={fetchData}
            style={({ pressed }) => [
              styles.retryBtn,
              {
                borderColor: theme.colors.muted,
                backgroundColor: pressed ? theme.colors.surface0 : theme.colors.card,
              },
            ]}
          >
            <Text variant="label" color="text">
              Reîncearcă
            </Text>
          </Pressable>
        </Box>
      </SafeAreaView>
    );
  }

  if (!filtered.length) {
    return (
      <SafeAreaView style={{ flex: 1 }}>
        <Box flex={1} bg="background" p="l" alignItems="center" justifyContent="center">
          <Text variant="title" color="text" mb="s">
            Galeria Mea
          </Text>
          <Text variant="label" color="muted" textAlign="center">
            Nu ai încărcat nimic încă. Întoarce-te la ecranul de încărcare pentru a adăuga conținut.
          </Text>
        </Box>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={{ flex: 1 }}>
      <Box flex={1} bg="background" pt="m">
        <Box
          flexDirection="row"
          alignItems="center"
          justifyContent="space-between"
          mb="m"
          px="m"
        >
          <Text variant="title" color="text">
            Galeria Mea
          </Text>
          <Box flexDirection="row">
            {[
              { key: 'all', label: 'Toate' },
              { key: 'images', label: 'Imagini' },
              { key: 'videos', label: 'Video' },
            ].map((f) => {
              const active = filter === (f.key as typeof filter);
              return (
                <Pressable
                  key={f.key}
                  onPress={() => setFilter(f.key as typeof filter)}
                  style={({ pressed }) => [
                    styles.chip,
                    {
                      borderColor: theme.colors.muted,
                      backgroundColor: active
                        ? theme.colors.primary100
                        : pressed
                        ? theme.colors.surface0
                        : theme.colors.card,
                      marginLeft: spacing.s,
                    },
                  ]}
                  accessibilityRole="button"
                >
                  <Text
                    variant="label"
                    color={active ? 'primary500' : 'text'}
                    numberOfLines={1}
                  >
                    {f.label}
                  </Text>
                </Pressable>
              );
            })}
          </Box>
        </Box>

        <FlatList
          data={filtered}
          keyExtractor={(i) => i.media_id.toString()}
          numColumns={columns}
          contentContainerStyle={{
            paddingHorizontal: horizontalPad,
            paddingBottom: spacing.l,
          }}
          refreshing={refreshing}
          onRefresh={onRefresh}
          initialNumToRender={12}
          windowSize={7}
          removeClippedSubviews
          getItemLayout={(_, index) => {
            const row = Math.floor(index / columns);
            const rowHeight = itemSize + gutter;
            return {
              length: rowHeight,
              offset: row * rowHeight,
              index,
            };
          }}
          renderItem={({ item, index }) => {
            const isLastInRow = (index + 1) % columns === 0;
            return (
              <GalleryItem
                item={item}
                thumbUri={thumbs[item.media_id]}
                size={itemSize}
                marginRight={gutter}
                navigation={navigation}
                isLastInRow={isLastInRow}
              />
            );
          }}
        />
      </Box>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  centerOverlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
  playBadge: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(0,0,0,0.45)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  cardContainer: {
    overflow: 'hidden',
  },
  card: {
    flex: 1,
    borderWidth: 1,
    overflow: 'hidden',
  },
  media: {
    width: '100%',
    height: '100%',
  },
  gradient: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    height: 56,
  },
  metaRow: {
    position: 'absolute',
    left: 10,
    right: 10,
    bottom: 8,
    flexDirection: 'row',
    alignItems: 'center',
  },
  metaText: {
    color: '#fff',
    fontSize: 12,
    maxWidth: '42%',
  },
  metaDot: {
    color: '#ffffffaa',
    paddingHorizontal: 6,
  },
  chip: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    borderWidth: 1,
  },
  retryBtn: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 10,
    borderWidth: 1,
  },
});