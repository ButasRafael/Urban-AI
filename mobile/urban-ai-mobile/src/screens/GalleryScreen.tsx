import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import {
  SafeAreaView,
  FlatList,
  Animated,
  Image,
  ActivityIndicator,
  Pressable,
  useWindowDimensions,
  StyleSheet,
  View,
  Platform,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useTheme } from '@shopify/restyle';
import * as VideoThumbnails from 'expo-video-thumbnails';
import { LinearGradient } from 'expo-linear-gradient';

import { Box, Text } from '../components/restylePrimitives';
import type { Theme } from '../theme';
import { spacing } from '../theme';
import { RootStackParamList } from '../navigation/types';
import client from '../api/client';
import { API_BASE } from '../config';

type MediaItem = {
  media_id: number;
  media_type: 'image' | 'video';
  annotated_image_url?: string;
  annotated_video_url?: string;
  created_at?: string;
  address: string;
  predicted_classes: string[];
  descriptions?: string[];
};

type Props = NativeStackScreenProps<RootStackParamList, 'Gallery'>;

type GalleryItemProps = {
  item: MediaItem;
  thumbUri?: string;
  size: number;
  marginRight: number;
  navigation: Props['navigation'];
  index: number;
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
    const pressScale = useRef(new Animated.Value(1)).current;
    const fadeIn = useRef(new Animated.Value(0)).current;

    const onPressIn = () =>
      Animated.spring(pressScale, {
        toValue: 0.97,
        useNativeDriver: true,
        speed: 50,
        bounciness: 6,
      }).start();
    const onPressOut = () =>
      Animated.spring(pressScale, {
        toValue: 1,
        useNativeDriver: true,
        speed: 50,
        bounciness: 6,
      }).start();

    const base = item.media_type === 'image'
    ? item.annotated_image_url
    : (thumbUri || item.annotated_video_url);const resolved = base?.startsWith('/') ? API_BASE + base : base;
    const onLoad = () =>
      Animated.timing(fadeIn, { toValue: 1, duration: 250, useNativeDriver: true }).start();

    const goDetail = () =>
      navigation.navigate('Detail', {
        media: {
          ...item,
          predicted_classes: item.predicted_classes,
          descriptions: item.descriptions,
        },
        showInfo: true,
      });

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
              backgroundColor: theme.colors.surface0,
              borderColor: theme.colors.muted,
              transform: [{ scale: pressScale }],
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
          {/* media */}
          {resolved ? (
            <Animated.Image
              source={{ uri: resolved }}
              style={[styles.media, { opacity: fadeIn }]}
              resizeMode="cover"
              onLoad={onLoad}
            />
          ) : (
            <Box flex={1} alignItems="center" justifyContent="center">
              <Text variant="label" color="muted">
                Fără previzualizare
              </Text>
            </Box>
          )}

          {/* top-right badge for video */}
          {item.media_type === 'video' ? (
            <Box
              position="absolute"
              top={8}
              right={8}
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
          ) : null}

          {/* bottom gradient + meta */}
          <LinearGradient
            colors={['transparent', 'rgba(0,0,0,0.55)']}
            style={styles.gradient}
          />
          <View style={styles.metaRow}>
            <Text numberOfLines={1} style={styles.metaText}>
              {firstNonEmpty(item.predicted_classes) || item.address || '—'}
            </Text>
            <Text style={styles.metaDot}>•</Text>
            <Text style={styles.metaText}>{formatWhen(item.created_at) || ''}</Text>
          </View>
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
    const available = width - horizontalPad * 2 - totalGutter;
    return Math.floor(available / columns);
  }, [width, horizontalPad, gutter, columns]);

  const fetchData = useCallback(async () => {
    try {
      setError(null);
      const r = await client.get<MediaItem[]>('/infer/list');
      const items = r.data ?? [];
      setData(items);

      // Create thumbnails for videos
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

  const onRefresh = () => {
    setRefreshing(true);
    fetchData();
  };

  const filtered = useMemo(() => {
    const withMedia = data.filter((i) => i.annotated_image_url || i.annotated_video_url);
    if (filter === 'images') return withMedia.filter((i) => i.media_type === 'image');
    if (filter === 'videos') return withMedia.filter((i) => i.media_type === 'video');
    return withMedia;
  }, [data, filter]);

  if (loading) {
    return (
      <SafeAreaView style={{ flex: 1 }}>
        <Box flex={1} bg="background" p="m" alignItems="center" justifyContent="center">
          <ActivityIndicator size="large" color={theme.colors.text} />
          <Text variant="label" color="muted" mt="m">
            Se încarcă galeria…
          </Text>
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
            My Uploads
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
      <Box flex={1} bg="background" p="m">
        {/* Header */}
        <Box
          flexDirection="row"
          alignItems="center"
          justifyContent="space-between"
          mb="m"
        >
          <Text variant="title" color="text">
            My Uploads
          </Text>

          {/* Tiny filter chips */}
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
            paddingBottom: spacing.l,
            paddingLeft: horizontalPad,
            paddingRight: horizontalPad,
          }}
          refreshing={refreshing}
          onRefresh={onRefresh}
          initialNumToRender={12}
          windowSize={7}
          removeClippedSubviews
          getItemLayout={(_, index) => {
            // Helps scroll perf a bit (square items, fixed size rows)
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
                index={index}
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
  cardContainer: {
    overflow: 'hidden',
  },
  card: {
    flex: 1,
    borderWidth: 1,
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
