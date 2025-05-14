// src/screens/GalleryScreen.tsx
import React, { useEffect, useState, useRef } from 'react';
import {
  SafeAreaView,
  FlatList,
  Animated,
  TouchableWithoutFeedback,
  Image,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useTheme } from '@shopify/restyle';
import * as VideoThumbnails from 'expo-video-thumbnails';

import { Box, Text } from '../components/restylePrimitives';
import type { Theme } from '../theme';
import { spacing } from '../theme';
import { RootStackParamList } from '../navigation/types';
import client from '../api/client';
import { API_BASE } from '../config';


/* ------------------------- types -------------------------- */
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
  navigation: Props['navigation'];
};

/* ------------------- card component ----------------------- */
const GalleryItem: React.FC<GalleryItemProps> = ({
  item,
  thumbUri,
  navigation,
}) => {
  const scale = useRef(new Animated.Value(1)).current;
  const theme = useTheme<Theme>();

  const handlePressIn = () =>
    Animated.spring(scale, { toValue: 0.96, useNativeDriver: true }).start();
  const handlePressOut = () =>
    Animated.spring(scale, { toValue: 1, useNativeDriver: true }).start();

  const uri =
    item.media_type === 'image'
      ? item.annotated_image_url!
      : thumbUri || 'placeholder-white-frame.png';
  const resolved = uri.startsWith('/') ? API_BASE + uri : uri;

  return (
    <TouchableWithoutFeedback
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      onPress={() =>
        navigation.navigate('Detail', {
          media: { ...item, predicted_classes: item.predicted_classes, descriptions: item.descriptions },
          showInfo: true,
        })
      }
    >
      <Animated.View
        style={[
          styles.card,
          { borderRadius: theme.borderRadii.s, transform: [{ scale }] },
        ]}
      >
        <Image source={{ uri: resolved }} style={styles.thumb} resizeMode="cover" />
      </Animated.View>
    </TouchableWithoutFeedback>
  );
};

/* --------------------- screen ------------------------------ */
export default function GalleryScreen({ navigation }: Props) {
  const theme = useTheme<Theme>();
  const [data, setData] = useState<MediaItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [thumbs, setThumbs] = useState<Record<number, string>>({});
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      const r = await client.get<MediaItem[]>('/infer/list');
      setData(r.data);

      // create thumbnails for videos
      r.data.forEach(item => {
        if (item.media_type === 'video' && item.annotated_video_url) {
          const uri = item.annotated_video_url.startsWith('/')
            ? API_BASE + item.annotated_video_url
            : item.annotated_video_url;
          VideoThumbnails.getThumbnailAsync(uri, { time: 1000 })
            .then(({ uri: thumbUri }) =>
              setThumbs(t => ({ ...t, [item.media_id]: thumbUri })),
            )
            .catch(() => {});
        }
      });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  /* ----------  loading   ---------- */
  if (loading) {
    return (
      <SafeAreaView style={{ flex: 1 }}>
        <Box flex={1} bg="background" p="m" alignItems="center">
          <Text variant="title" textAlign="center" color="text" mb="m">
            My Uploads
          </Text>
          <ActivityIndicator
            size="large"
            color={theme.colors.text}
            style={{ marginTop: theme.spacing.l }}
          />
        </Box>
      </SafeAreaView>
    );
  }

  /* ----------  error   ---------- */
  if (error) {
    return (
      <SafeAreaView style={{ flex: 1 }}>
        <Box flex={1} bg="background" alignItems="center" justifyContent="center">
          <Text color="error">{error}</Text>
        </Box>
      </SafeAreaView>
    );
  }

  /* ----------  success   ---------- */
  const items = data.filter(i => i.annotated_image_url || i.annotated_video_url);

  const onRefresh = () => {
    setRefreshing(true);
    fetchData();
  };

  return (
    <SafeAreaView style={{ flex: 1 }}>
      <Box flex={1} bg="background" p="m">
        <Text variant="title" textAlign="center" color="text" mb="m">
          My Uploads
        </Text>

        <FlatList
          data={items}
          keyExtractor={i => i.media_id.toString()}
          numColumns={3}
          columnWrapperStyle={{ justifyContent: 'space-between' }}
          contentContainerStyle={{ paddingBottom: spacing.l }}
          refreshing={refreshing}
          onRefresh={onRefresh}
          renderItem={({ item }) => (
            <GalleryItem
              item={item}
              thumbUri={thumbs[item.media_id]}
              navigation={navigation}
            />
          )}
        />
      </Box>
    </SafeAreaView>
  );
}

/* ----------  styles (non-colour props only)  ---------- */
const styles = StyleSheet.create({
  card: {
    width: '30%',
    aspectRatio: 1,
    overflow: 'hidden',
    marginBottom: spacing.m,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  thumb: {
    width: '100%',
    height: '100%',
  },
});
