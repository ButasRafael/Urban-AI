import React from 'react';
import {
  SafeAreaView,
  Image,
  Dimensions,
  Modal,
  TouchableOpacity,
  StyleSheet,
  View,
  ScrollView,
} from 'react-native';
import ImageZoomLib from 'react-native-image-pan-zoom';
import { Video, ResizeMode } from 'expo-av';
import { Feather } from '@expo/vector-icons';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useTheme } from '@shopify/restyle';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Box, Text } from '../components/restylePrimitives';
import type { Theme } from '../theme';
import { spacing } from '../theme';
import { RootStackParamList } from '../navigation/types';
import { API_BASE } from '../config';

type Props = NativeStackScreenProps<RootStackParamList, 'Detail'>;
const ImageZoom: any = ImageZoomLib;

export default function DetailScreen({ route }: Props) {
  const { media, showInfo } = route.params;
  const [showInfoModal, setShowInfoModal] = React.useState(false);
  const theme = useTheme<Theme>();
  const insets = useSafeAreaInsets();
  const descriptions = media.descriptions ?? [];

  // Use the appropriate URL based on media type
  let uri = media.media_type === 'video'
    ? media.annotated_video_url
    : media.annotated_image_url;
  if (!uri) {
    return (
      <SafeAreaView style={{ flex: 1 }}>
        <Box flex={1} bg="background" alignItems="center" justifyContent="center">
          <Text color="text">No media to display</Text>
        </Box>
      </SafeAreaView>
    );
  }
  if (uri.startsWith('/')) uri = `${API_BASE}${uri}`;

  const { width, height } = Dimensions.get('window');
  const mediaHeight = Math.floor(height * 0.72);

  const formattedDate =
    media.created_at ? new Date(media.created_at).toLocaleString() : '-';

  return (
    <SafeAreaView style={{ flex: 1 }}>
      <Box flex={1} bg="background" alignItems="center" justifyContent="center">
        {/* MEDIA FRAME */}
        <Box
          width="100%"
          style={{ maxWidth: 900 }}
          px="m"
        >
          <Box
            bg="card"
            borderRadius="m"
            borderWidth={1}
            borderColor="muted"
            overflow="hidden"
            style={{
              shadowColor: '#000',
              shadowOpacity: 0.08,
              shadowRadius: 8,
              shadowOffset: { width: 0, height: 4 },
              elevation: 4,
            }}
          >
            {/* Media badge (top-left) */}
            <Box
              position="absolute"
              top={8}
              left={8}
              px="s"
              py="xs"
              borderRadius="s"
              borderWidth={1}
              borderColor="muted"
              bg="overlay"
              flexDirection="row"
              alignItems="center"
            >
              <Feather
                name={media.media_type === 'video' ? 'video' : 'image'}
                size={14}
                color="#fff"
              />
              <Text ml="xs" variant="label" style={{ color: '#fff' }}>
                {media.media_type === 'video' ? 'Video' : 'Image'}
              </Text>
            </Box>

            {/* IMAGE */}
            {media.media_type === 'image' ? (
              <ImageZoom
                cropWidth={width}
                cropHeight={mediaHeight}
                imageWidth={width}
                imageHeight={mediaHeight}
                enableCenterFocus={false}
                enableDoubleClickZoom
                minScale={1}
                maxScale={3}
              >
                <Image
                  source={{ uri }}
                  style={{ width, height: mediaHeight, backgroundColor: theme.colors.background }}
                  resizeMode="contain"
                />
              </ImageZoom>
            ) : (
              // VIDEO
              <View style={{ width, height: mediaHeight, backgroundColor: theme.colors.background }}>
                <Video
                  source={{ uri }}
                  style={{ width, height: mediaHeight }}
                  useNativeControls
                  resizeMode={ResizeMode.CONTAIN}
                  shouldPlay={false}
                />
              </View>
            )}

            {/* Caption strip (bottom-left) */}
            <Box
              position="absolute"
              bottom={8}
              left={8}
              right={8}
              px="s"
              py="xs"
              borderRadius="s"
              bg="overlay"
              >
              <Text numberOfLines={1} variant="label" style={{ color: '#fff' }}>
                {media.address || '-'}
              </Text>
              <Text numberOfLines={1} variant="label" style={{ color: '#fff' }}>
                {formattedDate}
              </Text>
            </Box>
          </Box>
        </Box>

        {/* FAB: details (bottom-right) */}
        {showInfo && (
          <TouchableOpacity
            activeOpacity={0.85}
            onPress={() => setShowInfoModal(true)}
            style={[
              styles.fab,
              {
                right: spacing.m,
                bottom: Math.max(insets.bottom, spacing.m),
              },
            ]}
          >
            <Box
              width={56}
              height={56}
              borderRadius="l"
              bg="card"
              alignItems="center"
              justifyContent="center"
              borderWidth={1}
              borderColor="muted"
              style={{
                shadowColor: '#000',
                shadowOpacity: 0.12,
                shadowRadius: 10,
                shadowOffset: { width: 0, height: 6 },
                elevation: 5,
              }}
            >
              <Feather name="info" size={24} color={theme.colors.text} />
            </Box>
          </TouchableOpacity>
        )}

        {/* INFO MODAL (bottom sheet style) */}
        <Modal
          visible={showInfoModal}
          transparent
          animationType="fade"
          onRequestClose={() => setShowInfoModal(false)}
        >
          <View style={styles.modalOverlay}>
            <Box
              bg="surface0"
              borderTopLeftRadius="l"
              borderTopRightRadius="l"
              width="100%"
              style={{ maxHeight: height * 0.7, paddingBottom: insets.bottom || spacing.m }}
            >
              {/* Handle / Header */}
              <Box alignItems="center" pt="s" pb="m">
                <Box
                  width={40}
                  height={4}
                  borderRadius="s"
                  bg="muted"
                  opacity={0.7}
                />
              </Box>

              <Box px="l">
                <Box
                  mb="m"
                  flexDirection="row"
                  alignItems="center"
                  justifyContent="space-between"
                >
                  <Text variant="title" color="text">Details</Text>
                  <TouchableOpacity onPress={() => setShowInfoModal(false)}>
                    <Feather name="x" size={22} color={theme.colors.text} />
                  </TouchableOpacity>
                </Box>

                <ScrollView
                  contentContainerStyle={{ paddingBottom: spacing.l }}
                  showsVerticalScrollIndicator={false}
                >
                  {/* Row: Date */}
                  <Box flexDirection="row" alignItems="center" mb="s">
                    <Feather name="calendar" size={16} color={theme.colors.text} />
                    <Text ml="s" color="text">{formattedDate}</Text>
                  </Box>

                  {/* Row: Address */}
                  <Box flexDirection="row" alignItems="center" mb="s">
                    <Feather name="map-pin" size={16} color={theme.colors.text} />
                    <Text ml="s" color="text">{media.address || '-'}</Text>
                  </Box>

                  {/* Row: Type */}
                  <Box flexDirection="row" alignItems="center" mb="s">
                    <Feather name={media.media_type === 'video' ? 'video' : 'image'} size={16} color={theme.colors.text} />
                    <Text ml="s" color="text" textTransform="capitalize">
                      {media.media_type}
                    </Text>
                  </Box>

                  {/* Row: Tags / Classes */}
                  <Box flexDirection="row" alignItems="flex-start" mb="s">
                    <Feather name="tag" size={16} color={theme.colors.text} style={{ marginTop: 2 }} />
                    <Box ml="s" flexDirection="row" flexWrap="wrap">
                      {(media.predicted_classes ?? []).length ? (
                        (media.predicted_classes ?? []).map((cls, idx) => (
                          <Box
                            key={`${cls}-${idx}`}
                            mr="s"
                            mb="s"
                            px="s"
                            py="xs"
                            borderRadius="s"
                            bg="card"
                            borderWidth={1}
                            borderColor="muted"
                          >
                            <Text color="onSurface" variant="label">{cls}</Text>
                          </Box>
                        ))
                      ) : (
                        <Text color="text">-</Text>
                      )}
                    </Box>
                  </Box>

                  {/* Summary Description */}
                  {media.summary_description && (
                    <Box mb="m">
                      <Box flexDirection="row" alignItems="flex-start" mb="s">
                        <Feather name="file-text" size={16} color={theme.colors.primary500} style={{ marginTop: 2 }} />
                        <Text ml="s" variant="subtitle" color="primary500">Descriere generală</Text>
                      </Box>
                      <Box ml="l" p="m" bg="card" borderRadius="s" borderWidth={1} borderColor="border">
                        <Text color="text" lineHeight={22}>
                          {media.summary_description}
                        </Text>
                      </Box>
                    </Box>
                  )}

                  {/* Summary Solution */}
                  {media.summary_solution && (
                    <Box mb="m">
                      <Box flexDirection="row" alignItems="flex-start" mb="s">
                        <Feather name="tool" size={16} color={theme.colors.success} style={{ marginTop: 2 }} />
                        <Text ml="s" variant="subtitle" color="success">Plan de acțiune</Text>
                      </Box>
                      <Box ml="l" p="m" bg="card" borderRadius="s" borderWidth={1} borderColor="border">
                        <Text color="text" lineHeight={22}>
                          Authorities will {media.summary_solution}
                        </Text>
                      </Box>
                    </Box>
                  )}

                  {/* Individual Descriptions (fallback if no summary) */}
                  {!media.summary_description && descriptions.length > 0 && (
                    <Box flexDirection="row" alignItems="flex-start" mb="s">
                      <Feather name="align-left" size={16} color={theme.colors.text} style={{ marginTop: 2 }} />
                      <Box ml="s" flex={1}>
                        {descriptions.map((d, i) => (
                          <Text key={i} color="text" mb="xs">
                            • {d}
                          </Text>
                        ))}
                      </Box>
                    </Box>
                  )}
                </ScrollView>
              </Box>
            </Box>
          </View>
        </Modal>
      </Box>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  fab: {
    position: 'absolute',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.45)',
    justifyContent: 'flex-end',
  },
});
