// src/screens/DetailScreen.tsx
import React from 'react';
import {
  SafeAreaView,
  Image,
  Dimensions,
  Modal,
  TouchableOpacity,
  StyleSheet,
  View,
} from 'react-native';
import ImageZoomLib from 'react-native-image-pan-zoom';
import { Video, ResizeMode } from 'expo-av';
import { Feather } from '@expo/vector-icons';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useTheme } from '@shopify/restyle';

import { Box, Text } from '../components/restylePrimitives';
import type { Theme } from '../theme';
import { spacing } from '../theme';
import { RootStackParamList } from '../navigation/types';
import { API_BASE } from '../config';

type Props = NativeStackScreenProps<RootStackParamList, 'Detail'>;
const ImageZoom: any = ImageZoomLib;              // avoid TS errors on lib types

export default function DetailScreen({ route }: Props) {
  const { media, showInfo } = route.params;
  const [showInfoModal, setShowInfoModal] = React.useState(false);
  const theme = useTheme<Theme>();
  const descriptions = media.descriptions ?? [];

  // -------------------- derive media uri -----------------------
  let uri = media.annotated_image_url ?? media.annotated_video_url;
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

  // ----------------------------- UI ----------------------------
  return (
    <SafeAreaView style={{ flex: 1 }}>
      <Box flex={1} bg="background" alignItems="center" justifyContent="center">
        {/*  more / info button  */}
        {showInfo && (
          <TouchableOpacity
            style={styles.moreBtn}
            onPress={() => setShowInfoModal(true)}
          >
            <Feather name="more-vertical" size={24} color={theme.colors.text} />
          </TouchableOpacity>
        )}

        {/*  IMAGE  */}
        {media.annotated_image_url ? (
          <ImageZoom
            cropWidth={width}
            cropHeight={height * 0.8}
            imageWidth={width}
            imageHeight={height * 0.8}
            enableCenterFocus={false}
          >
            <Image
              source={{ uri }}
              style={{ width, height: height * 0.8 }}
              resizeMode="contain"
            />
          </ImageZoom>
        ) : (
          /*  VIDEO  */
          <View style={{ width, height: height * 0.8 }}>
            <Video
              source={{ uri }}
              style={{ width, height: height * 0.8 }}
              useNativeControls
              resizeMode={ResizeMode.CONTAIN}
              shouldPlay
            />
          </View>
        )}

        {/*  INFO MODAL  */}
        <Modal
          visible={showInfoModal}
          transparent
          animationType="slide"
          onRequestClose={() => setShowInfoModal(false)}
        >
          <View style={styles.modalOverlay}>
            <Box
              bg="surface0"
              p="l"
              borderRadius="s"
              width="80%"
            >
              <Text mb="s" color="text">
                <Text fontWeight="600">Date: </Text>
                {media.created_at}
              </Text>
              <Text mb="s" color="text">
                <Text fontWeight="600">Address: </Text>
                {media.address}
              </Text>
              <Text mb="s" color="text">
                <Text fontWeight="600">Type: </Text>
                {media.media_type}
              </Text>
              <Text mb="s" color="text">
                <Text fontWeight="600">Classes: </Text>
                {(media.predicted_classes ?? []).length
                  ? media.predicted_classes.join(', ')
                  : '-'}
              </Text>
              <Text mb="s" color="text">
                <Text fontWeight="600">Descriptions: </Text>
                {descriptions.length > 0 ? (
                  descriptions.map((d, i) => (
                    <Text key={i}>
                      {i + 1}. {d}
                      {i < descriptions.length - 1 ? '\n' : ''}
                    </Text>
                  ))
                ) : (
                  '-'
                )}
              </Text>


              <TouchableOpacity
                style={{ alignSelf: 'flex-end', marginTop: theme.spacing.m }}
                onPress={() => setShowInfoModal(false)}
              >
                <Text color="text" fontWeight="600">
                  Close
                </Text>
              </TouchableOpacity>
            </Box>
          </View>
        </Modal>
      </Box>
    </SafeAreaView>
  );
}

/* --------- only keeps numeric / positioning constants --------- */
const styles = StyleSheet.create({
  moreBtn: {
    position: 'absolute',
    top: spacing.m,
    right: spacing.m,
    zIndex: 10,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
});
