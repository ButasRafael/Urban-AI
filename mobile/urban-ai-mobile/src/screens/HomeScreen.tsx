// src/screens/HomeScreen.tsx  – Restyle-ready, fully fixed
//--------------------------------------------------------
import React from 'react';
import {
  SafeAreaView,
  Alert,
  StyleSheet,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useTheme } from '@shopify/restyle';

import { RootStackParamList } from '../navigation/types';
import StyledButton from '../components/StyledButton';
import FullScreenLoader from '../components/FullScreenLoader';
import { Box, Text } from '../components/restylePrimitives';
import type { Theme } from '../theme';
// re-import for StyleSheet at bottom:
import { spacing } from '../theme';

import { Feather } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import * as Location from 'expo-location';
import client from '../api/client';
import { logout } from '../api/auth';
import AddressPickerModal from '../components/AddressPickerModal';
import MapAdjustModal     from '../components/MapAdjustModal';
import type { LatLng } from 'react-native-maps';
import { GOOGLE_API_KEY } from '../config';

type Props = NativeStackScreenProps<RootStackParamList, 'Home'>;

export default function HomeScreen({ navigation }: Props) {
  // pull spacing scale from the Restyle theme
  const { spacing: sp } = useTheme<Theme>();
  const [uploading,     setUploading]       = React.useState(false);
  const [addrPickerVisible, setAddrPickerVisible] = React.useState(false);
  const [mapAdjustVisible, setMapAdjustVisible]   = React.useState(false);
  const pinRef = React.useRef<LatLng | null>(null);


  const stashRef = React.useRef<{
    uri: string; name: string; mime: string; useSam: boolean; isVideo: boolean;
  }|null>(null);

  async function reverseGeocode({ latitude, longitude }: LatLng) {
  const url = `https://maps.googleapis.com/maps/api/geocode/json?latlng=${latitude},${longitude}&key=${GOOGLE_API_KEY}`;
  const res = await fetch(url);
  const json = await res.json();
  return json.results?.[0]?.formatted_address ?? '';
}

  async function handleLogout() {
    try {
      await logout();
      navigation.replace('Login');
    } catch (e: any) {
      Alert.alert('Logout failed', e.message);
    }
  }

  const pickAndUploadImage = async () => {
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
    });
    if (res.canceled) return;
    const { uri } = res.assets[0];
    stashRef.current = {
      uri,
      name: uri.split('/').pop()!,
      mime: 'image/jpeg',
      useSam: false,
      isVideo: false,
    };
    Alert.alert('Use SAM masks?', 'Choose segmentation mode for this image:', [
      { text: 'No (YOLO only)', onPress: () => askLocation(false) },
      { text: 'Yes (YOLO + SAM)',  onPress: () => askLocation(true) },
    ]);
  };

  const pickAndUploadVideo = async () => {
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
    });
    if (res.canceled) return;
    const { uri } = res.assets[0];
    stashRef.current = {
      uri,
      name: uri.split('/').pop()!,
      mime: 'video/mp4',
      useSam: false,
      isVideo: true,
    };
    Alert.alert('Use SAM masks?', 'Choose segmentation mode for this video:', [
      { text: 'No (YOLO only)', onPress: () => askLocation(false) },
      { text: 'Yes (YOLO + SAM)',  onPress: () => askLocation(true) },
    ]);
  };

  async function askLocation(useSam: boolean) {
    stashRef.current!.useSam = useSam;
    Alert.alert(
      'At issue location now?',
      'If yes, we’ll grab your GPS; otherwise enter address manually.',
      [
        {
          text: 'Yes',
          onPress: async () => {
            const { status } = await Location.requestForegroundPermissionsAsync();
            if (status === 'granted') {
              const loc = await Location.getCurrentPositionAsync({});
              pinRef.current = { latitude: loc.coords.latitude, longitude: loc.coords.longitude };
              setMapAdjustVisible(true);       
            } else {
              Alert.alert('Location permission denied');
            }
          },
        },
        { text: 'No', onPress: () => setAddrPickerVisible(true) },
      ],
    );
  }

  async function performUpload(opts: { latitude?: number; longitude?: number; address?: string }) {
    setUploading(true);
    try {
      const { uri, name, mime, useSam, isVideo } = stashRef.current!;
      const form = new FormData();
      form.append('file', { uri, name, type: mime } as any);
      const endpoint = isVideo
        ? `/infer/video?use_sam=${useSam}`
        : `/infer/image?use_sam=${useSam}`;
      if (opts.latitude != null && opts.longitude != null) {
        form.append('latitude',  opts.latitude.toString());
        form.append('longitude', opts.longitude.toString());
      } else if (opts.address) {
        form.append('address', opts.address);
      }
      const { data } = await client.post(endpoint, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      navigation.navigate('Detail', { media: data, showInfo: false });
    } catch (e: any) {
      Alert.alert('Upload failed', e.message);
    } finally {
      setUploading(false);
      setAddrPickerVisible(false);
      setMapAdjustVisible(false);
    }
  }

  return (
    <>
      <FullScreenLoader visible={uploading} />
      {(
        <AddressPickerModal
        visible={addrPickerVisible}
          onCancel={() => setAddrPickerVisible(false)}
          onConfirm={async (coords /* LatLng */, formatted) => {
            setAddrPickerVisible(false);

            // 📌 user may have dragged pin after picking → always re-geocode
            const finalAddress = await reverseGeocode(coords);

            performUpload({
              latitude:  coords.latitude,
              longitude: coords.longitude,
              address:   finalAddress || formatted,   // fallback to autocomplete text
            });
          }}
        />
      )}


        {/* ───────────────────────── GPS pin adjuster ─────────────────────── */}
        {pinRef.current && (
        <MapAdjustModal
          visible={mapAdjustVisible}
          initial={pinRef.current}
          onCancel={() => setMapAdjustVisible(false)}
          onConfirm={async coords => {
            setMapAdjustVisible(false);

            const finalAddress = await reverseGeocode(coords);

            performUpload({
              latitude:  coords.latitude,
              longitude: coords.longitude,
              address:   finalAddress,              // always fresh
            });
          }}
        />
      )}

      <SafeAreaView style={{ flex: 1 }}>
        <Box flex={1} bg="background" p="l">
          <Box
            bg="card"
            p="l"
            borderRadius="l"
            style={{
              shadowColor: '#000',
              shadowOpacity: 0.06,
              shadowRadius: 6,
              shadowOffset: { width: 0, height: 3 },
              elevation: 2,
            }}
          >
            {/* Logout */}
            <Box alignItems="flex-end" mb="m">
              <StyledButton
                title="Logout"
                onPress={handleLogout}
                variant="danger"
                style={{ width: 140 }}
              />
            </Box>

            {/* Title */}
            <Text variant="title" textAlign="center" color="text" mb="l">
              Upload&nbsp;Media
            </Text>

            {/* Pick Photo / Video row */}
            <Box flexDirection="row" columnGap="s" mb="m">
              <StyledButton
                title={<><Feather name="camera" size={16} /> Pick Photo</>}
                onPress={pickAndUploadImage}
                flex={1}
              />
              <StyledButton
                title={<><Feather name="video" size={16} /> Pick Video</>}
                onPress={pickAndUploadVideo}
                variant="secondary"
                flex={1}
              />
            </Box>

            {/* My Uploads needs a top margin */}
            <StyledButton
              title="My Uploads"
              onPress={() => navigation.navigate('Gallery')}
              style={{ marginTop: sp.m }}
            />

            <StyledButton
              title="Test Refresh"
              onPress={() => navigation.navigate('TestRefresh')}
            />
          </Box>
        </Box>
      </SafeAreaView>
    </>
  );
}

