import React, { useCallback, useEffect, useLayoutEffect, useMemo, useState } from 'react';
import { Image, Platform, Pressable } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useTheme } from '@shopify/restyle';
import { Feather } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import * as Location from 'expo-location';
import AsyncStorage from '@react-native-async-storage/async-storage';

import { RootStackParamList } from '../navigation/types';
import { Box, Text } from '../components/restylePrimitives';
import Screen from '../components/ui/Screen';
import HeroHeader from '../components/ui/HeroHeader';
import Card from '../components/ui/Card';
import Badge from '../components/ui/Badge';
import StyledButton from '../components/StyledButton';
import InlineNotice from '../components/ui/InlineNotice';
import FullScreenLoader from '../components/FullScreenLoader';
import AddressPickerModal from '../components/AddressPickerModal';
import MapAdjustModal from '../components/MapAdjustModal';
import { notify } from '../components/ui/Toast';

import type { Theme } from '../theme';
import type { LatLng } from 'react-native-maps';
import client from '../api/client';
import { logout } from '../api/auth';
import { GOOGLE_API_KEY } from '../config';

type Props = NativeStackScreenProps<RootStackParamList, 'Home'>;

type StashedMedia = {
  uri: string;
  name: string;
  mime: string;
  isVideo: boolean;
};

const REMEMBER_SAM_KEY = 'ui:useSamDefault';

export default function HomeScreen({ navigation }: Props) {
  const theme = useTheme<Theme>();
  const sp = theme.spacing;
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [addrPickerVisible, setAddrPickerVisible] = useState(false);
  const [mapAdjustVisible, setMapAdjustVisible] = useState(false);

  const [useSam, setUseSam] = useState<boolean>(false);
  const [selected, setSelected] = useState<StashedMedia | null>(null);
  const [pin, setPin] = useState<LatLng | null>(null);

  const hasMedia = !!selected;
  const hasLocation = !!pin;
  const canUpload = hasMedia && hasLocation;

  useLayoutEffect(() => {
    navigation.setOptions({
      title: 'Upload',
      headerLeft: () => (
        <Pressable
          onPress={handleLogout}
          hitSlop={10}
          style={{ paddingHorizontal: 8, paddingVertical: 6 }}
          accessibilityRole="button"
          accessibilityLabel="Logout"
        >
          <Feather name="log-out" size={20} color={theme.colors.error} />
        </Pressable>
      ),
    });
  }, [navigation, theme.colors.error]);

  useEffect(() => {
    (async () => {
      const saved = await AsyncStorage.getItem(REMEMBER_SAM_KEY);
      if (saved != null) setUseSam(saved === '1');
    })();
  }, []);
  useEffect(() => {
    AsyncStorage.setItem(REMEMBER_SAM_KEY, useSam ? '1' : '0').catch(() => {});
  }, [useSam]);

  const friendlyFileName = (name: string) => (name?.length > 36 ? name.slice(0, 33) + '…' : name);
  const getMimeFromUri = (uri: string, fallback: string) => {
    const lower = uri.split('?')[0].toLowerCase();
    if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) return 'image/jpeg';
    if (lower.endsWith('.png')) return 'image/png';
    if (lower.endsWith('.heic')) return Platform.OS === 'ios' ? 'image/heic' : 'image/jpeg';
    if (lower.endsWith('.webp')) return 'image/webp';
    if (lower.endsWith('.mp4')) return 'video/mp4';
    if (lower.endsWith('.mov')) return 'video/quicktime';
    return fallback;
  };

  async function reverseGeocode({ latitude, longitude }: LatLng) {
    try {
      const url =
        `https://maps.googleapis.com/maps/api/geocode/json?latlng=${latitude},${longitude}` +
        `&language=ro&key=${GOOGLE_API_KEY}`;
      const res = await fetch(url);
      const json = await res.json();
      return json.results?.[0]?.formatted_address ?? '';
    } catch {
      return '';
    }
  }

  const resetFlow = () => {
    setAddrPickerVisible(false);
    setMapAdjustVisible(false);
  };

  async function handleLogout() {
    try {
      await logout();
      navigation.replace('Login');
    } catch (e: any) {
      notify.error('Logout eșuat', e.message);
    }
  }

  const pickImage = useCallback(async () => {
    setError(null);
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.9,
      selectionLimit: 1,
    });
    if (res.canceled) return;
    const asset = res.assets[0];
    setSelected({
      uri: asset.uri,
      name: asset.fileName || asset.uri.split('/').pop() || 'image.jpg',
      mime: getMimeFromUri(asset.uri, asset.mimeType || 'image/jpeg'),
      isVideo: false,
    });
    notify.info('Imagine selectată');
  }, []);

  const pickVideo = useCallback(async () => {
    setError(null);
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
      selectionLimit: 1,
    });
    if (res.canceled) return;
    const asset = res.assets[0];
    setSelected({
      uri: asset.uri,
      name: asset.fileName || asset.uri.split('/').pop() || 'video.mp4',
      mime: getMimeFromUri(asset.uri, asset.mimeType || 'video/mp4'),
      isVideo: true,
    });
    notify.info('Video selectat');
  }, []);

  const clearSelected = useCallback(() => {
    setSelected(null);
    notify.info('Selecție ștearsă');
  }, []);

  const useCurrentLocation = useCallback(async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        notify.error('Acces la locație refuzat', 'Activează permisiunea pentru a continua.');
        return;
      }
      const loc = await Location.getCurrentPositionAsync({});
      setPin({ latitude: loc.coords.latitude, longitude: loc.coords.longitude });
      setMapAdjustVisible(true); // fine-tune on map
    } catch (e: any) {
      notify.error('Locație indisponibilă', e.message);
    }
  }, []);

  const onAddressPicked = useCallback(async (coords: LatLng, formatted: string) => {
    setAddrPickerVisible(false);
    setPin(coords);
    const final = await reverseGeocode(coords);
    if (final && final !== formatted) notify.info(final);
    setMapAdjustVisible(true);
  }, []);

  const onMapConfirm = useCallback(async (coords: LatLng) => {
    setMapAdjustVisible(false);
    setPin(coords);
    const finalAddress = await reverseGeocode(coords);
    if (finalAddress) notify.success('Locație setată', finalAddress);
  }, []);

  const performUpload = useCallback(async () => {
    if (!selected || !pin) return;
    setUploading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append('file', { uri: selected.uri, name: selected.name, type: selected.mime } as any);
      form.append('latitude', String(pin.latitude));
      form.append('longitude', String(pin.longitude));

      const endpoint = selected.isVideo
        ? `/infer/video?use_sam=${useSam}`
        : `/infer/image?use_sam=${useSam}`;

      const { data } = await client.post(endpoint, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      notify.success('Încărcare reușită');
      setSelected(null);
      navigation.navigate('Detail', { media: data, showInfo: false });
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e?.message || 'Încărcare eșuată';
      setError(msg);
      notify.error('Eroare', msg);
    } finally {
      setUploading(false);
      resetFlow();
    }
  }, [selected, pin, useSam, navigation]);

  const SelectedPreview = () =>
    selected ? (
      <Box marginTop="m">
        <Card variant="outlined" padding="m">
        <Box flexDirection="row" alignItems="center">
          <Box
            width={56}
            height={56}
            borderRadius="s"
            overflow="hidden"
            mr="m"
            alignItems="center"
            justifyContent="center"
            bg="card"
          >
            {selected.isVideo ? (
              <Feather name="video" size={22} color={theme.colors.primary500} />
            ) : (
              <Image source={{ uri: selected.uri }} style={{ width: 56, height: 56 }} resizeMode="cover" />
            )}
          </Box>
          <Box flex={1}>
            <Text variant="label" color="text" numberOfLines={1}>
              {friendlyFileName(selected.name)}
            </Text>
            <Text variant="label" color="muted" numberOfLines={1}>
              {selected.isVideo ? 'Video' : 'Imagine'}
            </Text>
          </Box>
          <StyledButton
            title={<Feather name="x" size={16} />}
            variant="ghost"
            onPress={clearSelected}
            size="sm"
          />
        </Box>
        </Card>
      </Box>
    ) : null;

  return (
    <>
      <FullScreenLoader visible={uploading} />

      <AddressPickerModal
        visible={addrPickerVisible}
        onCancel={() => setAddrPickerVisible(false)}
        onConfirm={onAddressPicked}
      />

      {pin && (
        <MapAdjustModal
          visible={mapAdjustVisible}
          initial={pin}
          onCancel={() => setMapAdjustVisible(false)}
          onConfirm={onMapConfirm}
        />
      )}

      <Screen
        scroll
        padded={false}
        edges={['left', 'right']}
        center
        maxWidth={640}
        dismissKeyboardOnTap
        statusBarStyle="light-content"
        translucentStatusBar
      >
        {/* Lighter than auth: smaller hero, no blobs */}
        <HeroHeader
          title="Trimite sesizare"
          subtitle="Încarcă o imagine sau video pentru analizare AI"
          height={200}
          blobs={true}
          rightSlot={
            <Box
              width={56}
              height={56}
              borderRadius="l"
              style={{
                backgroundColor: 'rgba(255,255,255,0.15)',
                borderWidth: 1,
                borderColor: 'rgba(255,255,255,0.25)',
                ...theme.shadows.sm,
              }}
              alignItems="center"
              justifyContent="center"
            >
              <Feather name="upload-cloud" size={24} color="#fff" />
            </Box>
          }
        />

        <Box paddingHorizontal="l" style={{ marginTop: -32 }}>
          <Card variant="elevated" padding="l">
            {error && (
              <InlineNotice
                variant="error"
                title="Eroare"
                message={error}
                compact
                style={{ marginBottom: 12 }}
              />
            )}

            {/* Step 1: pick file */}
            <Box flexDirection="row" alignItems="center" marginBottom="s">
              <Box
                width={24} height={24} borderRadius="m"
                backgroundColor="primary500" marginRight="s"
                alignItems="center" justifyContent="center"
                style={theme.shadows.sm}
              >
                <Text variant="caption" style={{ color: 'white', fontWeight: '800' }}>1</Text>
              </Box>
              <Text variant="subtitle" color="text">
                Alege fișierul
              </Text>
            </Box>
            <Box flexDirection="row" columnGap="s">
              <StyledButton
                title={
                  <>
                    <Feather name="image" size={16} /> Galerie foto
                  </>
                }
                onPress={pickImage}
                flex={1}
                gradient
              />
              <StyledButton
                title={
                  <>
                    <Feather name="video" size={16} /> Galerie video
                  </>
                }
                onPress={pickVideo}
                variant="secondary"
                flex={1}
              />
            </Box>

            <SelectedPreview />

            {/* Step 2: segmentation */}
            <Box mt="l">
              <Box flexDirection="row" alignItems="center" marginBottom="s">
                <Box
                  width={24} height={24} borderRadius="m"
                  backgroundColor="primary500" marginRight="s"
                  alignItems="center" justifyContent="center"
                  style={theme.shadows.sm}
                >
                  <Text variant="caption" style={{ color: 'white', fontWeight: '800' }}>2</Text>
                </Box>
                <Text variant="subtitle" color="text">
                  Opțiuni AI
                </Text>
              </Box>
              <Card variant="outlined" padding="m">
                <Box flexDirection="row" alignItems="center" justifyContent="space-between">
                  <Box flex={1} marginRight="m">
                    <Box flexDirection="row" alignItems="center" marginBottom="xs">
                      <Text variant="label" color="text" style={{ fontWeight: '600' }}>
                        Segmentare precisă SAM2
                      </Text>
                      <Badge variant="primary" size="sm" style={{ marginLeft: 8 }}>
                        AI
                      </Badge>
                    </Box>
                    <Text variant="caption" color="muted">
                      Creează măști detaliate ale obiectelor pentru analiză precisă
                    </Text>
                  </Box>
                  <StyledButton
                    title={useSam ? 'Activ' : 'Inactiv'}
                    variant={useSam ? 'primary' : 'ghost'}
                    onPress={() => setUseSam((s) => !s)}
                    size="sm"
                    accessibilityLabel={useSam ? 'Măști precise activate' : 'Măști precise dezactivate'}
                  />
                </Box>
              </Card>
            </Box>

            {/* Step 3: location */}
            <Box mt="l">
              <Box flexDirection="row" alignItems="center" marginBottom="s">
                <Box
                  width={24} height={24} borderRadius="m"
                  backgroundColor="primary500" marginRight="s"
                  alignItems="center" justifyContent="center"
                  style={theme.shadows.sm}
                >
                  <Text variant="caption" style={{ color: 'white', fontWeight: '800' }}>3</Text>
                </Box>
                <Text variant="subtitle" color="text">
                  Locație
                </Text>
              </Box>
              <Box flexDirection="row" columnGap="s">
                <StyledButton
                  title={
                    <>
                      <Feather name="navigation" size={16} /> Locația mea
                    </>
                  }
                  onPress={useCurrentLocation}
                  flex={1}
                  variant="tonal"
                />
                <StyledButton
                  title={
                    <>
                      <Feather name="map-pin" size={16} /> Introdu adresă
                    </>
                  }
                  onPress={() => setAddrPickerVisible(true)}
                  flex={1}
                  variant="ghost"
                />
              </Box>
              {hasLocation ? (
                <Box mt="s">
                  <InlineNotice
                    variant="success"
                    title="Locație selectată"
                    message="Poți ajusta pinul înainte de încărcare."
                    compact
                  />
                </Box>
              ) : null}
            </Box>

            {/* Primary CTA */}
            <Box mt="xl">
              <StyledButton
                title={
                  <>
                    <Feather name="zap" size={18} /> Analizează cu AI
                  </>
                }
                onPress={performUpload}
                disabled={!canUpload}
                gradient
                fullWidth
                size="lg"
                loading={uploading}
                loadingText="Se procesează..."
                showSpinnerOnly={false}
              />
              
              {canUpload && (
                <Box alignItems="center" marginTop="s">
                  <Badge variant="success" size="sm" dot>
                    Gata pentru analiză
                  </Badge>
                </Box>
              )}
            </Box>
          </Card>
          {/* Quick Actions */}
          <Box mt="l">
            <Card variant="outlined" padding="m">
              <Box flexDirection="row" columnGap="s" alignItems="center">
                <StyledButton
                  title={
                    <>
                      <Feather name="image" size={16} /> Galeria mea
                    </>
                  }
                  onPress={() => navigation.navigate('Gallery')}
                  flex={1}
                  variant="tonal"
                  size="sm"
                />
                <StyledButton
                  title={
                    <>
                      <Feather name="activity" size={16} /> Test
                    </>
                  }
                  onPress={() => navigation.navigate('TestRefresh')}
                  flex={1}
                  variant="ghost"
                  size="sm"
                />
              </Box>
            </Card>
          </Box>

        </Box>

        <Box height={sp.l} />
      </Screen>
    </>
  );
}
