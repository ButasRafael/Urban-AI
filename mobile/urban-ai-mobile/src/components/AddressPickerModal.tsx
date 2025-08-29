import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Modal,
  View,
  StyleSheet,
  SafeAreaView,
  Alert,
  Platform,
  StatusBar
} from 'react-native';
import {
  GooglePlacesAutocomplete,
  GooglePlacesAutocompleteRef,
} from 'react-native-google-places-autocomplete';
import MapView, {
  Marker,
  PROVIDER_GOOGLE,
  LatLng,
} from 'react-native-maps';
import { Feather } from '@expo/vector-icons';
import { GOOGLE_API_KEY } from '../config';
import StyledButton from './StyledButton';
import { useTheme } from '@shopify/restyle';
import { Box, Text } from './restylePrimitives';
import type { Theme } from '../theme';
import { darkMapStyle, lightMapStyle } from '../theme/mapStyles';

type Props = {
  visible: boolean;
  onCancel: () => void;
  onConfirm: (coords: LatLng, formattedAddress: string) => void;
  disableUserLocation?: boolean;
};

export default function AddressPickerModal({ visible, onCancel, onConfirm, disableUserLocation = false }: Props) {
  const [coords, setCoords] = useState<LatLng | null>(null);
  const [address, setAddress] = useState('');
  const placesRef = useRef<GooglePlacesAutocompleteRef>(null);
  const mapRef = useRef<MapView>(null);
  const theme = useTheme<Theme>();
  const isDark = (theme as any).mode === 'dark';
  useEffect(() => {
    if (coords && mapRef.current) {
      mapRef.current.animateToRegion(
        {
          ...coords,
          latitudeDelta: 0.01,
          longitudeDelta: 0.01,
        },
        250
      );
    }
  }, [coords]);

  const handlePoiClick = useCallback(
  async (placeId: string, coordinate: LatLng) => {
    try {
      setCoords(coordinate)

      const res = await fetch(
        `https://maps.googleapis.com/maps/api/place/details/json?` +
        `place_id=${placeId}&key=${GOOGLE_API_KEY}&fields=formatted_address`
      )
      const json = await res.json()
      setAddress(json.result.formatted_address ?? '')
    } catch (err) {
      Alert.alert('Place Details error', (err as any).message)
    }
  },
  []
)

  const handlePlaceSelect = useCallback((data: any, details: any | null) => {
    const location = details?.geometry?.location || data?.geometry?.location;
    if (!location) return;

    const { lat, lng } = location;
    setCoords({ latitude: lat, longitude: lng });
    setAddress(details?.formatted_address ?? data?.description ?? '');
  }, []);

  const handleFail = useCallback((err: unknown) => {
    Alert.alert('Places API error', JSON.stringify(err), [{ text: 'OK' }]);
  }, []);

  const handleNotFound = useCallback(() => {
    Alert.alert('No results', 'No places match that search.', [{ text: 'OK' }]);
  }, []);

  return (
    <Modal visible={visible} animationType="slide" onRequestClose={onCancel}>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} backgroundColor={theme.colors.background} />
      
      <Box flex={1} backgroundColor="background">
        {/* Header */}
        <SafeAreaView>
          <Box 
            flexDirection="row" 
            alignItems="center" 
            justifyContent="space-between"
            paddingHorizontal="l"
            paddingVertical="m"
            backgroundColor="card"
            borderBottomWidth={1}
            borderBottomColor="border"
            style={theme.shadows.sm}
          >
            <Text variant="title" color="text">Selectează adresa</Text>
            <StyledButton
              title=""
              onPress={onCancel}
              variant="ghost"
              style={{ padding: theme.spacing.s, minWidth: 40 }}
              leftIcon={<Feather name="x" size={20} color={theme.colors.text} />}
            />
          </Box>
        </SafeAreaView>

        {/* Map Container */}
        <Box flex={1} position="relative">
          {coords && (
            <MapView
              ref={mapRef}
              provider={PROVIDER_GOOGLE}
              style={StyleSheet.absoluteFillObject}
              customMapStyle={isDark ? darkMapStyle : lightMapStyle}
              initialRegion={{
                ...coords,
                latitudeDelta: 0.01,
                longitudeDelta: 0.01,
              }}
              showsUserLocation={!disableUserLocation}
              showsMyLocationButton={!disableUserLocation}
              showsCompass
              zoomControlEnabled={Platform.OS === 'android'}
              zoomTapEnabled
              onPoiClick={({ nativeEvent: { placeId, coordinate } }) =>
                handlePoiClick(placeId, coordinate)
              }
            >
              <Marker
                draggable
                pinColor={theme.colors.primary500} 
                coordinate={coords}
                onDragEnd={e => setCoords(e.nativeEvent.coordinate)}
              />
            </MapView>
          )}

          {/* Enhanced Search Bar */}
          <Box
            position="absolute"
            top={theme.spacing.l}
            left={theme.spacing.l}
            right={theme.spacing.l}
            zIndex={100}
            backgroundColor="card"
            borderRadius="m"
            style={theme.shadows.lg}
            overflow="hidden"
          >
            <Box
              flexDirection="row"
              alignItems="center"
              paddingHorizontal="m"
              paddingVertical="xs"
              backgroundColor="surface50"
              borderBottomWidth={1}
              borderBottomColor="border"
            >
              <Feather 
                name="map-pin" 
                size={16} 
                color={theme.colors.primary500} 
                style={{ marginRight: theme.spacing.s }} 
              />
              <Text variant="label" color="muted">Caută o adresă</Text>
            </Box>
            
            <GooglePlacesAutocomplete
              predefinedPlaces={[]}
              ref={placesRef}
              placeholder="Introduceți strada, numărul..."
              fetchDetails
              minLength={2}
              debounce={300}
              autoFillOnNotFound
              keyboardShouldPersistTaps="always"
              enablePoweredByContainer={false}
              timeout={20000} // 20 seconds timeout for Android
              query={{
                key: GOOGLE_API_KEY,
                language: 'ro',
                components: 'country:ro',
                types: 'address',
              }}
              onPress={handlePlaceSelect}
              onFail={handleFail}
              onNotFound={handleNotFound}
              textInputProps={{
                onFocus: () => {},
                autoCapitalize: 'none',
                autoCorrect: false,
                returnKeyType: 'search',
                placeholderTextColor: theme.colors.muted,
                style: { 
                  color: theme.colors.text,
                  fontSize: 16,
                  fontFamily: 'Inter_400Regular'
                },
              }}
              styles={{
                container: {
                  flex: 0,
                },
                textInputContainer: {
                  backgroundColor: 'transparent',
                  borderTopWidth: 0,
                  borderBottomWidth: 0,
                  paddingHorizontal: theme.spacing.m,
                  paddingVertical: theme.spacing.xs,
                },
                textInput: {
                  height: 48,
                  fontSize: 16,
                  backgroundColor: 'transparent',
                  paddingHorizontal: 0,
                  marginLeft: 0,
                  marginRight: 0,
                },
                listView: {
                  backgroundColor: theme.colors.card,
                  borderTopWidth: 1,
                  borderTopColor: theme.colors.border,
                  maxHeight: 200,
                },
                row: {
                  backgroundColor: 'transparent',
                  paddingHorizontal: theme.spacing.m,
                  paddingVertical: theme.spacing.m,
                  borderBottomWidth: 1,
                  borderBottomColor: theme.colors.surface200,
                },
                description: {
                  fontSize: 15,
                  color: theme.colors.text,
                  fontFamily: 'Inter_400Regular',
                },
                poweredContainer: {
                  display: 'none',
                },
              }}
            />
          </Box>

          {/* Location Info Card */}
          {coords && address && (
            <Box
              position="absolute"
              bottom={100}
              left={theme.spacing.l}
              right={theme.spacing.l}
              backgroundColor="card"
              borderRadius="m"
              padding="m"
              style={theme.shadows.md}
            >
              <Box flexDirection="row" alignItems="flex-start">
                <Box
                  width={32}
                  height={32}
                  borderRadius="s"
                  backgroundColor="primary100"
                  alignItems="center"
                  justifyContent="center"
                  marginRight="m"
                >
                  <Feather name="map-pin" size={16} color={theme.colors.primary500} />
                </Box>
                <Box flex={1}>
                  <Text variant="label" color="primary500" marginBottom="xs">
                    Adresa selectată
                  </Text>
                  <Text variant="body" color="text" numberOfLines={2}>
                    {address}
                  </Text>
                </Box>
              </Box>
            </Box>
          )}
        </Box>

        {/* Bottom Actions */}
        <SafeAreaView>
          <Box
            backgroundColor="card"
            paddingHorizontal="l"
            paddingVertical="m"
            borderTopWidth={1}
            borderTopColor="border"
            style={theme.shadows.sm}
          >
            <Box flexDirection="row" justifyContent="space-between">
              <Box flex={1} marginRight="m">
                <StyledButton 
                  title="Anulează" 
                  onPress={onCancel} 
                  variant="ghost" 
                />
              </Box>
              <Box flex={1} marginLeft="m">
                <StyledButton
                  title="Confirmă"
                  onPress={() => coords && onConfirm(coords, address)}
                  disabled={!coords}
                  variant="primary"
                />
              </Box>
            </Box>
          </Box>
        </SafeAreaView>
      </Box>
    </Modal>
  );
}

