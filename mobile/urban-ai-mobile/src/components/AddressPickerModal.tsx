import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Modal,
  View,
  StyleSheet,
  SafeAreaView,
  Alert,
  Platform
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
import { GOOGLE_API_KEY } from '../config';
import StyledButton from './StyledButton';
import { useTheme } from '@shopify/restyle'
import type { Theme } from '../theme'

type Props = {
  visible: boolean;
  onCancel: () => void;
  onConfirm: (coords: LatLng, formattedAddress: string) => void;
};

export default function AddressPickerModal({ visible, onCancel, onConfirm }: Props) {
  const [coords, setCoords] = useState<LatLng | null>(null);
  const [address, setAddress] = useState('');
  const placesRef = useRef<GooglePlacesAutocompleteRef>(null);
  const mapRef = useRef<MapView>(null);
  const { colors } = useTheme<Theme>()
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
      <View style={{ flex: 1 }}>
        {coords && (
          <MapView
            ref={mapRef}
            provider={PROVIDER_GOOGLE}
            style={StyleSheet.absoluteFillObject}
            initialRegion={{
              ...coords,
              latitudeDelta: 0.01,
              longitudeDelta: 0.01,
            }}
            showsUserLocation
            showsMyLocationButton
            showsCompass
            zoomControlEnabled={Platform.OS === 'android'}
            zoomTapEnabled
             onPoiClick={({ nativeEvent: { placeId, coordinate } }) =>
    handlePoiClick(placeId, coordinate)
  }

          >
            <Marker
              draggable
              pinColor={colors.primary500} 
              coordinate={coords}
              onDragEnd={e => setCoords(e.nativeEvent.coordinate)}
            />
          </MapView>
        )}

        {/* Autocomplete overlay */}
        <SafeAreaView style={styles.autocompleteContainer}>
          <GooglePlacesAutocomplete
            predefinedPlaces={[]}
            ref={placesRef}
            placeholder="Strada, număr…"
            fetchDetails
            minLength={2}
            debounce={300}
            autoFillOnNotFound
            keyboardShouldPersistTaps="always"
            enablePoweredByContainer
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
            }}
            styles={{
              container: {
                position: 'absolute',
                top: 16,
                left: 16,
                right: 16,
                zIndex: 100,
                elevation: 100,
              },
              textInputContainer: {
                backgroundColor: 'white',
                borderTopWidth: 0,
                borderBottomWidth: 0,
                paddingHorizontal: 0,
              },
              textInput: {
                height: 44,
                fontSize: 16,
              },
              listView: {
                backgroundColor: 'white',
                zIndex: 100,
                elevation: 100,
                marginTop: 4,
              },
            }}
          />
        </SafeAreaView>

        <View style={styles.btnRow}>
          <StyledButton title="Cancel" onPress={onCancel} variant="ghost" />
          <StyledButton
            title="OK"
            onPress={() => coords && onConfirm(coords, address)}
            disabled={!coords}
          />
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  autocompleteContainer: {
    position: 'absolute',
    left: 16,
    right: 16,
    top: 44,
    zIndex: 100,
    elevation: 100,
  },
  btnRow: {
    position: 'absolute',
    bottom: 32,
    width: '100%',
    flexDirection: 'row',
    justifyContent: 'space-evenly',
    zIndex: 1,
  },
});
