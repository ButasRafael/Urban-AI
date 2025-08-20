import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Modal, StyleSheet, View, Platform } from 'react-native';
import MapView, { Marker, PROVIDER_GOOGLE, LatLng } from 'react-native-maps';
import StyledButton from './StyledButton';
import { useTheme } from '@shopify/restyle'
import type { Theme } from '../theme'
import { GOOGLE_API_KEY } from '../config';
import { Alert } from 'react-native';

type Props = {
  visible: boolean;
  initial: LatLng;
  onCancel: () => void;
  onConfirm: (coords: LatLng) => void;
};

export default function MapAdjustModal({
  visible,
  initial,
  onCancel,
  onConfirm,
}: Props) {
  const mapRef = useRef<MapView>(null);
  const [coords, setCoords] = useState<LatLng>(initial);
  const [address, setAddress] = useState('');
  const { colors } = useTheme<Theme>()

  useEffect(() => {
    if (mapRef.current) {
      mapRef.current.animateToRegion(
        {
          ...coords,
          latitudeDelta: 0.01,
          longitudeDelta: 0.01,
        },
        300
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


  const handleDragEnd = useCallback(
    (e: { nativeEvent: { coordinate: LatLng } }) => {
      setCoords(e.nativeEvent.coordinate);
    },
    []
  );

  const handleConfirm = useCallback(() => {
    onConfirm(coords);
  }, [coords, onConfirm]);

  return (
    <Modal
      visible={visible}
      animationType="slide"
      onRequestClose={onCancel}
    >
      <View style={styles.container}>
        <MapView
          ref={mapRef}
          provider={PROVIDER_GOOGLE}
          style={styles.map}
          initialRegion={{
            ...initial,
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
          <Marker draggable coordinate={coords} pinColor={colors.primary500}  onDragEnd={handleDragEnd} />
        </MapView>

        <View style={styles.btnRow}>
          <StyledButton title="Cancel" onPress={onCancel} variant="ghost" />
          <StyledButton title="OK" onPress={handleConfirm} />
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  map: {
    ...StyleSheet.absoluteFillObject,
  },
  btnRow: {
    position: 'absolute',
    bottom: 32,
    width: '100%',
    flexDirection: 'row',
    justifyContent: 'space-evenly',
    paddingHorizontal: 16,
  },
});