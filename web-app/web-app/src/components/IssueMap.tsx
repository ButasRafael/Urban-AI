import {
  GoogleMap,
  Marker,
  InfoWindow,
  useJsApiLoader,
} from '@react-google-maps/api';
import { useEffect, useState, useMemo } from 'react';
import type { Problem } from '../api/problems';
import IssueModal from './IssueModal';

type LatLng = { lat: number; lng: number };
type Geocoded = Problem & { latlng: LatLng };

const GOOGLE = import.meta.env.VITE_GOOGLE_MAPS_API_KEY as string;

function useExactCoords(problems: Problem[]): Geocoded[] {
  return problems
    .filter(p => p.latitude != null && p.longitude != null)
    .map(p => ({
      ...p,
      latlng: { lat: p.latitude!, lng: p.longitude! }
    }));
}

export default function IssueMap({ problems }: { problems: Problem[] }) {
  const geocoded = useExactCoords(problems);
  const [active, setActive] = useState<Geocoded | null>(null);

  const { isLoaded, loadError } = useJsApiLoader({
    googleMapsApiKey: GOOGLE,
    id: 'urban-ai-map',
  });

  /* fallback: Cluj; dar centrat pe ptrimul issue */
  const center = useMemo<LatLng>(
    () =>
      geocoded[0]?.latlng ?? { lat: 46.7712, lng: 23.6236 },
    [geocoded],
  );

  if (loadError) return <p>Map failed to load</p>;
  if (!isLoaded) return <p>Loading Google Maps…</p>;

  return (
    <>
      <GoogleMap
        mapContainerStyle={{ width: '100%', height: '70vh' }}
        center={center}
        zoom={13}
        options={{
          streetViewControl: false,
          mapTypeControl: false,
          fullscreenControl: false,
        }}
      >
        {geocoded.map((p) => (
          <Marker
            key={p.media_id}
            position={p.latlng}
            onClick={() => setActive(p)}
          />
        ))}

        {active && (
          <InfoWindow
            position={active.latlng}
            onCloseClick={() => setActive(null)}
          >
            <div>
              <strong>
                {active.predicted_classes.join(', ') || '(unknown)'}
              </strong>
              <br />
              {active.address}
            </div>
          </InfoWindow>
        )}
      </GoogleMap>

      <IssueModal problem={active} onClose={() => setActive(null)} />
    </>
  );
}
