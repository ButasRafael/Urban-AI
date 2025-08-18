import {
  GoogleMap,
  Marker,
  InfoWindow,
  useJsApiLoader,
  MarkerClustererF,
} from '@react-google-maps/api';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { Problem } from '../api/problems';
import IssueModal from './IssueModal';
import '../styles/map.css';

type LatLng = { lat: number; lng: number };
type Geocoded = Problem & { latlng: LatLng };

const GOOGLE = import.meta.env.VITE_GOOGLE_MAPS_API_KEY as string;

/* ---------------- Helpers ---------------- */

function getTheme(): 'light' | 'dark' {
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
}

function useExactCoords(problems: Problem[]): Geocoded[] {
  return problems
    .filter((p) => p.latitude != null && p.longitude != null)
    .map((p) => ({ ...p, latlng: { lat: p.latitude!, lng: p.longitude! } }));
}

function primaryClass(p: Problem) {
  const c = (p as any)?.predicted_classes?.[0];
  return c || 'Issue';
}

function hashHue(label: string): number {
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) % 360;
  return h;
}

function colorFor(label: string): string {
  return `hsl(${hashHue(label)} 85% 46%)`;
}

/** Themed pin (cached). Active pins get a subtle glow + larger size */
const pinCache = new Map<string, google.maps.Icon>();
function pinIcon(theme: 'light' | 'dark', color: string, active = false): google.maps.Icon {
  const key = `${theme}-${color}-${active ? 'a' : 'n'}`;
  if (pinCache.has(key)) return pinCache.get(key)!;

  const stroke = theme === 'dark' ? '#0b1220' : '#ffffff';
  const halo = theme === 'dark' ? 'rgba(255,255,255,.14)' : 'rgba(0,0,0,.12)';
  const size = active ? 36 : 30;
  const height = active ? 46 : 38;
  const anchorY = active ? 46 : 38;

  const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${height}" viewBox="0 0 30 38">
  <defs>
    <filter id="g" x="-50%" y="-50%" width="200%" height="200%">
      <feDropShadow dx="0" dy="1" stdDeviation="1.4" flood-opacity=".28"/>
    </filter>
    ${active ? `<filter id="halo"><feGaussianBlur in="SourceGraphic" stdDeviation="2"/></filter>` : ''}
  </defs>
  <path filter="url(#g)" d="M15 1C8.373 1 3 6.318 3 12.9 3 24.2 15 37 15 37s12-12.8 12-24.1C27 6.318 21.627 1 15 1Z" fill="${color}" stroke="${stroke}" stroke-width="1"/>
  <circle cx="15" cy="13" r="5.6" fill="white" fill-opacity="${theme === 'dark' ? '.9' : '.95'}"/>
  ${active ? `<circle cx="15" cy="13" r="10" fill="${halo}" filter="url(#halo)"/>` : ''}
</svg>`.trim();

  const icon: google.maps.Icon = {
    url: 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(svg),
    scaledSize: new google.maps.Size(size, height),
    anchor: new google.maps.Point(size / 2, anchorY),
  };
  pinCache.set(key, icon);
  return icon;
}

/** Themed cluster bubbles */
function clusterSVG(theme: 'light' | 'dark', size: number) {
  const bg = theme === 'dark' ? '#0e1628' : '#e8f1ff';
  const border = theme === 'dark' ? '#2a3a4a' : '#bcd7ff';
  return `
<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
  <defs><filter id="s" x="-50%" y="-50%" width="200%" height="200%">
    <feDropShadow dx="0" dy="1" stdDeviation="1.2" flood-opacity=".25"/>
  </filter></defs>
  <circle cx="${size / 2}" cy="${size / 2}" r="${Math.floor(size / 2) - 2}" fill="${bg}" stroke="${border}" stroke-width="2" filter="url(#s)"/>
</svg>`.trim();
}

function clusterStyles(theme: 'light' | 'dark') {
  return [32, 40, 48].map((px) => ({
    url: 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(clusterSVG(theme, px)),
    width: px,
    height: px,
    textColor: theme === 'dark' ? '#dbeafe' : '#0b3b6f',
    textSize: 14,
  }));
}

/* Map styles */
const lightStyle: google.maps.MapTypeStyle[] = [
  { featureType: 'poi', stylers: [{ visibility: 'off' }] },
  { featureType: 'transit', stylers: [{ visibility: 'off' }] },
  { featureType: 'road', elementType: 'geometry', stylers: [{ lightness: 30 }] },
  { elementType: 'labels.icon', stylers: [{ visibility: 'off' }] },
];

const darkStyle: google.maps.MapTypeStyle[] = [
  { elementType: 'geometry', stylers: [{ color: '#0b1220' }] },
  { elementType: 'labels.text.stroke', stylers: [{ color: '#0b1220' }] },
  { elementType: 'labels.text.fill', stylers: [{ color: '#94a3b8' }] },
  { featureType: 'road', elementType: 'geometry', stylers: [{ color: '#1f2c38' }] },
  { featureType: 'poi', stylers: [{ visibility: 'off' }] },
  { featureType: 'transit', stylers: [{ visibility: 'off' }] },
  { featureType: 'water', stylers: [{ color: '#0e1628' }] },
];

/* ---------------- Component ---------------- */

export default function IssueMap({ problems }: { problems: Problem[] }) {
  const geocoded = useExactCoords(problems);

  const [active, setActive] = useState<Geocoded | null>(null);
  const [modal, setModal] = useState<Geocoded | null>(null);
  const [myLoc, setMyLoc] = useState<LatLng | null>(null);

  // NEW: unique primary classes for legend
  const classesList = useMemo(
    () => Array.from(new Set(geocoded.map(primaryClass))).slice(0, 6),
    [geocoded]
  );

  /* Theme: reacts to app toggle + OS changes */
  const [theme, setTheme] = useState<'light' | 'dark'>(getTheme());
  useEffect(() => {
    const onClassChange = () => setTheme(getTheme());
    const obs = new MutationObserver(onClassChange);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });

    const mql = window.matchMedia?.('(prefers-color-scheme: dark)');
    const onMql = () => setTheme(getTheme());
    mql?.addEventListener?.('change', onMql);

    window.addEventListener('themechange', onClassChange);
    return () => {
      obs.disconnect();
      mql?.removeEventListener?.('change', onMql);
      window.removeEventListener('themechange', onClassChange);
    };
  }, []);

  const { isLoaded, loadError } = useJsApiLoader({
    id: 'urban-ai-map',
    googleMapsApiKey: GOOGLE,
  });

  const center = useMemo<LatLng>(
    () => geocoded[0]?.latlng ?? { lat: 46.7712, lng: 23.6236 },
    [geocoded]
  );

  const mapRef = useRef<google.maps.Map | null>(null);

  const fitToMarkers = () => {
    const map = mapRef.current;
    if (!map) return;
    if (geocoded.length === 0) {
      map.setCenter(center);
      map.setZoom(13);
      return;
    }
    const bounds = new google.maps.LatLngBounds();
    geocoded.forEach((g) => bounds.extend(g.latlng));
    if (myLoc) bounds.extend(myLoc);
    map.fitBounds(bounds, 64);
  };

  const locateMe = () => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const pt = { lat: pos.coords.latitude, lng: pos.coords.longitude };
        setMyLoc(pt);
        mapRef.current?.panTo(pt);
        mapRef.current?.setZoom(Math.max(mapRef.current?.getZoom() ?? 14, 15));
      },
      () => {},
      { enableHighAccuracy: true, timeout: 8000 }
    );
  };

  if (loadError) {
    return (
      <div className="map-fallback card">
        <div className="map-fallback__inner">
          <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden>
            <path d="M12 9v4m0 4h.01M10.29 3.86l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.71-3.14l-8-14a2 2 0 0 0-3.42 0Z" fill="none" stroke="currentColor" strokeWidth="2"/>
          </svg>
          <div>
            <strong>Map failed to load</strong>
            <p className="muted">Check your connection or API key, then refresh.</p>
          </div>
        </div>
      </div>
    );
  }

  if (!isLoaded) {
    return (
      <div className="map-skeleton">
        <div className="sk shimmer" />
      </div>
    );
  }

  return (
    <div className="map-shell">
      {/* Overlay UI (controls + legend + status) */}
      <div className="map-ui" role="region" aria-label="Map controls and legend">
        <div className="map-card map-controls" role="toolbar" aria-label="Map controls">
          <button className="control-btn" onClick={fitToMarkers} title="Fit to markers" aria-label="Fit to markers">
            <svg width="16" height="16" viewBox="0 0 24 24"><path d="M3 3h7M3 3v7M21 3h-7M21 3v7M3 21h7M3 21v-7M21 21h-7M21 21v-7" fill="none" stroke="currentColor" strokeWidth="2"/></svg>
          </button>
          <button className="control-btn" onClick={locateMe} title="My location" aria-label="My location">
            <svg width="16" height="16" viewBox="0 0 24 24"><path d="M12 8v8m-4-4h8" stroke="currentColor" strokeWidth="2" fill="none"/><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" fill="none"/></svg>
          </button>
        </div>

        <div className="map-card map-legend" aria-label="Legend">
          <strong>Legend</strong>
          <ul>
            {classesList.map((c) => (
              <li key={c}>
                <span className="legend-dot" style={{ background: colorFor(c) }} />
                <span className="legend-label">{c}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="map-card map-status" aria-live="polite">
          <span className="dot" /> {geocoded.length} issues
        </div>
      </div>

      <GoogleMap
        onLoad={(map) => {
          mapRef.current = map;
          map.setOptions({ styles: theme === 'dark' ? darkStyle : lightStyle });
          requestAnimationFrame(fitToMarkers);
        }}
        onUnmount={() => { mapRef.current = null; }}
        mapContainerClassName="map-el"
        center={center}
        zoom={13}
        options={{
          streetViewControl: false,
          mapTypeControl: false,
          fullscreenControl: false,
          clickableIcons: false,
          gestureHandling: 'greedy',
          disableDefaultUI: true,
        }}
      >
        {/* keep map style in sync with theme */}
        <StyleSync theme={theme} getMap={() => mapRef.current} />

        <MarkerClustererF key={theme} options={{ styles: clusterStyles(theme), minimumClusterSize: 2, gridSize: 56 }}>
          {(clusterer) => (
            <>
              {geocoded.map((p) => {
                const label = primaryClass(p);
                const color = colorFor(label);
                const isActive = active?.media_id === p.media_id;
                return (
                  <Marker
                    key={p.media_id}
                    position={p.latlng}
                    clusterer={clusterer}
                    icon={pinIcon(theme, color, isActive)}
                    zIndex={isActive ? 20 : 10}
                    onClick={() => {
                      setActive(p);
                      // Nudge camera a bit so info window is visible above the marker
                      const m = mapRef.current;
                      if (!m) return;
                      const off = m.getZoom() && (m.getZoom()! >= 15 ? 0.0012 : 0.003) || 0;
                      m.panTo({ lat: p.latlng.lat + off, lng: p.latlng.lng });
                    }}
                  />
                );
              })}
            </>
          )}
        </MarkerClustererF>

        {myLoc && (
          <Marker
            position={myLoc}
            title="You are here"
            icon={pinIcon(theme, '#3b82f6', true)}
            zIndex={30}
          />
        )}

        {active && (
          <InfoWindow
            position={active.latlng}
            onCloseClick={() => setActive(null)}
            options={{ maxWidth: 300 }}
          >
            <div className="info">
              <div className="info__title">
                {primaryClass(active)}
                <span className="pill">{(active as any)?.predicted_classes?.slice(0, 3)?.join(' · ')}</span>
              </div>
              {active.address && <div className="info__addr">{active.address}</div>}
              <div className="info__actions">
                <button className="btn btn-sm btn-secondary" onClick={() => setActive(null)}>Close</button>
                <button className="btn btn-sm btn-primary" onClick={() => setModal(active)}>Open details</button>
              </div>
            </div>
          </InfoWindow>
        )}
      </GoogleMap>

      {/* Mobile bottom sheet for active item */}
      {active && (
        <div className="map-sheet" role="dialog" aria-labelledby="sheet-title" aria-describedby="sheet-body">
          <div className="sheet-row">
            <div className="sheet-title" id="sheet-title">
              {primaryClass(active)}
              <span className="pill">{(active as any)?.predicted_classes?.slice(0,3)?.join(' · ')}</span>
            </div>
            <button className="sheet-close" aria-label="Close" onClick={() => setActive(null)}>✕</button>
          </div>
          <div className="sheet-body" id="sheet-body">
            {active.address ?? '—'}
          </div>
          <div className="sheet-actions">
            <button className="btn btn-secondary btn-sm" onClick={() => setActive(null)}>Close</button>
            <button className="btn btn-primary btn-sm" onClick={() => setModal(active)}>Open details</button>
          </div>
        </div>
      )}

      {/* Dedicated details modal state */}
      <IssueModal problem={modal} onClose={() => setModal(null)} />
    </div>
  );
}

/** Updates map styles when theme changes (no re-mount) */
function StyleSync({
  theme,
  getMap,
}: {
  theme: 'light' | 'dark';
  getMap: () => google.maps.Map | null | undefined;
}) {
  useEffect(() => {
    const m = getMap();
    if (m) m.setOptions({ styles: theme === 'dark' ? darkStyle : lightStyle });
  }, [theme, getMap]);
  return null;
}
