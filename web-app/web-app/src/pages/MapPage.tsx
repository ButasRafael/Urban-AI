import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  GoogleMap,
  Marker,
  InfoWindow,
  useLoadScript,
  MarkerClustererF,
  Rectangle,
  GroundOverlay,
} from '@react-google-maps/api';

import type { Problem } from '../api/problems';
import { getProblems } from '../api/problems';
import Input from '../components/Input';
import Button from '../components/Button';
import IssueModal from '../components/IssueModal';
import { issuesSummaryByMedia, type IssuesSummary } from "../api/issues";
import { LIGHT_STYLE, DARK_STYLE } from '../styles/map-styles';
import '../styles/map-page.css';

/* ---------- Map constants ---------- */
const MAP_CONTAINER_CLASS = 'map-el';
const CENTER_CLUJ = { lat: 46.7712, lng: 23.6236 } as const;


/* ---------- Helpers ---------- */
type MarkerGroup = {
  key: string;
  address: string;
  position: google.maps.LatLngLiteral;
  problems: Problem[];
};

function parseBBoxParam(sp: URLSearchParams): google.maps.LatLngBoundsLiteral | null {
  const v = sp.get('bbox');
  if (!v) return null;
  const [s, w, n, e] = v.split(',').map(Number);
  return [s, w, n, e].every(Number.isFinite)
    ? { south: s, west: w, north: n, east: e }
    : null;
}

function boundsFromGeohash(gh: string): google.maps.LatLngBoundsLiteral | null {
  if (!gh) return null;
  const base32 = '0123456789bcdefghjkmnpqrstuvwxyz';
  let even = true;
  const lat: [number, number] = [-90, 90];
  const lon: [number, number] = [-180, 180];

  for (const ch of gh.toLowerCase()) {
    const cd = base32.indexOf(ch);
    if (cd === -1) break;
    for (let mask = 16; mask > 0; mask >>= 1) {
      if (even) {
        const mid = (lon[0] + lon[1]) / 2;
        if (cd & mask) {
          lon[0] = mid;
        } else {
          lon[1] = mid;
        }
      } else {
        const mid = (lat[0] + lat[1]) / 2;
        if (cd & mask) {
          lat[0] = mid;
        } else {
          lat[1] = mid;
        }
      }
      even = !even;
    }
  }

  return { south: lat[0], west: lon[0], north: lat[1], east: lon[1] };
}

function cssVar(name: string, fallback: string) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function usePinPalette(theme: 'light' | 'dark') {
  return useMemo(() => {
    const danger500 = cssVar('--danger-500', '#ef4444'); // fallback red
    const info500 = cssVar('--info-500', '#3b82f6'); // fallback blue
    const ringIdle = theme === 'dark' ? '#2a3a4a' : '#d9e2ee';
    const ringActive =
      theme === 'dark' ? cssVar('--primary-300', '#60a5fa') : cssVar('--primary-600', '#2563eb');

    return {
      imageTone: cssVar('--pin-image', danger500),
      videoTone: cssVar('--pin-video', info500),
      ringIdle,
      ringActive,
    };
  }, [theme]);
}

type PinState = 'default' | 'hover' | 'active';

const iconCache = new Map<string, google.maps.Icon>();

function makePinSvg(opts: {
  tone: string;
  theme: 'light' | 'dark';
  size?: number;
  state?: PinState;
  ringIdle: string;
  ringActive: string;
  label?: string;
}) {
  const { tone, theme, size = 32, state = 'default', ringIdle, ringActive, label } = opts;
  const w = size;
  const h = Math.round(size * 1.2667);
  const dark = theme === 'dark';

  const ringStroke = state === 'active' ? ringActive : ringIdle;
  const ringOpacity = state === 'active' ? (dark ? 0.55 : 0.7) : (dark ? 0.28 : 0.45);
  const topHi = dark ? '#203247' : '#ffffff';
  const topHiOpacity = dark ? (state !== 'default' ? 0.28 : 0.2) : (state !== 'default' ? 0.95 : 0.82);
  const dropGlow = dark ? (state !== 'default' ? 0.55 : 0.45) : (state !== 'default' ? 0.26 : 0.2);

  const textFill = dark ? '#e6eef9' : '#0b1220';

  return `
<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 30 38">
  <defs>
    <radialGradient id="rg" cx="50%" cy="36%" r="70%">
      <stop offset="0%" stop-color="${topHi}" stop-opacity="${topHiOpacity}"/>
      <stop offset="100%" stop-color="${tone}"/>
    </radialGradient>
    <filter id="sh" x="-50%" y="-50%" width="200%" height="200%">
      <feDropShadow dx="0" dy="2" stdDeviation="2.4" flood-opacity="${dropGlow}"/>
    </filter>
  </defs>

  <!-- base pin -->
  <path filter="url(#sh)"
        d="M15 1C8.373 1 3 6.318 3 12.9 3 24.2 15 37 15 37s12-12.8 12-24.1C27 6.318 21.627 1 15 1Z"
        fill="url(#rg)"/>

  <!-- inner ring -->
  <path d="M15 3.2c-5.4 0-9.8 4.24-9.8 9.56 0 9.73 9.8 20.67 9.8 20.67s9.8-10.94 9.8-20.67C24.8 7.44 20.4 3.2 15 3.2Z"
        fill="none" stroke="${ringStroke}" stroke-opacity="${ringOpacity}" stroke-width="1.6"/>

  <!-- center capsule (shows text when label present) -->
  <circle cx="15" cy="13" r="5.8" fill="${label ? (dark ? '#0f141a' : '#ffffff') : (dark ? '#0b1220' : '#ffffff')}"
          fill-opacity="${label ? (dark ? 0.95 : 0.98) : (dark ? 0.85 : 0.95)}"/>

  ${label ? `
    <text x="15" y="13.6" text-anchor="middle" dominant-baseline="central"
      font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto"
      font-size="8.8" font-weight="800" fill="${textFill}">${label}</text>
  ` : ''}

</svg>`;
}

function pinIcon(params: {
  theme: 'light' | 'dark';
  tone: string;
  state?: PinState;
  ringIdle: string;
  ringActive: string;
  size?: number;
  label?: string;
}): google.maps.Icon {
  const { theme, tone, state = 'default', ringIdle, ringActive, size, label } = params;
  const w = size ?? (state === 'active' ? 38 : state === 'hover' ? 34 : 30);
  const key = `${theme}|${tone}|${state}|${w}|${ringIdle}|${ringActive}|${label ?? ''}`;
  const cached = iconCache.get(key);
  if (cached) return cached;

  const h = Math.round(w * 1.2667);
  const svg = makePinSvg({ tone, theme, size: w, state, ringIdle, ringActive, label });
  const icon: google.maps.Icon = {
    url: 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(svg),
    scaledSize: new google.maps.Size(w, h),
    anchor: new google.maps.Point(Math.round(w / 2), h),
  };
  iconCache.set(key, icon);
  return icon;
}

function useToneFor(palette: ReturnType<typeof usePinPalette>) {
  return useCallback(
    (group: { problems: { media_type?: 'image' | 'video' }[] }) => {
      const hasVideo = group.problems.some((p) => p.media_type === 'video');
      return hasVideo ? palette.videoTone : palette.imageTone;
    },
    [palette]
  );
}

function myLocationIcon(theme: 'light' | 'dark'): google.maps.Icon {
  const ring = theme === 'dark' ? '#0f141a' : '#ffffff';
  const pulse = '#3b82f6';
  const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 26 26">
  <circle cx="13" cy="13" r="8.5" fill="${ring}" opacity=".9"/>
  <circle cx="13" cy="13" r="6.4" fill="${pulse}" opacity=".25"/>
  <circle cx="13" cy="13" r="4.2" fill="${pulse}"/>
</svg>`;
  return {
    url: 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(svg),
    scaledSize: new google.maps.Size(26, 26),
    anchor: new google.maps.Point(13, 13),
  };
}

function isDark(): 'light' | 'dark' {
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
}

function clusterCircleSvg(theme: 'light' | 'dark', s: number, tone: string) {
  const isDarkTheme = theme === 'dark';
  const stroke = isDarkTheme ? '#2a3a4a' : '#d9e2ee';
  const glow = isDarkTheme ? '.45' : '.18';
  const cx = s / 2;
  const cy = s / 2;
  const r = s / 2 - 2;

  return `
<svg xmlns="http://www.w3.org/2000/svg" width="${s}" height="${s}" viewBox="0 0 ${s} ${s}">
  <defs>
    <radialGradient id="rg" cx="50%" cy="38%" r="70%">
      <stop offset="0%"  stop-color="${isDarkTheme ? '#203247' : '#ffffff'}" stop-opacity="${isDarkTheme ? 0.22 : 1}"/>
      <stop offset="100%" stop-color="${tone}" />
    </radialGradient>
    <filter id="sh" x="-50%" y="-50%" width="200%" height="200%">
      <feDropShadow dx="0" dy="2" stdDeviation="2" flood-opacity="${glow}"/>
    </filter>
  </defs>

  <circle cx="${cx}" cy="${cy}" r="${r}" fill="url(#rg)" stroke="${stroke}" stroke-width="1.5" filter="url(#sh)"/>
  <circle cx="${cx}" cy="${cy}" r="${r - 5}" fill="none" stroke="rgba(255,255,255,${isDarkTheme ? 0.18 : 0.55})" stroke-width="2"/>
  <circle cx="${cx}" cy="${cy}" r="${Math.max(1, s*0.04)}" fill="${isDarkTheme ? '#b6c2d1' : '#6b7a90'}" opacity="${isDarkTheme ? 0.35 : 0.25}"/>
</svg>`;
}
const clusterUrl = (theme: 'light' | 'dark', size: number, tone: string) =>
  'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(clusterCircleSvg(theme, size, tone));

function getClusterStyles(theme: 'light' | 'dark') {
  const textColor = theme === 'dark' ? '#e6eef9' : '#0b1220';
  const sizes = [34, 40, 48, 58, 72] as const;
  const tones =
    theme === 'dark'
      ? ['#182635', '#1e3a5f', '#284a82', '#365fa6', '#4a75c6']
      : ['#eef4ff', '#dbeafe', '#c7d2fe', '#bcd2ff', '#a8c0ff'];

  return sizes.map((s, i) => ({
    url: clusterUrl(theme, s, tones[i]),
    height: s,
    width: s,
    textColor,
    textSize: i >= 3 ? 14 : 13,
  }));
}

type MCCalcResult = { text: string; index: number; title?: string };
type MCCalculator = (markers: google.maps.Marker[], numStyles: number) => MCCalcResult;

const clusterCalculator: MCCalculator = (markers, numStyles) => {
  const count = markers.length;

  let idx = count >= 1000 ? 5 : count >= 200 ? 4 : count >= 50 ? 3 : count >= 10 ? 2 : 1;
  idx = Math.min(idx, numStyles);

  const formatCount = (n: number) =>
    n >= 1_000_000
      ? (n / 1_000_000).toFixed(n % 1_000_000 >= 100_000 ? 1 : 0) + 'M'
      : n >= 1_000
      ? (n / 1_000).toFixed(n % 1_000 >= 100 ? 1 : 0) + 'k'
      : String(n);

  return { text: formatCount(count), index: idx, title: `${count} locations` };
};

function fitPadding(): google.maps.Padding {
  return { top: 96, right: 64, bottom: 64, left: 64 };
}

function pinSizeForZoom(z?: number | null) {
  if (!z) return 30;
  return Math.round(28 + Math.min(6, Math.max(0, (z - 12) * 1.1)));
}

function bboxSizeMeters(b: google.maps.LatLngBoundsLiteral) {
  const midLat = (b.north + b.south) / 2;
  const metersPerDegLat = 110_574; // ~
  const metersPerDegLon = 111_320 * Math.cos((midLat * Math.PI) / 180);
  const h = Math.max(0, (b.north - b.south) * metersPerDegLat);
  const w = Math.max(0, (b.east - b.west) * metersPerDegLon);
  return { w, h };
}
function fmtMeters(m: number) {
  if (m >= 1000) return `${(m / 1000).toFixed(m >= 10_000 ? 0 : 1)} km`;
  if (m >= 100) return `${Math.round(m)} m`;
  return `${m.toFixed(1)} m`;
}

function summarize(s?: IssuesSummary): string {
  if (!s) return "…";
  const sevOrder: Array<keyof IssuesSummary> = ["high", "medium", "low"];
  const chunks: string[] = [];

  for (const sev of sevOrder) {
    const bucket = s[sev];
    if (!bucket) continue;

    const open = bucket.open ?? 0;
    const resolved = bucket.resolved ?? 0;
    const ignored = bucket.ignored ?? 0;

    const parts: string[] = [];
    if (open) parts.push(`${open} open`);
    if (resolved) parts.push(`${resolved} resolved`);
    if (ignored) parts.push(`${ignored} ignored`);

    if (parts.length) chunks.push(`${sev} ${parts.join(", ")}`);
  }

  return chunks.length ? `severity: ${chunks.join(" · ")}` : "no detections";
}


export default function MapPage() {
  const { isLoaded, loadError } = useLoadScript({
    googleMapsApiKey: import.meta.env.VITE_GOOGLE_MAPS_API_KEY as string,
    id: 'urban-ai-map',
  });

  /* ---------- State ---------- */
  const [mediaType, setMediaType] = useState<'all' | 'image' | 'video'>(() => {
    const sp = new URLSearchParams(location.search);
    const t = sp.get('type');
    return t === 'image' || t === 'video' ? t : 'all';
  });

  const [klassInput, setKlassInput] = useState(
    () => new URLSearchParams(location.search).get('q') ?? ''
  );
  const [klass, setKlass] = useState(klassInput);

  const mediaParam = useMemo(() => {
  const v = new URLSearchParams(location.search).get("media");
  return v ? Number(v) : null;
}, []);


  const [allProblems, setAllProblems] = useState<Problem[]>([]);
  const [problems, setProblems] = useState<Problem[]>([]);
  const [active, setActive] = useState<MarkerGroup | null>(null);
  const [modalProblem, setModalProblem] = useState<Problem | null>(null);
  const [myLoc, setMyLoc] = useState<google.maps.LatLngLiteral | null>(null);

  const [summCache, setSummCache] = useState<Record<number, IssuesSummary>>({});

  useEffect(() => {
    if (!active) return;
    const ids = active.problems
      .map(p => p.media_id)
      .filter(id => !(id in summCache));
    if (ids.length === 0) return;

    issuesSummaryByMedia(ids).then((resp) => {
      setSummCache(prev => ({ ...prev, ...resp }));
    });
  }, [active, summCache]);

  const [theme, setTheme] = useState<'light' | 'dark'>(isDark());
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);
  const [zoom, setZoom] = useState<number | null>(null);

  const [highlight, setHighlight] = useState<google.maps.LatLngBoundsLiteral | null>(null);
  const [selectionLabel, setSelectionLabel] = useState<string | null>(null);

  const mapRef = useRef<google.maps.Map | null>(null);
  const searchRef = useRef<HTMLInputElement | null>(null);

  const clusterStyles = useMemo(() => getClusterStyles(theme), [theme]);

  const palette = usePinPalette(theme);
  const toneFor = useToneFor(palette);
  const isTouch = useMemo(
    () => window.matchMedia?.('(pointer: coarse)').matches ?? false,
    []
  );

  useEffect(() => {
    const handler = () => setTheme(isDark());
    window.addEventListener('themechange', handler);
    const obs = new MutationObserver(handler);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => {
      window.removeEventListener('themechange', handler);
      obs.disconnect();
    };
  }, []);

  useEffect(() => {
    getProblems({
      media_type: mediaType === 'all' ? undefined : mediaType,
    }).then((items) => setAllProblems(items));
  }, [mediaType]);

  useEffect(() => {
    const filtered = klass
      ? allProblems.filter((p) =>
          p.predicted_classes?.some((c) => c.toLowerCase().includes(klass.toLowerCase()))
        )
      : allProblems;
    setProblems(filtered);
  }, [klass, allProblems]);

  useEffect(() => {
    const t = setTimeout(() => setKlass(klassInput.trim()), 240);
    return () => clearTimeout(t);
  }, [klassInput]);

  useEffect(() => {
    const sp = new URLSearchParams(location.search);
    if (klass) sp.set('q', klass);
    else sp.delete('q');
    if (mediaType !== 'all') sp.set('type', mediaType);
    else sp.delete('type');
    const newUrl = `${location.pathname}?${sp.toString()}`;
    history.replaceState(null, '', newUrl);
  }, [klass, mediaType]);

  const markers = useMemo<MarkerGroup[]>(() => {
    const map = new Map<string, MarkerGroup>();
    for (const p of problems) {
      if (p.latitude == null || p.longitude == null) continue;
      const key = `${p.latitude},${p.longitude}`;
      const mg = map.get(key);
      if (!mg) {
        map.set(key, {
          key,
          address: p.address ?? '(unknown)',
          position: { lat: p.latitude, lng: p.longitude },
          problems: [p],
        });
      } else {
        mg.problems.push(p);
      }
    }
    return Array.from(map.values());
  }, [problems]);

  useEffect(() => {
  if (!mediaParam || problems.length === 0 || !mapRef.current) return;
  const target = problems.find(p => p.media_id === mediaParam);
  if (!target || target.latitude == null || target.longitude == null) return;

  const key = `${target.latitude},${target.longitude}`;
  const group = markers.find(g => g.key === key);
  if (!group) return;

  setActive(group);
  const m = mapRef.current;
  m.panTo(group.position);
  m.setZoom(17);
}, [mediaParam, problems, markers]);

  const onLoad = useCallback(
    (m: google.maps.Map) => {
      mapRef.current = m;
      setZoom(m.getZoom() ?? null);
      m.addListener('zoom_changed', () => setZoom(m.getZoom() ?? null));

      const bounds = new google.maps.LatLngBounds();
      markers.forEach((g) => bounds.extend(g.position));
      if (!bounds.isEmpty()) {
        m.fitBounds(bounds, fitPadding());
      } else {
        m.setCenter(CENTER_CLUJ);
        m.setZoom(12);
      }
    },
    [markers]
  );

  const onUnmount = useCallback(() => {
    mapRef.current = null;
  }, []);

  useEffect(() => {
    const m = mapRef.current;
    if (m) {
      m.setOptions({
        styles: theme === 'dark' ? DARK_STYLE : LIGHT_STYLE,
        backgroundColor:
            theme === 'dark'
                ? cssVar('--secondary-100', '#0F161C')
                : '#F3F7F6',
      });
    }
  }, [theme]);

  useEffect(() => {
    const m = mapRef.current;
    if (!m || markers.length === 0) return;
    const b = new google.maps.LatLngBounds();
    for (const g of markers) b.extend(g.position);
    m.fitBounds(b, fitPadding());
  }, [markers]);

  const [urlSearch, setUrlSearch] = useState(location.search);

  useEffect(() => {
    const handleUrlChange = () => setUrlSearch(location.search);
    window.addEventListener('popstate', handleUrlChange);
    return () => window.removeEventListener('popstate', handleUrlChange);
  }, []);


  useEffect(() => {
    if (!isLoaded) return;
    const sp = new URLSearchParams(urlSearch);
    const gh = sp.get('geohash');
    const bbox =
      parseBBoxParam(sp) || (gh ? boundsFromGeohash(gh) : null);
    if (!bbox) {
      setHighlight(null);
      setSelectionLabel(null);
      return;
    }

    setHighlight(bbox);
    const { w, h } = bboxSizeMeters(bbox);
    const dims = `${fmtMeters(w)} × ${fmtMeters(h)}`;
    setSelectionLabel(
      gh ? `Selection • geohash "${gh}" • ${dims}` : `Selection • bbox • ${dims}`
    );

    const m = mapRef.current;
    if (m) {
      const b = new google.maps.LatLngBounds(
        new google.maps.LatLng(bbox.south, bbox.west),
        new google.maps.LatLng(bbox.north, bbox.east)
      );
      m.fitBounds(b, fitPadding());
    }
  }, [isLoaded, urlSearch]);

  const fitToLocation = useCallback((pt: google.maps.LatLngLiteral, radiusMeters = 60) => {
    const m = mapRef.current;
    if (!m) return;

    const dLat = radiusMeters / 110_574;
    const dLng = radiusMeters / (111_320 * Math.cos((pt.lat * Math.PI) / 180));

    const bounds = new google.maps.LatLngBounds(
      { lat: pt.lat - dLat, lng: pt.lng - dLng },
      { lat: pt.lat + dLat, lng: pt.lng + dLng }
    );
    m.fitBounds(bounds, { top: 48, right: 48, bottom: 48, left: 48 });
  }, []);

  const fitToMarkers = useCallback(() => {
    const m = mapRef.current;
    if (!m) return;
    const b = new google.maps.LatLngBounds();
    markers.forEach((g) => b.extend(g.position));

    if (!b.isEmpty()) {
      m.fitBounds(b, fitPadding());
      return;
    }
    if (myLoc) {
      m.panTo(myLoc);
      m.setZoom(15);
    } else {
      m.panTo(CENTER_CLUJ);
      m.setZoom(12);
    }
  }, [markers, myLoc]);

  const locateMe = useCallback(() => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const pt = { lat: pos.coords.latitude, lng: pos.coords.longitude };
        setMyLoc(pt);
        fitToLocation(pt);
      },
      () => {},
      {
        enableHighAccuracy: true,
        timeout: 15000,
        maximumAge: 60000
      }
    );
  }, [fitToLocation]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      switch (e.key.toLowerCase()) {
        case '/':
          e.preventDefault();
          searchRef.current?.focus();
          break;
        case 'f':
          e.preventDefault();
          if (e.shiftKey && myLoc && mapRef.current) {
            const b = new google.maps.LatLngBounds();
            markers.forEach((g) => b.extend(g.position));
            b.extend(myLoc);
            mapRef.current.fitBounds(b, fitPadding());
          } else {
            fitToMarkers();
          }
          break;
        case 'l':
          e.preventDefault();
          locateMe();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [fitToMarkers, locateMe]);

  const clearSelection = useCallback(() => {
    const sp = new URLSearchParams(location.search);
    sp.delete('bbox');
    sp.delete('geohash');
    const newUrl = `${location.pathname}?${sp.toString()}`;

    // Force a complete page reload to ensure map refreshes
    window.location.href = newUrl;
  }, []);

  if (loadError) {
    return (
      <div className="map-wrap">
        <div className="map-fallback card">
          <div className="map-fallback__inner">
            <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden>
              <path
                d="M12 9v4m0 4h.01M10.29 3.86l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.71-3.14l-8-14a2 2 0 0 0-3.42 0Z"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              />
            </svg>
            <div>
              <strong>Map failed to load</strong>
              <p className="muted">Check your API key/connection then refresh.</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!isLoaded) {
    return (
      <div className="map-wrap">
        <div className="map-skeleton">
          <div className="sk shimmer" />
        </div>
      </div>
    );
  }

  return (
    <div className="map-wrap" role="application">
      {/* Floating filter bar (glass) */}
      <div className="mapbar mapbar--floating">
        <div className="mapbar-left">
          <Input
            ref={searchRef}
            placeholder="Filter by class…"
            value={klassInput}
            onChange={(e) => setKlassInput(e.target.value)}
            stopGlobalKeys
            prefixIcon={
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="7" />
                <path d="M21 21l-4.3-4.3" />
              </svg>
            }
            clearable
          />

          <div className="select-wrap">
            <select
              className="select"
              value={mediaType}
              onChange={(e) => setMediaType(e.target.value as 'all' | 'image' | 'video')}
              title="Media type"
              aria-label="Media type"
            >
              <option value="all">All types</option>
              <option value="image">Images</option>
              <option value="video">Videos</option>
            </select>
          </div>

          {(klass || mediaType !== 'all') && (
            <div className="active-filters">
              {klass && <span className="pill">class: “{klass}”</span>}
              {mediaType !== 'all' && <span className="pill">{mediaType}</span>}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setKlassInput('');
                  setKlass('');
                  setMediaType('all');
                }}
                leftIcon={
                  <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M18 6L6 18M6 6l12 12" />
                  </svg>
                }
              >
                Clear
              </Button>
            </div>
          )}

          {selectionLabel && (
            <div className="selection-chip" title={selectionLabel}>
              <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden>
                <path d="M3 7h18M7 3v18" stroke="currentColor" strokeWidth="2" fill="none" />
              </svg>
              <span className="selection-chip__txt">{selectionLabel}</span>
              <button className="selection-chip__clear" onClick={clearSelection} aria-label="Clear selection">
                <svg width="12" height="12" viewBox="0 0 24 24" aria-hidden>
                  <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" fill="none" />
                </svg>
              </button>
            </div>
          )}
        </div>

        <div className="mapbar-right">
          <div className="stat">
            <span className="dot -uploads" />
            {problems.length} uploads
          </div>
          <div className="stat">
            <span className="dot -locs" />
            {markers.length} locations
          </div>

          <div className="kbd-hint" aria-hidden>
            <kbd>/</kbd> search · <kbd>F</kbd> fit · <kbd>L</kbd> locate
          </div>

          <Button
            variant="secondary"
            size="sm"
            iconOnly
            title="Fit to markers"
            aria-label="Fit to markers"
            onClick={fitToMarkers}
          >
            <svg width="16" height="16" viewBox="0 0 24 24">
              <path d="M3 3h7M3 3v7M21 3h-7M21 3v7M3 21h7M3 21v-7M21 21h-7M21 21v-7" fill="none" stroke="currentColor" strokeWidth="2" />
            </svg>
          </Button>
          <Button
            variant="secondary"
            size="sm"
            iconOnly
            title="My location"
            aria-label="My location"
            onClick={locateMe}
          >
            <svg width="16" height="16" viewBox="0 0 24 24">
              <path d="M12 8v8m-4-4h8" stroke="currentColor" strokeWidth="2" fill="none" />
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" fill="none" />
            </svg>
          </Button>
        </div>
      </div>

      {/* Map */}
      <GoogleMap
        onLoad={onLoad}
        onUnmount={onUnmount}
        mapContainerClassName={MAP_CONTAINER_CLASS}
        center={CENTER_CLUJ}
        zoom={12}
        options={{
          streetViewControl: false,
          mapTypeControl: false,
          fullscreenControl: false,
          clickableIcons: false,
          gestureHandling: 'greedy',
          disableDefaultUI: true,
          keyboardShortcuts: false,
          styles: theme === 'dark' ? DARK_STYLE : LIGHT_STYLE,
          backgroundColor:
              theme === 'dark'
                  ? cssVar('--secondary-100', '#0F161C')
                  : '#F3F7F6',

        }}
      >
        {/* Highlighted selection (bbox/geohash) */}
        {highlight ? (
          <React.Fragment key={`highlight-${JSON.stringify(highlight)}`}>
            <Rectangle
              bounds={highlight}
              options={{
                strokeColor: cssVar('--select-stroke-outer', 'rgba(37,99,235,.35)'),
                strokeOpacity: 0.35,
                strokeWeight: 6,
                fillOpacity: 0,
                clickable: false,
              }}
            />
            <Rectangle
              bounds={highlight}
              options={{
                strokeColor: cssVar('--select-stroke', '#2563eb'),
                strokeOpacity: 0.95,
                strokeWeight: 2,
                fillColor: cssVar('--select-fill', 'rgba(37,99,235,.12)'),
                fillOpacity: 1,
                clickable: false,
              }}
            />
            <GroundOverlay
              bounds={highlight}
              url={(() => {
                const c = cssVar('--select-stroke', '#2563eb');
                const op = parseFloat(cssVar('--select-hatch-opacity', '.12')) || 0.12;
                const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 40 40">
  <defs>
    <pattern id="p" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <rect width="4" height="8" fill="${c}" opacity="${op}"/>
    </pattern>
  </defs>
  <rect width="100%" height="100%" fill="url(#p)"/>
</svg>`;
                return 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(svg.trim());
              })()}
              opacity={1}
            />
          </React.Fragment>
        ) : null}

        <MarkerClustererF
          key={`clusters-${theme}`}
          averageCenter
          gridSize={64}
          options={{
            styles: clusterStyles,
            calculator: clusterCalculator,
            minimumClusterSize: 2,
            maxZoom: 18,
          }}
        >
          {(clusterer) => (
            <>
              {markers.map((m) => {
                const tone = toneFor(m);
                const isActive = active?.key === m.key;
                const isHover = hoveredKey === m.key;
                const state: PinState = isActive ? 'active' : isHover ? 'hover' : 'default';
                const label = m.problems.length > 1 ? String(m.problems.length) : undefined;

                // lift hovered/active pins above others
                const baseZ = Math.round((m.position.lat + 90) * 100);
                const z = baseZ + (isHover ? 100000 : 0) + (isActive ? 200000 : 0);

                return (
                  <Marker
                    key={m.key}
                    position={m.position}
                    clusterer={clusterer}
                    title={`${m.address} — ${m.problems.length} upload${m.problems.length > 1 ? 's' : ''}`}
                    zIndex={z}
                    icon={pinIcon({
                      theme,
                      tone,
                      state,
                      ringIdle: palette.ringIdle,
                      ringActive: palette.ringActive,
                      label,
                      size: pinSizeForZoom(zoom),
                    })}
                    onMouseOver={() => !isTouch && setHoveredKey(m.key)}
                    onMouseOut={() => !isTouch && setHoveredKey((k) => (k === m.key ? null : k))}
                    onClick={() => setActive(m)}
                  />
                );
              })}
            </>
          )}
        </MarkerClustererF>

        {myLoc && (
          <Marker position={myLoc} icon={myLocationIcon(theme)} zIndex={Number.MAX_SAFE_INTEGER} />
        )}

        {active && (
          <InfoWindow
            position={active.position}
            onCloseClick={() => setActive(null)}
            options={{ maxWidth: 300 }}
          >
            <div className="infocard">
              <div className="infocard__head">
                <h4 className="infocard__title">{active.address}</h4>
                <span className="pill">
                  {active.problems.length} upload{active.problems.length > 1 ? 's' : ''}
                </span>
              </div>

              <div className="infocard__list">
                {active.problems.map((p) => (
                    <div key={p.media_id} className="infocard__item">
                      <div className="infocard__meta">
                        <strong>#{p.media_id}</strong>
                        <span className="muted">{new Date(p.created_at).toLocaleString()}</span>
                      </div>
                      <div className="infocard__classes">
                        {(p.predicted_classes?.slice(0, 4) ?? []).map((c, i) => (
                            <span key={`${c}-${i}`} className="chip">
                          {c}
                        </span>
                        ))}
                        {(p.predicted_classes?.length ?? 0) > 4 && (
                            <span className="chip chip--muted">+{p.predicted_classes!.length - 4}</span>
                        )}
                      </div>

                      <div className="infocard__summary">
                        <span className="muted">{summarize(summCache[p.media_id])}</span>{" "}
                        <a className="ml8" href={`/issues?media_id=${p.media_id}`}>See detections →</a>
                      </div>


                      {p.annotated_video_url ? (
                          <div className="infocard__video-thumb" onClick={() => setModalProblem(p)}>
                              <video
                                  className="infocard__img"
                                  src={`${import.meta.env.VITE_API_BASE}${p.annotated_video_url}`}
                                  muted
                                  preload="metadata"
                                  poster={`${import.meta.env.VITE_API_BASE}${p.annotated_image_url || `/static/${p.media_id}.jpg`}`}
                              />
                              <div className="infocard__video-overlay">
                                  <svg width="48" height="48" viewBox="0 0 24 24" fill="white" opacity="0.9">
                                      <circle cx="12" cy="12" r="10" fill="rgba(0,0,0,0.6)" />
                                      <path d="M10 8l6 4-6 4V8z" />
                                  </svg>
                              </div>
                          </div>
                      ) : p.annotated_image_url && (
                          <img
                              className="infocard__img"
                              src={`${import.meta.env.VITE_API_BASE}${p.annotated_image_url}`}
                              alt=""
                              loading="lazy"
                          />
                      )}

                      <div className="infocard__actions">
                        <Button variant="secondary" size="sm" onClick={() => setModalProblem(p)}>
                          Open details
                        </Button>
                      </div>
                    </div>
                ))}
              </div>
            </div>
          </InfoWindow>
        )}
      </GoogleMap>

      {modalProblem && <IssueModal problem={modalProblem} onClose={() => setModalProblem(null)} />}
    </div>
  );
}
