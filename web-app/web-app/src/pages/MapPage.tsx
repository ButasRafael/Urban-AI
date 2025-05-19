import React, { useEffect, useMemo, useState, useCallback } from "react";
import { GoogleMap, InfoWindow, Marker, useLoadScript } from "@react-google-maps/api";
import type { Problem } from "../api/problems";
import { getProblems } from "../api/problems";
import Input from "../components/Input";
import Button from "../components/Button";
import { colors, spacing } from "../theme";

const MAP_STYLE: React.CSSProperties = {
  width: "100%",
  height: "calc(100vh - 72px)", // filter bar space
};
const CENTER_CLUJ = { lat: 46.7712, lng: 23.6236 } as const;

function mulberry(a: number) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function fakeCoords(seed: string) {
  let h = 0;
  for (let i = 0; i < seed.length; i++) {
    h = (h << 5) - h + seed.charCodeAt(i);
  }
  const rnd = mulberry(h);
  return {
    lat: CENTER_CLUJ.lat + (rnd() - 0.5) * 0.04,
    lng: CENTER_CLUJ.lng + (rnd() - 0.5) * 0.04,
  } as const;
}

interface MarkerData {
  address: string;
  position: google.maps.LatLngLiteral;
  problems: Problem[];
}

export default function MapPage() {
  // 1) Load Google Maps script
  const { isLoaded, loadError } = useLoadScript({
    googleMapsApiKey: import.meta.env.VITE_GOOGLE_MAPS_API_KEY as string,
  });

  // 2) App state
  const [mediaType, setMediaType] = useState<"all" | "image" | "video">("all");
  const [klass, setKlass] = useState("");
  const [problems, setProblems] = useState<Problem[]>([]);
  const [active, setActive] = useState<MarkerData | null>(null);
  const [map, setMap] = useState<google.maps.Map | null>(null);

  // 3) Map onLoad callback
  const onLoad = useCallback((mapInstance: google.maps.Map) => {
    setMap(mapInstance);
  }, []);

  // 4) Fetch data when filters change
  useEffect(() => {
    // Fetch by media type only; do substring filter client-side
    getProblems({
      media_type: mediaType === "all" ? undefined : mediaType,
    }).then(items => {
      const filtered = klass
        ? items.filter(p =>
            p.predicted_classes.some(c =>
              c.toLowerCase().includes(klass.toLowerCase())
            )
          )
        : items;
      setProblems(filtered);
    });
  }, [mediaType, klass]);
  // 5) Prepare marker icons once API is ready
  const PIN_ICONS = useMemo(() => {
    if (!isLoaded || !window.google) {
      return { image: undefined, video: undefined };
    }
    return {
      image: {
        url: 'https://maps.google.com/mapfiles/ms/icons/red-dot.png',
        scaledSize: new window.google.maps.Size(32, 32),
      },
      video: {
        url: 'https://maps.google.com/mapfiles/ms/icons/blue-dot.png',
        scaledSize: new window.google.maps.Size(32, 32),
      },
    };
  }, [isLoaded]);

  // 6) Group problems by address and generate fake coords
  // 6) Use exact lat/lng and group by those
  const markers = useMemo<MarkerData[]>(() => {
    const grouping = new Map<string, MarkerData>();
    problems.forEach(p => {
      if (p.latitude == null || p.longitude == null) return;
      const key = `${p.latitude},${p.longitude}`;
      const md = grouping.get(key);
      if (!md) {
        grouping.set(key, {
          address: p.address ?? '(unknown)',
          position: { lat: p.latitude, lng: p.longitude },
          problems: [p],
        });
      } else {
        md.problems.push(p);
      }
    });
    return Array.from(grouping.values());
  }, [problems]);

  // 7) Early returns after all hooks
  if (loadError) {
    return <p>Map failed to load.</p>;
  }
  if (!isLoaded) {
    return <p>Loading maps…</p>;
  }

  // 8) Main render
  return (
    <>
      <div
        style={{
          display: "flex",
          gap: spacing.s,
          padding: spacing.s,
          background: colors.surface,
          borderBottom: `1px solid ${colors.muted}`,
        }}
      >
        <Input
          placeholder="filter by class…"
          value={klass}
          onChange={(e) => setKlass(e.target.value)}
          style={{ maxWidth: 230 }}
        />

        <select
          value={mediaType}
          onChange={(e) => setMediaType(e.target.value as "all" | "image" | "video")}
          style={{ padding: spacing.s }}
        >
          <option value="all">all types</option>
          <option value="image">images</option>
          <option value="video">videos</option>
        </select>

        <Button
          variant="ghost"
          onClick={() => {
            setKlass("");
            setMediaType("all");
            if (map) {
              map.panTo(CENTER_CLUJ);
              map.setZoom(12);
            }
          }}
        >
          reset
        </Button>

        <div style={{ marginLeft: "auto", padding: spacing.s, color: colors.muted }}>
          {problems.length} uploads → {markers.length} locations
        </div>
      </div>

      <GoogleMap
        mapContainerStyle={MAP_STYLE}
        center={CENTER_CLUJ}
        zoom={12}
        onLoad={onLoad}
        options={{
          streetViewControl: false,
          mapTypeControl: false,
          fullscreenControl: false,
        }}
      >
        {markers.map((m) => (
          <Marker
            key={m.address}
            position={m.position}
            onClick={() => setActive(m)}
            icon={
              m.problems.some((p) => p.media_type === "video")
                ? PIN_ICONS.video
                : PIN_ICONS.image
            }
          />
        ))}

        {active && (
          <InfoWindow position={active.position} onCloseClick={() => setActive(null)}>
            <div style={{ maxWidth: 260 }}>
              <h4 style={{ margin: "0 0 4px" }}>{active.address}</h4>
              <p style={{ margin: "0 0 8px", fontSize: 12, color: colors.muted }}>
                {active.problems.length} upload
                {active.problems.length > 1 && "s"} here
              </p>

              {active.problems.map((p) => (
                <div
                  key={p.media_id}
                  style={{
                    borderTop: `1px solid ${colors.muted}`,
                    paddingTop: 4,
                    marginTop: 4,
                  }}
                >
                  <strong>#{p.media_id}</strong>{" "}
                  <em style={{ fontSize: 11 }}>
                    {new Date(p.created_at).toLocaleString()}
                  </em>
                  <br />
                  {p.predicted_classes.join(", ") || "(no classes)"}
                  {p.annotated_image_url && (
                    <img
                      src={`${import.meta.env.VITE_API_BASE}/static/${p.media_id}.jpg`}
                      style={{
                        width: "100%",
                        borderRadius: 4,
                        marginTop: 4,
                        boxShadow: "0 1px 3px rgba(0,0,0,.3)",
                      }}
                    />
                  )}
                  {p.annotated_video_url && <div style={{ marginTop: 4 }}>🎞️ video</div>}
                </div>
              ))}
            </div>
          </InfoWindow>
        )}
      </GoogleMap>
    </>
  );
}
