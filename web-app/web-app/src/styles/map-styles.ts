export const LIGHT_STYLE: google.maps.MapTypeStyle[] = [
  // Base land + labels
  { elementType: "geometry", stylers: [{ color: "#F3F7F6" }] },
  { elementType: "labels.text.fill", stylers: [{ color: "#2C3E50" }] },
  { elementType: "labels.text.stroke", stylers: [{ color: "#FFFFFF" }, { saturation: 0 }, { lightness: 0 }] },

  // Water (very light teal so pins pop)
  { featureType: "water", elementType: "geometry", stylers: [{ color: "#E8F6F3" }] },
  { featureType: "water", elementType: "labels.text.fill", stylers: [{ color: "#1F2C38" }] },

  // Natural / parks (slightly greener than land, still muted)
  { featureType: "landscape.natural", elementType: "geometry", stylers: [{ color: "#EAF4F1" }] },
  { featureType: "poi.park", elementType: "geometry", stylers: [{ color: "#E6F2EF" }] },
  { featureType: "poi.park", elementType: "labels.text.fill", stylers: [{ color: "#2C3E50" }] },

  // Man-made land (kept near land to avoid blotches)
  { featureType: "landscape.man_made", elementType: "geometry", stylers: [{ color: "#F5F8F7" }] },

  // Roads
  { featureType: "road", elementType: "geometry", stylers: [{ color: "#E4E8EC" }] },
  { featureType: "road", elementType: "geometry.stroke", stylers: [{ color: "#D2D9E0" }] },
  { featureType: "road.local", elementType: "geometry", stylers: [{ color: "#EDF1F3" }] },
  { featureType: "road.arterial", elementType: "geometry", stylers: [{ color: "#E6EBEF" }] },
  { featureType: "road.highway", elementType: "geometry", stylers: [{ color: "#C7DCEC" }] },
  { featureType: "road.highway", elementType: "geometry.stroke", stylers: [{ color: "#B5C6DA" }] },

  // Transit lines toned down (but not removed)
  { featureType: "transit.line", elementType: "geometry", stylers: [{ color: "#DBE6EF" }] },
  { featureType: "transit.station", stylers: [{ visibility: "off" }] },

  // Admin boundaries (subtle)
  { featureType: "administrative", elementType: "geometry.stroke", stylers: [{ color: "#D5E2EC" }] },
  { featureType: "administrative.locality", elementType: "labels.text.fill", stylers: [{ color: "#2C3E50" }] },

  // POI/Icons cleanup
  { featureType: "poi", stylers: [{ visibility: "off" }] },
  { elementType: "labels.icon", stylers: [{ visibility: "off" }] }
];

export const DARK_STYLE: google.maps.MapTypeStyle[] = [
  // Base land + labels
  { elementType: "geometry", stylers: [{ color: "#0F161C" }] },
  { elementType: "labels.text.fill", stylers: [{ color: "#ECF0F1" }] },
  { elementType: "labels.text.stroke", stylers: [{ color: "#0F161C" }] },

  // Water (deep teal, darker than pins)
  { featureType: "water", elementType: "geometry", stylers: [{ color: "#0B4F45" }] },
  { featureType: "water", elementType: "labels.text.fill", stylers: [{ color: "#E8F6F3" }] },

  // Natural / parks
  { featureType: "landscape.natural", elementType: "geometry", stylers: [{ color: "#10262B" }] },
  { featureType: "poi.park", elementType: "geometry", stylers: [{ color: "#0E332C" }] },
  { featureType: "poi.park", elementType: "labels.text.fill", stylers: [{ color: "#CFE4DE" }] },

  // Man-made land
  { featureType: "landscape.man_made", elementType: "geometry", stylers: [{ color: "#131D26" }] },

  // Roads (darker base, faint strokes so pins/clusters read clearly)
  { featureType: "road", elementType: "geometry", stylers: [{ color: "#233141" }] },
  { featureType: "road", elementType: "geometry.stroke", stylers: [{ color: "#1A2531" }] },
  { featureType: "road.local", elementType: "geometry", stylers: [{ color: "#223041" }] },
  { featureType: "road.arterial", elementType: "geometry", stylers: [{ color: "#243444" }] },
  { featureType: "road.highway", elementType: "geometry", stylers: [{ color: "#1F2C38" }] },
  { featureType: "road.highway", elementType: "geometry.stroke", stylers: [{ color: "#203247" }] },

  // Transit (very faint)
  { featureType: "transit.line", elementType: "geometry", stylers: [{ color: "#1A2A38" }] },
  { featureType: "transit.station", stylers: [{ visibility: "off" }] },

  // Admin boundaries (subtle)
  { featureType: "administrative", elementType: "geometry.stroke", stylers: [{ color: "#1C2A36" }] },
  { featureType: "administrative.locality", elementType: "labels.text.fill", stylers: [{ color: "#E0E6EC" }] },

  // POI/Icons cleanup
  { featureType: "poi", stylers: [{ visibility: "off" }] },
  { elementType: "labels.icon", stylers: [{ visibility: "off" }] }
];
