import { useEffect, useState } from "react";
import { useAuth } from "../auth/useAuth";
import { useNavigate, Link } from "react-router-dom";
import { GoogleMap, useJsApiLoader } from "@react-google-maps/api";
import { LIGHT_STYLE, DARK_STYLE } from "../styles/map-styles";
import "../styles/landing-page.css";

export const GOOGLE_MAPS_KEY =
  import.meta.env.VITE_GOOGLE_MAPS_API_KEY as string;

function getInitialTheme(): "light" | "dark" {
  const saved = localStorage.getItem("theme");
  if (saved === "light" || saved === "dark") return saved;
  const systemDark = window.matchMedia?.("(prefers-color-scheme: dark)")?.matches;
  return systemDark ? "dark" : "light";
}

export default function Landing() {
  const { user, loading } = useAuth();
  const nav = useNavigate();
  const [mapLoaded, setMapLoaded] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">(getInitialTheme());
  const [map, setMap] = useState<google.maps.Map | null>(null);

  const { isLoaded } = useJsApiLoader({
    googleMapsApiKey: GOOGLE_MAPS_KEY || "",
  });

  useEffect(() => {
    if (!loading && user) {
      nav(user.role === "admin" ? "/analytics" : "/map", { replace: true });
    }
  }, [user, loading, nav]);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("theme", theme);
    window.dispatchEvent(new Event("themechange"));
  }, [theme]);

  useEffect(() => {
    if (map) {
      map.setOptions({
        styles: theme === "dark" ? DARK_STYLE : LIGHT_STYLE,
      });
    }
  }, [map, theme]);

  if (loading) {
    return (
      <div className="fullCenter">
        <div className="spinner" aria-hidden />
        <div className="muted">Loading…</div>
      </div>
    );
  }

  const mapOptions = {
    center: { lat: 46.7712, lng: 23.6236 },
    zoom: 11,
    styles: theme === "dark" ? DARK_STYLE : LIGHT_STYLE,
    disableDefaultUI: true,
    gestureHandling: "none" as const,
  };

  return (
    <div className="page">
      {/* ---------- HERO ---------- */}
      <header className="hero">
        <div className="heroBg" aria-hidden />

        <div className="container">
          {/* Topbar */}
          <div className="topbar">
            <div className="brand">
              <span className="logoDot" />
              <span className="brandText">Urban AI</span>
            </div>
            <div className="topbarRight">
              <button
                className="themeToggle"
                onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
                aria-label="Toggle dark mode"
                title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
              >
                {theme === "dark" ? <IconSun /> : <IconMoon />}
              </button>
            </div>
          </div>

          <div className="heroGrid">
            <div className="copy">
              <span className="badge">
                <span className="badgeDot" /> Urban AI
              </span>

              <h1 className="title">
                Help keep <span className="city">Cluj-Napoca</span> clean &amp; safe.
              </h1>
              <p className="subtitle">
                Report street problems with a photo or video. Our AI highlights
                what matters so authorities can act faster.
              </p>

              <div className="ctaRow">
                <Link to="/map" className="cta ctaPrimary">
                  <IconMap /> Explore live map
                </Link>
                <Link to="/login" className="cta ctaGhost">
                  <IconShield /> Admin / Authorities
                </Link>
              </div>

              <ul className="stats">
                <li>
                  <strong>~10k</strong>
                  <span>reports processed</span>
                </li>
                <li>
                  <strong>30+</strong>
                  <span>issue classes</span>
                </li>
                <li>
                  <strong>100%</strong>
                  <span>district coverage</span>
                </li>
              </ul>
            </div>

            <div className="mapCard">
              {(!isLoaded || !mapLoaded) && <div className="mapSkeleton" />}
              {isLoaded && (
                <GoogleMap
                  mapContainerClassName="mapFrame"
                  center={mapOptions.center}
                  zoom={mapOptions.zoom}
                  options={mapOptions}
                  onLoad={(mapInstance) => {
                    setMap(mapInstance);
                    setMapLoaded(true);
                  }}
                />
              )}
              <div className="mapCaption">City overview • Google Maps</div>
            </div>
          </div>
        </div>
      </header>

      {/* ---------- FEATURES ---------- */}
      <section className="section">
        <div className="container">
          <div className="features">
            <article className="card">
              <div className="iconWrap" aria-hidden>
                <span>📷</span>
              </div>
              <h3 className="cardTitle">Report in seconds</h3>
              <p className="cardText">
                Upload an image or short clip, add a note, and you’re done. No
                account needed for citizens.
              </p>
            </article>

            <article className="card">
              <div className="iconWrap" aria-hidden>
                <span>🤖</span>
              </div>
              <h3 className="cardTitle">AI-assisted tagging</h3>
              <p className="cardText">
                Models detect potholes, trash, blocked sidewalks, damaged
                signage, and more—auto-labelled for faster triage.
              </p>
            </article>

            <article className="card">
              <div className="iconWrap" aria-hidden>
                <span>⚡</span>
              </div>
              <h3 className="cardTitle">Actionable for teams</h3>
              <p className="cardText">
                Clear evidence, locations, and classes help departments route
                and resolve issues quickly.
              </p>
            </article>
          </div>
        </div>
      </section>

      {/* ---------- HOW IT WORKS ---------- */}
      <section className="sectionAlt">
        <div className="container">
          <ol className="steps">
            <li>
              <span className="stepNum">1</span>
              <div>
                <h4>Capture</h4>
                <p>Take a photo/video of the issue and upload it.</p>
              </div>
            </li>
            <li>
              <span className="stepNum">2</span>
              <div>
                <h4>Detect</h4>
                <p>AI suggests classes and extracts location details.</p>
              </div>
            </li>
            <li>
              <span className="stepNum">3</span>
              <div>
                <h4>Resolve</h4>
                <p>Authorities track, prioritize, and close the loop.</p>
              </div>
            </li>
          </ol>
        </div>
      </section>

      {/* ---------- FOOTER ---------- */}
      <footer className="footer">
        <div className="container">
          <div className="footerRow">
            <span className="muted">
              © {new Date().getFullYear()} Urban AI • Built for communities
            </span>
            <nav className="footerNav">
              <Link to="/privacy">Privacy</Link>
              <Link to="/terms">Terms</Link>
              <a
                href="https://support.google.com/maps/answer/144349?hl=en"
                target="_blank"
                rel="noreferrer"
              >
                Map attribution
              </a>
            </nav>
          </div>
        </div>
      </footer>
    </div>
  );
}

function IconMap() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
      <path d="M9 18l-6 3V6l6-3 6 3 6-3v15l-6 3-6-3z" stroke="currentColor" strokeWidth="2"/>
      <path d="M9 3v15M15 6v15" stroke="currentColor" strokeWidth="2"/>
    </svg>
  );
}
function IconShield() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
      <path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3z" stroke="currentColor" strokeWidth="2"/>
    </svg>
  );
}
function IconMoon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
      <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" stroke="currentColor" strokeWidth="2"/>
    </svg>
  );
}
function IconSun() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="4" stroke="currentColor" strokeWidth="2"/>
      <path d="M12 2v2M12 20v2M4 12H2M22 12h-2M5 5l-1.5-1.5M20.5 20.5 19 19M5 19l-1.5 1.5M20.5 3.5 19 5" stroke="currentColor" strokeWidth="2"/>
    </svg>
  );
}
