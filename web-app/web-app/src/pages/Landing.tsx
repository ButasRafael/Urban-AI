import { useEffect } from 'react';
import { useAuth } from '../auth/useAuth';
import { useNavigate, Link } from 'react-router-dom';

export const GOOGLE_MAPS_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY as string;

export default function Landing() {
    const { user, loading } = useAuth();
  const nav = useNavigate();

  useEffect(() => {
    if (!loading) {
      if (user) {
        nav(user.role === 'admin' ? '/analytics' : '/map', { replace: true });
      }
    }
  }, [user, loading, nav]);

  if (loading) {
    return <p style={{ textAlign: 'center', marginTop: 50 }}>Loading…</p>;
  }

  return (
    <div style={{ padding: 'var(--l)', textAlign: 'center' }}>
      <h1 style={{ color: 'var(--primary)', fontSize: 32, fontWeight: 700 }}>
        Urban AI
      </h1>
      <p style={{ marginTop: 'var(--m)' }}>
        Help keep <strong>Cluj-Napoca</strong> clean & safe – upload photos or videos!
      </p>
      <iframe
        title="map"
        width="100%"
        height="450"
        loading="lazy"
        style={{
          border: 0,
          marginTop: 'var(--l)',
          borderRadius: 'var(--radius)',
        }}
        src={`https://www.google.com/maps/embed/v1/view?key=${GOOGLE_MAPS_KEY}&center=46.7712,23.6236&zoom=11`}
      />
      <p style={{ marginTop: 'var(--l)' }}>
        <Link to="/login">Website created for Admins and Authorities →</Link>
      </p>
    </div>
  );
}
