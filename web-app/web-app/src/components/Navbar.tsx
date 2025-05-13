import { Link } from 'react-router-dom';
import { useAuth } from '../auth/useAuth';
import Button from './Button';

export default function Navbar() {
  const { user } = useAuth();

  return (
    <nav style={{
      background: 'var(--secondary-500)',
      color: '#fff',
      padding: 'var(--m) var(--xl)',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      boxShadow: 'var(--shadow-sm)',
      position: 'sticky',
      top: 0,
      zIndex: 100
    }}>
      <div style={{ display: 'flex', alignItems: 'center' }}>
        <Link to="/" style={{ 
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--s)',
          fontWeight: 600,
          fontSize: '1.25rem'
        }}>
          Urban AI
        </Link>
      </div>

      {user && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--m)' }}>
          <span style={{ opacity: 0.9 }}>{user.username}</span>
          {user.role === 'admin' && (
            <Link to="/analytics" style={{ 
              color: 'white',
              padding: 'var(--s) var(--m)',
              borderRadius: 'var(--radius-sm)',
              transition: 'background-color 0.2s'
            }}>
              Analytics
            </Link>
          )}
          <Link to="/map" style={{ 
            color: 'white',
            padding: 'var(--s) var(--m)',
            borderRadius: 'var(--radius-sm)',
            transition: 'background-color 0.2s'
          }}>
            Map
          </Link>
          <Button 
            variant="ghost" 
            onClick={() => (window.location.href = '/login')}
            style={{ color: 'white', borderColor: 'white' }}
          >
            Logout
          </Button>
        </div>
      )}
    </nav>
  );
}
