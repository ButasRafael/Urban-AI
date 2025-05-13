import { NavLink } from 'react-router-dom';
import { useAuth } from '../auth/useAuth';
//import Navbar from './Navbar';

export default function Layout({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();

  if (!user) return <>{children}</>;

  const links =
    user.role === 'admin'
      ? [
          { to: '/analytics', label: 'Analytics' },
          { to: '/map', label: 'Map' },
          { to: '/list', label: 'List' },
        ]
      : [
          { to: '/map', label: 'Map' },
          { to: '/list', label: 'List' },
        ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* <Navbar /> */}
      <div className="app-shell">
        <nav className="sidebar">
          <h3 style={{ 
            marginBottom: 'var(--l)',
            color: 'white',
            fontSize: '1.25rem'
          }}>
            Urban-AI
          </h3>
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              className={({ isActive }) => 
                isActive ? 'active' : undefined
              }
            >
              {l.label}
            </NavLink>
          ))}

          
          <button
            onClick={() => {
              localStorage.clear();
              window.location.href = '/login';
            }}
            style={{
              marginTop: 'auto',
              display: 'flex',
              alignItems: 'center',
              gap: 'var(--s)',
              padding: 'var(--m)',
              borderRadius: 'var(--radius-sm)',
              border: 'none',
              background: 'var(--primary-500)',
              color: 'white',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              fontWeight: 500,
              fontSize: '14px',
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = 'var(--primary-700)';
              e.currentTarget.style.transform = 'translateY(-1px)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = 'var(--primary-500)';
              e.currentTarget.style.transform = 'translateY(0)';
            }}
          >
            <svg 
              width="16" 
              height="16" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2" 
              strokeLinecap="round" 
              strokeLinejoin="round"
            >
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
              <polyline points="16 17 21 12 16 7"></polyline>
              <line x1="21" y1="12" x2="9" y2="12"></line>
            </svg>
            Logout
          </button>
        </nav>

        <main className="content">{children}</main>
      </div>
    </div>
  );
}
