import {NavLink, Outlet, useLocation} from 'react-router-dom';
import { useAuth } from '../auth/useAuth';
import { useEffect, useMemo, useState, type ReactNode } from 'react';
import '../styles/shell.css';
import { cleanupNotifications, initializeNotifications, areNotificationsEnabled } from '../lib/notifications';
import { getNotificationPreferences } from '../api/notifications';
import { useNotifications } from '../contexts/NotificationContext';
import { logout } from '../api/auth';

type Role = 'user' | 'authority' | 'admin';

interface AuthUser {
  username?: string | null;
  email?: string | null;
  role: Role;
}

type LayoutProps = { children?: ReactNode };

type LinkItem = { to: string; label: string; icon: ReactNode };

export default function Layout({ children }: LayoutProps) {
  const { user } = useAuth() as { user: AuthUser | null };
  const location = useLocation();
  const isAuthed = !!user;
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('theme');
    if (saved === 'light' || saved === 'dark') return saved;
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });
  const { unreadCount } = useNotifications();

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    localStorage.setItem('theme', theme);
  }, [theme]);
  const toggleTheme = () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'));

  const role: Role = user?.role ?? 'user';

  const links: LinkItem[] = useMemo(() => {
    const base: LinkItem[] = [
      { to: '/map',  label: 'Map',  icon: <IconMap /> },
      { to: '/list', label: 'List', icon: <IconList /> },
    ];
    if (!isAuthed) return base;
    if (role === 'admin' || role === 'authority') {
      base.push({ to: '/issues', label: 'Issues', icon: <IconIssues /> });
    }
    if (role === 'admin') {
      base.unshift({ to: '/analytics', label: 'Analytics', icon: <IconChart /> });
    } else {
      base.push({ to: '/chat', label: 'Chat', icon: <IconChat /> });
    }
    // Add notifications for all authenticated users
    base.push({ to: '/notifications', label: 'Notifications', icon: <IconBell /> });
    // Add settings for all authenticated users
    base.push({ to: '/settings/notifications', label: 'Settings', icon: <IconSettings /> });
    return base;
  }, [isAuthed, role]);

  const [collapsed, setCollapsed] = useState<boolean>(() => localStorage.getItem('sidebarCollapsed') === '1');
  const [mobileOpen, setMobileOpen] = useState<boolean>(false);
  useEffect(() => { localStorage.setItem('sidebarCollapsed', collapsed ? '1' : '0'); }, [collapsed]);
  useEffect(() => { setMobileOpen(false); }, [location.pathname]);

  // Initialize push notifications on app load if user is logged in and has them enabled
  useEffect(() => {
    if (!isAuthed) return;

    async function setupNotifications() {
      try {
        // Check if browser permission is granted
        if (!areNotificationsEnabled()) return;

        // Get user preferences
        const prefs = await getNotificationPreferences();

        // If push is enabled in preferences, initialize FCM
        if (prefs.push_enabled) {
          await initializeNotifications();
          console.log('Push notifications re-initialized');
        }
      } catch (error) {
        console.error('Failed to initialize notifications:', error);
      }
    }

    setupNotifications();
  }, [isAuthed]);

  const isMap = location.pathname.startsWith('/map');
  const displayName = getDisplayName(user);
  const initials = getInitials(displayName);

  return (
      <div className="app-shell">
        {/* SIDEBAR (render only when authenticated) */}
        {isAuthed && (
            <aside
                className="sidebar"
                data-collapsed={collapsed || undefined}
                data-open={mobileOpen || undefined}
                aria-label="Primary"
            >
              {/* Brand / collapse control */}
              <div className="brand">
                <div className="logo" aria-hidden>UA</div>
                {!collapsed && <span className="brand-text">Urban-AI</span>}
                <button
                    className="icon-btn collapse-btn"
                    aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                    onClick={() => setCollapsed(v => !v)}
                    title={collapsed ? 'Expand' : 'Collapse'}
                >
                  {collapsed ? <IconChevronRight/> : <IconChevronLeft/>}
                </button>
              </div>

              {/* Nav */}
              <nav className="nav">
                {links.map(({to, label, icon}) => (
                    <NavLink
                        key={to}
                        to={to}
                        className={({isActive}) => 'nav-item' + (isActive ? ' active' : '')}
                        title={collapsed ? label : undefined}
                    >
                      <span className="nav-icon">{icon}</span>
                      {!collapsed && <span className="nav-label">{label}</span>}
                      {to === '/notifications' && unreadCount > 0 && (
                        <span className="notification-badge">{unreadCount > 99 ? '99+' : unreadCount}</span>
                      )}
                      <span className="active-pill" aria-hidden/>
                    </NavLink>
                ))}
              </nav>

              {/* User + actions */}
              <div className="sidebar-footer">
                <div className="user-card" title={displayName}>
                  <div className="avatar" aria-hidden>{initials}</div>
                  {!collapsed && (
                      <div className="user-meta">
                        <strong>{displayName}</strong>
                        <small className="muted">{role}</small>
                      </div>
                  )}
                </div>

                {/* Theme switch lives in the sidebar */}
                <button
                    type="button"
                    className="icon-btn theme-btn"
                    onClick={toggleTheme}
                    title={theme === 'dark' ? 'Switch to light' : 'Switch to dark'}
                    aria-label="Toggle theme"
                >
                  {theme === 'dark' ? <IconSun/> : <IconMoon/>}
                  {!collapsed && <span className="theme-text">{theme === 'dark' ? 'Light mode' : 'Dark mode'}</span>}
                </button>

                <button
                    className="btn-logout"
                    onClick={async () => {
                      try {
                        // Clean up notifications before logging out
                        await cleanupNotifications();
                        // Call backend logout endpoint
                        await logout();
                        // Redirect to login
                        window.location.href = '/login';
                      } catch (error) {
                        console.error('Logout failed:', error);
                        // Fallback: clear localStorage and redirect anyway
                        localStorage.clear();
                        window.location.href = '/login';
                      }
                    }}
                >
                  <IconLogout/>
                  {!collapsed && <span>Logout</span>}
                </button>
              </div>
            </aside>
        )}

        {/* MOBILE: the toggle (hidden on desktop) */}
        {isAuthed && (
            <>
              <button
                  className="mobile-toggle"
                  aria-label="Open menu"
                  onClick={() => setMobileOpen(true)}
              >
                <IconMenu/>
              </button>
              {mobileOpen && <div className="overlay" onClick={() => setMobileOpen(false)} aria-hidden/>}
            </>
        )}

        {/* CONTENT */}
        <main className={`content ${isMap ? 'content--full' : ''}`}>
          {children ?? <Outlet/>}
        </main>
      </div>
  );
}

function IconChart() {
  return (<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 3v18h18"/><rect x="7" y="10" width="3" height="7" rx="1"/><rect x="12" y="6" width="3" height="11" rx="1"/><rect x="17" y="13" width="3" height="4" rx="1"/></svg>); }
function IconMap()   { return (<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 18l-6 3V6l6-3 6 3 6-3v15l-6 3-6-3z"/><path d="M9 3v15"/><path d="M15 6v15"/></svg>); }
function IconList()  { return (<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><path d="M8 6h13M8 12h13M8 18h13"/><circle cx="4" cy="6" r="1.5"/><circle cx="4" cy="12" r="1.5"/><circle cx="4" cy="18" r="1.5"/></svg>); }
function IconChat()  { return (<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a4 4 0 0 1-4 4H8l-5 3V6a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/></svg>); }
function IconBell()  { return (<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>); }
function IconIssues() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M9 12h6M9 16h6M9 8h6" />
      <path d="M4 6h16v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z" />
    </svg>
  );
}
function IconSettings() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="4" y1="21" x2="4" y2="14" />
      <line x1="4" y1="10" x2="4" y2="3" />
      <line x1="12" y1="21" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12" y2="3" />
      <line x1="20" y1="21" x2="20" y2="16" />
      <line x1="20" y1="12" x2="20" y2="3" />
      <line x1="1" y1="14" x2="7" y2="14" />
      <line x1="9" y1="8" x2="15" y2="8" />
      <line x1="17" y1="16" x2="23" y2="16" />
    </svg>
  );
}
function IconLogout(){ return (<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path d="M16 17l5-5-5-5"/><path d="M21 12H9"/></svg>); }
function IconChevronLeft()  { return (<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><path d="M15 18l-6-6 6-6"/></svg>); }
function IconChevronRight() { return (<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 6l6 6-6 6"/></svg>); }
function IconMenu()  { return (<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 6h18M3 12h18M3 18h18"/></svg>); }
function IconMoon()  { return (<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>); }
function IconSun()   { return (<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>); }

function getInitials(name?: string): string {
  const safe = (name ?? '').trim();
  if (!safe) return 'U';
  const parts = safe.split(/[\s._-]+/).filter(Boolean);
  const a = parts[0]?.[0] ?? '';
  const b = (parts.length > 1 ? parts[parts.length - 1]?.[0] : parts[0]?.[1]) ?? '';
  return (a + b).toUpperCase() || 'U';
}

function getDisplayName(u: AuthUser | null | undefined): string {
  if (!u) return 'User';
  return (u.username && u.username.trim()) || (u.email && u.email.trim()) || 'User';
}
