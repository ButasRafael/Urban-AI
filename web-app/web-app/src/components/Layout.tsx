// src/components/Layout.tsx
import { NavLink, useLocation } from 'react-router-dom';
import { useAuth } from '../auth/useAuth';
import { useEffect, useMemo, useState, type ReactNode } from 'react';
import '../styles/shell.css';

type LinkItem = { to: string; label: string; icon: ReactNode };

export default function Layout({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  const location = useLocation();

  if (!user) return <>{children}</>;

  // THEME: persisted light/dark
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('theme');
    if (saved === 'light' || saved === 'dark') return saved;
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    localStorage.setItem('theme', theme);
  }, [theme]);
  const toggleTheme = () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'));

  const links: LinkItem[] = useMemo(() => {
    const base: LinkItem[] = [
      { to: '/map',  label: 'Map',  icon: <IconMap /> },
      { to: '/list', label: 'List', icon: <IconList /> },
    ];
    if (user.role !== 'admin') {
      base.push({ to: '/chat', label: 'Chat', icon: <IconChat /> });
    } else {
      base.unshift({ to: '/analytics', label: 'Analytics', icon: <IconChart /> });
    }
    return base;
  }, [user.role]);

  const [collapsed, setCollapsed] = useState<boolean>(() => localStorage.getItem('sidebarCollapsed') === '1');
  const [mobileOpen, setMobileOpen] = useState<boolean>(false);
  useEffect(() => { localStorage.setItem('sidebarCollapsed', collapsed ? '1' : '0'); }, [collapsed]);
  useEffect(() => { setMobileOpen(false); }, [location.pathname]);

  const displayName = user?.username || (user as any)?.email || 'User';
  const initials = getInitials(displayName);

  return (
    <div className="app-shell">
      {/* SIDEBAR */}
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
            {collapsed ? <IconChevronRight /> : <IconChevronLeft />}
          </button>
        </div>

        {/* Nav */}
        <nav className="nav">
          {links.map(({ to, label, icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) => 'nav-item' + (isActive ? ' active' : '')}
              title={collapsed ? label : undefined}
            >
              <span className="nav-icon">{icon}</span>
              {!collapsed && <span className="nav-label">{label}</span>}
              <span className="active-pill" aria-hidden />
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
                <small className="muted">{user.role}</small>
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
            {theme === 'dark' ? <IconSun /> : <IconMoon />}
            {!collapsed && <span className="theme-text">{theme === 'dark' ? 'Light mode' : 'Dark mode'}</span>}
          </button>

          <button
            className="btn-logout"
            onClick={() => {
              localStorage.clear();
              window.location.href = '/login';
            }}
          >
            <IconLogout />
            {!collapsed && <span>Logout</span>}
          </button>
        </div>
      </aside>

      {/* MOBILE: the toggle (hidden on desktop) */}
      <button
        className="mobile-toggle"
        aria-label="Open menu"
        onClick={() => setMobileOpen(true)}
      >
        <IconMenu />
      </button>
      {mobileOpen && <div className="overlay" onClick={() => setMobileOpen(false)} aria-hidden />}

      {/* CONTENT */}
      <main className="content">{children}</main>
    </div>
  );
}

/* ---------------- Icons ---------------- */
function IconChart() { return (<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 3v18h18"/><rect x="7" y="10" width="3" height="7" rx="1"/><rect x="12" y="6" width="3" height="11" rx="1"/><rect x="17" y="13" width="3" height="4" rx="1"/></svg>); }
function IconMap()   { return (<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 18l-6 3V6l6-3 6 3 6-3v15l-6 3-6-3z"/><path d="M9 3v15"/><path d="M15 6v15"/></svg>); }
function IconList()  { return (<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><path d="M8 6h13M8 12h13M8 18h13"/><circle cx="4" cy="6" r="1.5"/><circle cx="4" cy="12" r="1.5"/><circle cx="4" cy="18" r="1.5"/></svg>); }
function IconChat()  { return (<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a4 4 0 0 1-4 4H8l-5 3V6a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/></svg>); }
function IconLogout(){ return (<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path d="M16 17l5-5-5-5"/><path d="M21 12H9"/></svg>); }
function IconChevronLeft()  { return (<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><path d="M15 18l-6-6 6-6"/></svg>); }
function IconChevronRight() { return (<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 6l6 6-6 6"/></svg>); }
function IconMenu()  { return (<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 6h18M3 12h18M3 18h18"/></svg>); }
function IconMoon()  { return (<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>); }
function IconSun()   { return (<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>); }

/* ---------------- Helper ---------------- */
function getInitials(name?: string): string {
  const safe = (name ?? '').trim();
  if (!safe) return 'U';
  const parts = safe.split(/[\s._-]+/).filter(Boolean);
  const a = parts[0]?.[0] ?? '';
  const b = (parts.length > 1 ? parts[parts.length - 1]?.[0] : parts[0]?.[1]) ?? '';
  return (a + b).toUpperCase() || 'U';
}
