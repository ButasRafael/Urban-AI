import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { getNotifications, markNotificationAsRead, markAllNotificationsAsRead, deleteNotification, type StoredNotification } from '../api/notifications';
import { notify } from '../lib/notify';
import { useNotifications } from '../contexts/NotificationContext';
import '../styles/notifications.css';

interface NotificationGroup {
  label: string;
  notifications: StoredNotification[];
}

export default function Notifications() {
  const [notifications, setNotifications] = useState<StoredNotification[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [filter, setFilter] = useState<'all' | 'unread'>('all');
  const [hasMore, setHasMore] = useState(true);
  const offsetRef = useRef(0); // Use ref to avoid stale closure
  const { notificationReceived, refreshUnreadCount } = useNotifications();

  const loadNotifications = useCallback(async (append = false) => {
    try {
      if (append) {
        setLoadingMore(true);
      } else {
        setLoading(true);
        offsetRef.current = 0;
      }

      const currentOffset = append ? offsetRef.current : 0;
      const data = await getNotifications(50, currentOffset, filter === 'unread');

      setNotifications(prev => {
        if (append) {
          // De-duplicate by ID to prevent duplicates if pagination somehow overlaps
          const existingIds = new Set(prev.map(n => n.id));
          const newItems = data.filter(n => !existingIds.has(n.id));
          return [...prev, ...newItems];
        }
        return data;
      });
      setHasMore(data.length === 50);

      if (append) {
        offsetRef.current += data.length; // Increment by actual items received
      } else {
        offsetRef.current = data.length;
      }
    } catch {
      notify.error('Failed to load notifications');
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [filter]); // No need for offset in deps since using ref

  useEffect(() => {
    loadNotifications();
  }, [filter, loadNotifications]);

  // Reload notifications when a new one arrives via WebSocket
  useEffect(() => {
    if (notificationReceived > 0) {
      loadNotifications();
    }
  }, [notificationReceived, loadNotifications]);

  async function handleMarkAsRead(id: number) {
    try {
      await markNotificationAsRead(id);
      setNotifications(prev =>
        prev.map(n => n.id === id ? { ...n, read_at: new Date().toISOString() } : n)
      );
      // Refresh unread count in context
      refreshUnreadCount();
    } catch {
      notify.error('Failed to mark as read');
    }
  }

  async function handleMarkAllAsRead() {
    try {
      await markAllNotificationsAsRead();
      setNotifications(prev =>
        prev.map(n => ({ ...n, read_at: new Date().toISOString() }))
      );
      // Refresh unread count in context
      refreshUnreadCount();
      notify.success('All notifications marked as read');
    } catch {
      notify.error('Failed to mark all as read');
    }
  }

  async function handleDelete(id: number) {
    try {
      await deleteNotification(id);
      setNotifications(prev => prev.filter(n => n.id !== id));
      // Refresh unread count in context
      refreshUnreadCount();
      notify.success('Notification deleted');
    } catch {
      notify.error('Failed to delete notification');
    }
  }

  const unreadCount = notifications.filter(n => !n.read_at).length;

  // Group notifications by date
  const groupedNotifications = useMemo(() => {
    return groupNotificationsByDate(notifications);
  }, [notifications]);

  return (
    <div className="notifications-page">
      {/* Header */}
      <header className="notifications__header">
        <div className="title-stack">
          <div className="eyebrow">Inbox</div>
          <h1 className="grad-title">Notifications</h1>
          {unreadCount > 0 && (
            <div className="badge">{unreadCount} unread</div>
          )}
        </div>
        <div className="header-actions">
          <div className="segmented">
            <button
              className={filter === 'all' ? 'active' : ''}
              onClick={() => setFilter('all')}
            >
              All
            </button>
            <button
              className={filter === 'unread' ? 'active' : ''}
              onClick={() => setFilter('unread')}
            >
              Unread
            </button>
          </div>
          {unreadCount > 0 && (
            <button className="btn" onClick={handleMarkAllAsRead}>
              <IconCheckAll />
              Mark all read
            </button>
          )}
        </div>
      </header>

      {/* Content */}
      <div className="notifications-content">
        {loading ? (
          <div className="notifications-skeleton">
            {[1, 2, 3, 4, 5].map(i => (
              <div key={i} className="notification-skeleton" />
            ))}
          </div>
        ) : notifications.length === 0 ? (
          <div className="empty-state card">
            <div className="empty-icon">
              <IconBell />
            </div>
            <div>
              <h3>No notifications</h3>
              <p className="muted">
                {filter === 'unread' ? "You're all caught up!" : 'Notifications will appear here'}
              </p>
            </div>
          </div>
        ) : (
          <>
            <div className="notifications-list">
              {groupedNotifications.map((group) => (
                <div key={group.label}>
                  <div className="date-group-header">
                    <span className="date-group-label">{group.label}</span>
                  </div>
                  {group.notifications.map(notification => (
                    <NotificationCard
                      key={notification.id}
                      notification={notification}
                      onMarkAsRead={handleMarkAsRead}
                      onDelete={handleDelete}
                    />
                  ))}
                </div>
              ))}
            </div>

            {/* Load More Button */}
            {hasMore && (
              <div style={{ display: 'flex', justifyContent: 'center', padding: '2rem' }}>
                <button
                  className="btn btn-secondary"
                  onClick={() => loadNotifications(true)}
                  disabled={loadingMore}
                >
                  {loadingMore ? (
                    <>
                      <div className="spinner-small" style={{ marginRight: '8px' }} />
                      Loading...
                    </>
                  ) : (
                    <>
                      <IconRefresh />
                      Load More
                    </>
                  )}
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function NotificationCard({
  notification,
  onMarkAsRead,
  onDelete,
}: {
  notification: StoredNotification;
  onMarkAsRead: (id: number) => void;
  onDelete: (id: number) => void;
}) {
  const isUnread = !notification.read_at;
  const eventType = notification.event_type.replace(/_/g, ' ');
  const timestamp = formatTimestamp(notification.created_at);

  // Get message from event data
  const message = getNotificationMessage(notification);
  const icon = getNotificationIcon(notification.event_type);

  return (
    <div className={`notification-card card ${isUnread ? 'unread' : ''}`}>
      <div className="notification-icon">{icon}</div>
      <div className="notification-content">
        <div className="notification-header">
          <span className="notification-type">{eventType}</span>
          <span className="notification-time">{timestamp}</span>
        </div>
        <div className="notification-message">{message}</div>
        {notification.channel === 'email' && (
          <div className="notification-meta">
            <IconMail />
            <span>Email sent</span>
          </div>
        )}
        {notification.channel === 'push' && (
          <div className="notification-meta">
            <IconBell />
            <span>Push notification</span>
          </div>
        )}
      </div>
      <div className="notification-actions">
        {isUnread && (
          <button
            className="btn-icon"
            onClick={() => onMarkAsRead(notification.id)}
            title="Mark as read"
          >
            <IconCheck />
          </button>
        )}
        <button
          className="btn-icon btn-danger"
          onClick={() => onDelete(notification.id)}
          title="Delete"
        >
          <IconTrash />
        </button>
      </div>
    </div>
  );
}

function getNotificationMessage(notification: StoredNotification): string {
  const data = notification.event_data;
  const type = notification.event_type;

  switch (type) {
    case 'media_created':
      return `New media uploaded with ${data.num_detections} detection(s)`;
    case 'issue_verified':
      return `Your ${data.class_name} report has been verified`;
    case 'issue_assigned':
      return `${data.class_name} issue has been assigned`;
    case 'issue_status_changed':
      return `${data.class_name} status changed: ${data.old_status} → ${data.new_status}`;
    case 'issue_severity_changed':
      return `${data.class_name} severity changed: ${data.old_severity} → ${data.new_severity}`;
    default:
      return `Notification: ${type}`;
  }
}

function getNotificationIcon(eventType: string) {
  switch (eventType) {
    case 'media_created':
      return <IconUpload />;
    case 'issue_verified':
      return <IconCheckCircle />;
    case 'issue_assigned':
      return <IconUser />;
    case 'issue_status_changed':
      return <IconActivity />;
    case 'issue_severity_changed':
      return <IconAlert />;
    default:
      return <IconBell />;
  }
}

function formatTimestamp(iso: string): string {
  const date = new Date(iso);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

function groupNotificationsByDate(notifications: StoredNotification[]): NotificationGroup[] {
  const groups: Record<string, StoredNotification[]> = {};
  const now = new Date();

  notifications.forEach(notification => {
    const date = new Date(notification.created_at);
    const key = getDateGroupKey(date, now);

    if (!groups[key]) {
      groups[key] = [];
    }
    groups[key].push(notification);
  });

  // Convert to array and maintain chronological order
  const groupOrder = ['Today', 'Yesterday', 'This Week', 'This Month'];
  const result: NotificationGroup[] = [];

  groupOrder.forEach(key => {
    if (groups[key]) {
      result.push({ label: key, notifications: groups[key] });
      delete groups[key];
    }
  });

  // Add remaining groups (older months) sorted by most recent first
  Object.entries(groups)
    .sort(([, a], [, b]) => {
      const dateA = new Date(a[0].created_at);
      const dateB = new Date(b[0].created_at);
      return dateB.getTime() - dateA.getTime();
    })
    .forEach(([label, notifications]) => {
      result.push({ label, notifications });
    });

  return result;
}

function getDateGroupKey(date: Date, now: Date): string {
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startOfYesterday = new Date(startOfToday);
  startOfYesterday.setDate(startOfYesterday.getDate() - 1);
  const startOfThisWeek = new Date(startOfToday);
  startOfThisWeek.setDate(startOfThisWeek.getDate() - 7);
  const startOfThisMonth = new Date(startOfToday);
  startOfThisMonth.setDate(startOfThisMonth.getDate() - 30);

  const notificationDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());

  if (notificationDate.getTime() >= startOfToday.getTime()) {
    return 'Today';
  } else if (notificationDate.getTime() >= startOfYesterday.getTime()) {
    return 'Yesterday';
  } else if (notificationDate.getTime() >= startOfThisWeek.getTime()) {
    return 'This Week';
  } else if (notificationDate.getTime() >= startOfThisMonth.getTime()) {
    return 'This Month';
  } else {
    return date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
  }
}

// Icons
function IconBell() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
      <path d="M13.73 21a2 2 0 0 1-3.46 0" />
    </svg>
  );
}

function IconCheckAll() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M2 12l5 5L22 2" />
      <path d="M7 12l5 5" />
    </svg>
  );
}

function IconCheck() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M20 6L9 17l-5-5" />
    </svg>
  );
}

function IconTrash() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6h14z" />
    </svg>
  );
}

function IconMail() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="2" y="4" width="20" height="16" rx="2" />
      <path d="m2 7 10 6 10-6" />
    </svg>
  );
}

function IconUpload() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" />
    </svg>
  );
}

function IconCheckCircle() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  );
}

function IconUser() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  );
}

function IconActivity() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
    </svg>
  );
}

function IconAlert() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3z" />
      <path d="M12 9v4M12 17h.01" />
    </svg>
  );
}

function IconRefresh() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2" />
    </svg>
  );
}
