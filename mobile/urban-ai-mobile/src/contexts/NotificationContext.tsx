import React, { createContext, useContext, useState, useEffect, useCallback, useRef, type ReactNode } from 'react';
import { useNotificationWebSocket } from '../hooks/useNotificationWebSocket';
import { getUnreadCount } from '../api/notifications';

interface NotificationContextValue {
  unreadCount: number;
  refreshUnreadCount: () => Promise<void>;
  notificationReceived: number; // Timestamp of last notification
}

const NotificationContext = createContext<NotificationContextValue | null>(null);

export function NotificationProvider({ children, enabled }: { children: ReactNode; enabled: boolean }) {
  const [unreadCount, setUnreadCount] = useState<number>(0);
  const [notificationReceived, setNotificationReceived] = useState<number>(0);
  const seenNotificationIdsRef = useRef<Set<number>>(new Set()); // Prevent duplicate notifications

  // Fetch initial unread count
  const refreshUnreadCount = useCallback(async () => {
    if (!enabled) return;
    try {
      const count = await getUnreadCount();
      setUnreadCount(count);
    } catch (error) {
      console.error('Failed to fetch unread count:', error);
    }
  }, [enabled]);

  // Load unread count on mount
  useEffect(() => {
    refreshUnreadCount();
  }, [refreshUnreadCount]);

  // Handle new notifications from WebSocket
  const handleNewNotification = useCallback((message?: { notification_id?: number }) => {
    const notificationId = message?.notification_id;

    // Deduplicate by notification_id to prevent double-counting
    if (notificationId && seenNotificationIdsRef.current.has(notificationId)) {
      console.log(`Ignoring duplicate notification ID: ${notificationId}`);
      return;
    }

    // Mark this notification as seen
    if (notificationId) {
      seenNotificationIdsRef.current.add(notificationId);

      // Clean up old IDs after 5 minutes to prevent memory leak
      setTimeout(() => {
        seenNotificationIdsRef.current.delete(notificationId);
      }, 5 * 60 * 1000);
    }

    // Increment unread count
    setUnreadCount(prev => prev + 1);
    // Signal that a new notification was received
    setNotificationReceived(Date.now());
  }, []);

  // Connect to WebSocket for real-time notification updates
  useNotificationWebSocket({
    onNotification: handleNewNotification,
    enabled,
  });

  return (
    <NotificationContext.Provider value={{ unreadCount, refreshUnreadCount, notificationReceived }}>
      {children}
    </NotificationContext.Provider>
  );
}

export function useNotifications() {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within NotificationProvider');
  }
  return context;
}
