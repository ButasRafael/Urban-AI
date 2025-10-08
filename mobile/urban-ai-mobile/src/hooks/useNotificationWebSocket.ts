import { useEffect, useRef, useState } from 'react';
import { notify } from '../components/ui/Toast';
import { WebSocketNotificationManager, type NotificationMessage } from '../lib/WebSocketNotificationManager';

interface UseNotificationWebSocketOptions {
  onNotification?: (message: NotificationMessage) => void;
  enabled?: boolean;
}

/**
 * Hook to subscribe to WebSocket notifications
 * Uses a singleton manager to ensure only one connection per app
 */
export function useNotificationWebSocket({
  onNotification,
  enabled = true
}: UseNotificationWebSocketOptions = {}) {
  const [isConnected, setIsConnected] = useState(false);
  const onNotificationRef = useRef(onNotification);

  // Keep the callback ref up to date
  useEffect(() => {
    onNotificationRef.current = onNotification;
  }, [onNotification]);

  useEffect(() => {
    if (!enabled || !onNotification) return;

    // Get the singleton manager
    const manager = WebSocketNotificationManager.getInstance();

    // Create a stable callback wrapper
    const callback = (message: NotificationMessage) => {
      // Call the parent callback
      onNotificationRef.current?.(message);

      // Show toast notification for mobile
      const eventName = message.event_type.replace(/_/g, ' ');
      notify.info(eventName, message.message);
    };

    // Subscribe to notifications
    console.log('Subscribing to WebSocket notifications');
    manager.subscribe(callback);

    // Update connection status periodically
    const checkConnection = setInterval(() => {
      setIsConnected(manager.isConnected());
    }, 1000);

    // Cleanup: unsubscribe on unmount
    return () => {
      console.log('Unsubscribing from WebSocket notifications');
      manager.unsubscribe(callback);
      clearInterval(checkConnection);
    };
  }, [enabled, onNotification]);

  return {
    isConnected,
  };
}
