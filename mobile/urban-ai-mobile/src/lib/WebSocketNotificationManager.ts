import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE } from '../config';

export interface NotificationMessage {
  type: 'notification';
  event_type: string;
  message: string;
  notification_id: number;
}

type MessageCallback = (message: NotificationMessage) => void;

/**
 * Singleton WebSocket manager for notifications
 * Ensures only one WebSocket connection exists per app instance
 */
export class WebSocketNotificationManager {
  private static instance: WebSocketNotificationManager | null = null;
  private ws: WebSocket | null = null;
  private listeners: Set<MessageCallback> = new Set();
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private pingInterval: NodeJS.Timeout | null = null;
  private token: string | null = null;
  private shouldBeConnected = false;

  // Private constructor to prevent direct instantiation
  private constructor() {}

  /**
   * Get the singleton instance
   */
  static getInstance(): WebSocketNotificationManager {
    if (!WebSocketNotificationManager.instance) {
      WebSocketNotificationManager.instance = new WebSocketNotificationManager();
    }
    return WebSocketNotificationManager.instance;
  }

  /**
   * Subscribe to notification messages
   * Automatically connects if this is the first subscriber
   */
  subscribe(callback: MessageCallback): void {
    this.listeners.add(callback);
    console.log(`WebSocket subscriber added (${this.listeners.size} total)`);

    // Clean up dead/closing connections before checking
    if (this.ws && (this.ws.readyState === WebSocket.CLOSING || this.ws.readyState === WebSocket.CLOSED)) {
      console.log('Cleaning up dead WebSocket connection');
      this.ws = null;
    }

    // Connect if we have subscribers and not already connected
    if (this.listeners.size > 0 && !this.ws) {
      this.connect();
    }
  }

  /**
   * Unsubscribe from notification messages
   * Automatically disconnects if this was the last subscriber
   */
  unsubscribe(callback: MessageCallback): void {
    this.listeners.delete(callback);
    console.log(`WebSocket subscriber removed (${this.listeners.size} remaining)`);

    // Disconnect if no more subscribers
    if (this.listeners.size === 0) {
      this.shouldBeConnected = false;
      this.disconnect();
    }
  }

  /**
   * Manually connect the WebSocket
   */
  private async connect(): Promise<void> {
    // Don't connect if already connected or connecting
    if (this.ws?.readyState === WebSocket.OPEN || this.ws?.readyState === WebSocket.CONNECTING) {
      return;
    }

    // Get token from AsyncStorage
    this.token = await AsyncStorage.getItem('accessToken');
    if (!this.token) {
      console.warn('No access token found, cannot connect WebSocket');
      return;
    }

    this.shouldBeConnected = true;

    // Clear any existing reconnect timeout
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    // Create WebSocket URL
    const wsUrl = `${API_BASE.replace(/^http/i, 'ws')}/ws/inference?token=${encodeURIComponent(this.token)}`;

    console.log('WebSocket connecting...');
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log('✅ WebSocket connected');
      this.reconnectAttempts = 0;

      // Start ping interval to keep connection alive
      this.startPingInterval();
    };

    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);

        if (message.type === 'notification') {
          // Broadcast to all subscribers
          this.listeners.forEach(callback => {
            try {
              callback(message as NotificationMessage);
            } catch (error) {
              console.error('Error in notification callback:', error);
            }
          });
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = (event) => {
      console.log('WebSocket disconnected', { code: event.code, reason: event.reason });
      this.ws = null;
      this.stopPingInterval();

      // Only reconnect if we should be connected and didn't receive a clean close
      if (this.shouldBeConnected && event.code !== 1000) {
        this.scheduleReconnect();
      }
    };
  }

  /**
   * Manually disconnect the WebSocket
   */
  private disconnect(): void {
    console.log('WebSocket disconnecting...');
    this.shouldBeConnected = false;

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    this.stopPingInterval();

    if (this.ws) {
      const currentState = this.ws.readyState;

      // Close the connection (works for both OPEN and CONNECTING states)
      if (currentState === WebSocket.OPEN || currentState === WebSocket.CONNECTING) {
        this.ws.close(1000, 'No more subscribers');
      }

      // Don't set this.ws = null here - let onclose handler do it
      // This prevents race conditions with new subscriptions
    }
  }

  /**
   * Schedule a reconnect with exponential backoff
   */
  private scheduleReconnect(): void {
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    this.reconnectAttempts++;

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    this.reconnectTimeout = setTimeout(() => {
      if (this.shouldBeConnected) {
        this.connect();
      }
    }, delay);
  }

  /**
   * Start sending ping messages to keep connection alive
   */
  private startPingInterval(): void {
    this.stopPingInterval();

    this.pingInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000); // Ping every 30 seconds
  }

  /**
   * Stop sending ping messages
   */
  private stopPingInterval(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  /**
   * Check if WebSocket is currently connected
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Get the number of active subscribers
   */
  getSubscriberCount(): number {
    return this.listeners.size;
  }
}