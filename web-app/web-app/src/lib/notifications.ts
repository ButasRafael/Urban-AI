import {getMessagingInstance, getToken, onMessage, VAPID_KEY} from './firebase';
import {notify} from './notify';
import {deleteFCMToken, updateFCMToken} from '../api/notifications';

let currentToken: string | null = null;
let messageUnsubscribe: (() => void) | null = null;
let initializingPromise: Promise<string | null> | null = null;

/**
 * Request notification permission from the user
 * @returns Promise<NotificationPermission>
 */
export async function requestNotificationPermission(): Promise<NotificationPermission> {
  if (!('Notification' in window)) {
    console.warn('This browser does not support notifications');
    return 'denied';
  }

  if (Notification.permission === 'granted') {
    return 'granted';
  }

  if (Notification.permission === 'denied') {
    return 'denied';
  }

  // Ask for permission
  return await Notification.requestPermission();
}

/**
 * Get FCM token and send it to the backend
 * @returns Promise<string | null> - The FCM token or null if failed
 */
export async function initializeNotifications(): Promise<string | null> {
  // If already initializing, return the existing promise
  if (initializingPromise) {
    console.log('Already initializing notifications, reusing existing promise');
    return initializingPromise;
  }

  // If already initialized, return current token
  if (currentToken) {
    console.log('Notifications already initialized');
    return currentToken;
  }

  // Start initialization
  initializingPromise = (async () => {
    try {
    // Check if notifications are supported
    const messaging = await getMessagingInstance();
    if (!messaging) {
      console.warn('Firebase Messaging is not supported in this browser');
      return null;
    }

    // Request permission
    const permission = await requestNotificationPermission();
    if (permission !== 'granted') {
      console.log('Notification permission not granted');
      return null;
    }

    // Wait for service worker to be ready (vite-plugin-pwa handles registration)
    let registration: ServiceWorkerRegistration | undefined;
    if ('serviceWorker' in navigator) {
      try {
        registration = await navigator.serviceWorker.ready;
        console.log('Service Worker ready:', registration);
      } catch (error) {
        console.error('Service Worker not available:', error);
      }
    }

    // Get FCM token with service worker registration
    const token = await getToken(messaging, {
      vapidKey: VAPID_KEY,
      serviceWorkerRegistration: registration,
    });

    if (token) {
      console.log('FCM Token:', token);
      currentToken = token;

      // Send token to backend
      await updateFCMToken(token);

      // Set up foreground message handler
      setupForegroundMessageHandler();

      return token;
    } else {
      console.warn('No FCM token available');
      return null;
    }
  } catch (error) {
    console.error('Error initializing notifications:', error);
    notify.error('Notification Error', 'Failed to initialize push notifications');
    return null;
  } finally {
    initializingPromise = null;
  }
  })();

  return initializingPromise;
}

/**
 * Set up handler for foreground messages (when app is open)
 */
function setupForegroundMessageHandler() {
  getMessagingInstance().then((messaging) => {
    if (!messaging) return;

    // Unsubscribe previous listener if exists
    if (messageUnsubscribe) {
      messageUnsubscribe();
    }

    // Listen for foreground messages
    messageUnsubscribe = onMessage(messaging, (payload) => {
      console.log('Foreground message received:', payload);

      // Show notification using your toast library
      const title = payload.notification?.title || 'Notification';
      const body = payload.notification?.body || '';

      notify.info(title, body);

      // Optionally play a sound or show a custom UI element
    });
  });
}

/**
 * Clean up notifications on logout
 */
export async function cleanupNotifications(): Promise<void> {
  try {
    // Delete token from backend
    if (currentToken) {
      await deleteFCMToken();
      currentToken = null;
    }

    // Unsubscribe from messages
    if (messageUnsubscribe) {
      messageUnsubscribe();
      messageUnsubscribe = null;
    }

    // Clear initializing promise
    initializingPromise = null;

    console.log('Notifications cleaned up');
  } catch (error) {
    console.error('Error cleaning up notifications:', error);
  }
}

/**
 * Check if notifications are enabled
 */
export function areNotificationsEnabled(): boolean {
  return Notification.permission === 'granted';
}

/**
 * Check if we should show the custom notification prompt
 * Returns true if permission is 'default' (not asked yet)
 */
export function shouldShowNotificationPrompt(): boolean {
  if (!('Notification' in window)) {
    return false;
  }
  return Notification.permission === 'default';
}

/**
 * Get current FCM token
 */
export function getCurrentToken(): string | null {
  return currentToken;
}