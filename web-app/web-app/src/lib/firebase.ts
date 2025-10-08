import { initializeApp } from 'firebase/app';
import { getMessaging, getToken, onMessage, isSupported } from 'firebase/messaging';
import { firebaseConfig, VAPID_KEY } from './firebase-config';

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Cloud Messaging
let messaging: ReturnType<typeof getMessaging> | null = null;

// Check if messaging is supported in this browser
const initMessaging = async () => {
  const supported = await isSupported();
  if (supported) {
    messaging = getMessaging(app);
  } else {
    console.warn('Firebase Messaging is not supported in this browser');
  }
  return messaging;
};

// Export messaging getter
export const getMessagingInstance = async () => {
  if (!messaging) {
    await initMessaging();
  }
  return messaging;
};

export { getToken, onMessage, VAPID_KEY };
