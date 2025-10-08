import client from './client';

export interface NotificationPreferences {
  email_enabled: boolean;
  push_enabled: boolean;
  event_preferences: {
    [key: string]: {
      email: boolean;
      push: boolean;
    };
  };
  min_severity: 'low' | 'medium' | 'high';
  quiet_hours: {
    enabled: boolean;
    start?: string;
    end?: string;
    timezone?: string;
  };
}

export interface StoredNotification {
  id: number;
  event_type: string;
  channel: string;
  status: string;
  event_data: Record<string, unknown>;
  sent_at: string | null;
  read_at: string | null;
  created_at: string;
}

export interface UnreadCountResponse {
  unread_count: number;
}

/**
 * Update FCM token on the backend
 */
export async function updateFCMToken(fcmToken: string): Promise<void> {
  await client.post('/notifications/fcm-token', { fcm_token: fcmToken });
}

/**
 * Delete FCM token from the backend
 */
export async function deleteFCMToken(): Promise<void> {
  await client.delete('/notifications/fcm-token');
}

/**
 * Get user notification preferences
 */
export async function getNotificationPreferences(): Promise<NotificationPreferences> {
  const { data } = await client.get<NotificationPreferences>('/notifications/preferences');
  return data;
}

/**
 * Update user notification preferences
 */
export async function updateNotificationPreferences(
  preferences: Partial<NotificationPreferences>
): Promise<NotificationPreferences> {
  const { data } = await client.patch<NotificationPreferences>('/notifications/preferences', preferences);
  return data;
}

// Notification Store API

/**
 * Get user's stored notifications
 */
export async function getNotifications(limit = 50, offset = 0, unreadOnly = false): Promise<StoredNotification[]> {
  const { data } = await client.get<StoredNotification[]>('/notifications/store', {
    params: { limit, offset, unread_only: unreadOnly }
  });
  return data;
}

/**
 * Get unread notification count
 */
export async function getUnreadCount(): Promise<number> {
  const { data} = await client.get<UnreadCountResponse>('/notifications/store/unread-count');
  return data.unread_count;
}

/**
 * Mark a notification as read
 */
export async function markNotificationAsRead(notificationId: number): Promise<void> {
  await client.patch(`/notifications/store/${notificationId}/read`);
}

/**
 * Mark all notifications as read
 */
export async function markAllNotificationsAsRead(): Promise<void> {
  await client.post('/notifications/store/mark-all-read');
}

/**
 * Delete a notification
 */
export async function deleteNotification(notificationId: number): Promise<void> {
  await client.delete(`/notifications/store/${notificationId}`);
}