import { useState, useEffect } from 'react';
import { getNotificationPreferences, updateNotificationPreferences, type NotificationPreferences } from '../api/notifications';
import { notify } from '../lib/notify';
import { requestNotificationPermission, initializeNotifications } from '../lib/notifications';
import CustomTimePicker from '../components/CustomTimePicker';
import '../styles/notification-settings.css';

export default function NotificationSettings() {
  const [preferences, setPreferences] = useState<NotificationPreferences | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [browserPermission, setBrowserPermission] = useState<NotificationPermission>('default');

  useEffect(() => {
    loadPreferences();
    checkBrowserPermission();
  }, []);

  async function loadPreferences() {
    try {
      const prefs = await getNotificationPreferences();
      setPreferences(prefs);
    } catch {
      notify.error('Failed to load preferences', 'Please try again later');
    } finally {
      setLoading(false);
    }
  }

  function checkBrowserPermission() {
    if ('Notification' in window) {
      setBrowserPermission(Notification.permission);
    }
  }

  async function handleTogglePushNotifications() {
    if (!preferences) return;

    // If user wants to enable push notifications but hasn't granted browser permission
    if (!preferences.push_enabled && browserPermission !== 'granted') {
      const permission = await requestNotificationPermission();
      if (permission === 'granted') {
        await initializeNotifications();
        setBrowserPermission('granted');
        await updatePreference({ push_enabled: true });
      } else {
        notify.error('Permission denied', 'Please enable notifications in your browser settings');
      }
    } else {
      await updatePreference({ push_enabled: !preferences.push_enabled });
    }
  }

  async function updatePreference(updates: Partial<NotificationPreferences>) {
    if (!preferences) return;

    setSaving(true);
    try {
      const updated = await updateNotificationPreferences(updates);
      setPreferences(updated);
      notify.success('Preferences updated');
    } catch {
      notify.error('Failed to save', 'Please try again');
    } finally {
      setSaving(false);
    }
  }

  async function handleEventPreferenceChange(event: string, channel: 'email' | 'push', enabled: boolean) {
    if (!preferences) return;

    const newEventPrefs = {
      ...preferences.event_preferences,
      [event]: {
        ...preferences.event_preferences[event],
        [channel]: enabled
      }
    };

    await updatePreference({ event_preferences: newEventPrefs });
  }

  if (loading) {
    return (
      <div className="notification-settings">
        <div className="settings-loading">
          <div className="spinner-large" />
          <p>Loading preferences...</p>
        </div>
      </div>
    );
  }

  if (!preferences) {
    return (
      <div className="notification-settings">
        <div className="settings-error">
          <p>Failed to load notification preferences</p>
          <button className="btn btn-primary" onClick={loadPreferences}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="notification-settings">
      {/* Header */}
      <header className="settings-header">
        <div className="settings-header-content">
          <div className="settings-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
          </div>
          <div>
            <h1 className="settings-title">Notification Preferences</h1>
            <p className="settings-subtitle">Manage how and when you receive notifications</p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="settings-content">
        {/* Notification Channels */}
        <section className="settings-section">
          <div className="section-header">
            <h2 className="section-title">Notification Channels</h2>
            <p className="section-description">Choose how you want to receive notifications</p>
          </div>

          <div className="settings-card">
            {/* Email Toggle */}
            <div className="setting-item">
              <div className="setting-info">
                <div className="setting-label-group">
                  <svg className="setting-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="2" y="4" width="20" height="16" rx="2" />
                    <path d="m2 7 10 6 10-6" />
                  </svg>
                  <div>
                    <div className="setting-label">Email Notifications</div>
                    <div className="setting-hint">Receive notifications via email</div>
                  </div>
                </div>
              </div>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={preferences.email_enabled}
                  onChange={(e) => updatePreference({ email_enabled: e.target.checked })}
                  disabled={saving}
                />
                <span className="toggle-slider" />
              </label>
            </div>

            {/* Push Notifications Toggle */}
            <div className="setting-item">
              <div className="setting-info">
                <div className="setting-label-group">
                  <svg className="setting-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
                    <path d="M13.73 21a2 2 0 0 1-3.46 0" />
                  </svg>
                  <div>
                    <div className="setting-label">Push Notifications</div>
                    <div className="setting-hint">
                      {browserPermission === 'denied'
                        ? 'Browser permission denied - enable in browser settings'
                        : 'Receive notifications in your browser'}
                    </div>
                  </div>
                </div>
              </div>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={preferences.push_enabled && browserPermission === 'granted'}
                  onChange={handleTogglePushNotifications}
                  disabled={saving || browserPermission === 'denied'}
                />
                <span className="toggle-slider" />
              </label>
            </div>
          </div>
        </section>

        {/* Event Preferences */}
        <section className="settings-section">
          <div className="section-header">
            <h2 className="section-title">Event Preferences</h2>
            <p className="section-description">Customize which events trigger notifications</p>
          </div>

          <div className="settings-card">
            <div className="event-preferences-table">
              <div className="table-header">
                <div className="table-header-cell event-col">Event Type</div>
                <div className="table-header-cell channel-col">Email</div>
                <div className="table-header-cell channel-col">Push</div>
              </div>

              {Object.entries(preferences.event_preferences).map(([eventType, channels]) => (
                <div key={eventType} className="table-row">
                  <div className="event-name">
                    <svg className="event-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" />
                      <path d="M12 6v6l4 2" />
                    </svg>
                    {formatEventName(eventType)}
                  </div>
                  <div className="channel-toggle">
                    <label className="checkbox-wrapper">
                      <input
                        type="checkbox"
                        checked={channels.email}
                        onChange={(e) => handleEventPreferenceChange(eventType, 'email', e.target.checked)}
                        disabled={saving || !preferences.email_enabled}
                      />
                      <span className="checkbox-custom" />
                    </label>
                  </div>
                  <div className="channel-toggle">
                    <label className="checkbox-wrapper">
                      <input
                        type="checkbox"
                        checked={channels.push}
                        onChange={(e) => handleEventPreferenceChange(eventType, 'push', e.target.checked)}
                        disabled={saving || !preferences.push_enabled}
                      />
                      <span className="checkbox-custom" />
                    </label>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Severity Filter */}
        <section className="settings-section">
          <div className="section-header">
            <h2 className="section-title">Severity Filter</h2>
            <p className="section-description">Set minimum severity level for notifications</p>
          </div>

          <div className="settings-card">
            <div className="severity-selector">
              {(['low', 'medium', 'high'] as const).map((severity) => (
                <label key={severity} className={`severity-option ${preferences.min_severity === severity ? 'active' : ''}`}>
                  <input
                    type="radio"
                    name="severity"
                    value={severity}
                    checked={preferences.min_severity === severity}
                    onChange={() => updatePreference({ min_severity: severity })}
                    disabled={saving}
                  />
                  <span className="severity-label">
                    <span className={`severity-badge severity-${severity}`} />
                    {severity.charAt(0).toUpperCase() + severity.slice(1)}
                  </span>
                  <svg className="severity-check" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                    <path d="M20 6 9 17l-5-5" />
                  </svg>
                </label>
              ))}
            </div>
          </div>
        </section>

        {/* Quiet Hours */}
        <section className="settings-section">
          <div className="section-header">
            <h2 className="section-title">Quiet Hours</h2>
            <p className="section-description">Pause notifications during specific hours</p>
          </div>

          <div className="settings-card">
            <div className="setting-item">
              <div className="setting-info">
                <div className="setting-label-group">
                  <svg className="setting-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                  </svg>
                  <div>
                    <div className="setting-label">Enable Quiet Hours</div>
                    <div className="setting-hint">Suppress notifications during specified hours</div>
                  </div>
                </div>
              </div>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={preferences.quiet_hours?.enabled || false}
                  onChange={(e) => updatePreference({
                    quiet_hours: {
                      ...preferences.quiet_hours,
                      enabled: e.target.checked
                    }
                  })}
                  disabled={saving}
                />
                <span className="toggle-slider" />
              </label>
            </div>

            {preferences.quiet_hours?.enabled && (
              <div className="quiet-hours-config">
                <div className="time-inputs">
                  <CustomTimePicker
                    label="Start Time"
                    value={preferences.quiet_hours?.start || '22:00'}
                    onChange={(value) => updatePreference({
                      quiet_hours: {
                        ...preferences.quiet_hours,
                        start: value
                      }
                    })}
                    disabled={saving}
                  />
                  <CustomTimePicker
                    label="End Time"
                    value={preferences.quiet_hours?.end || '08:00'}
                    onChange={(value) => updatePreference({
                      quiet_hours: {
                        ...preferences.quiet_hours,
                        end: value
                      }
                    })}
                    disabled={saving}
                  />
                </div>
                <p className="quiet-hours-hint">
                  Notifications will be paused from {preferences.quiet_hours?.start || '22:00'} to {preferences.quiet_hours?.end || '08:00'} (UTC)
                </p>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

function formatEventName(eventType: string): string {
  return eventType
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}